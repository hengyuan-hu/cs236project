import pickle
import torch
import numpy as np
from robosuite_wrapper import PixelRobosuite
from common_utils import Recorder
from common_utils import ibrl_utils as utils

import torch
import torch.nn as nn
from vqvae.models.vqvae import VQVAE


d_model = 576  # Model dimension
nhead = 8  # Number of attention heads
num_layers = 6  # Number of decoder layers
dim_feedforward = 2048  # Feedforward dimension


class RandomShiftsAug:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class ConvInverseDynamicsModel(nn.Module):
    def __init__(self, numChannels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=numChannels, out_channels=20, kernel_size=(3, 3), stride=2
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=2000, out_features=500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=500)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=500, out_features=7)
        self.aug = RandomShiftsAug(4)

    def forward(self, x):
        # x = self.aug(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x


class PredictionInverseDynamicsModel(nn.Module):
    def __init__(self, vqvae_path, autoregressive_path, device="cuda"):
        super().__init__()
        self.device = torch.device(device)

        self.vqvae_encoder = torch.load(vqvae_path).eval().to(self.device)
        self.embedding_dim = self.vqvae_encoder.vector_quantization.e_dim
        self.n_codes = self.vqvae_encoder.vector_quantization.n_e

        self.prediction_model = torch.load(autoregressive_path).eval().to(self.device)
        self.seq_len = 2

        self.inverse_dynamics = ConvInverseDynamicsModel(6).to(self.device)

    def forward(self, img):
        # Image is batch, c, h, w
        with torch.no_grad():
            curr_embed, next_embed = self.get_current_and_next_embeddings(img)

        # Fuse embeddings along channel
        fusion = torch.cat((curr_embed, next_embed), dim=1)

        # Compute action
        action = self.inverse_dynamics(fusion)
        return action

    def act(self, obs: dict[str, torch.Tensor]):
        assert not self.training
        assert not self.prediction_model.training
        assert not self.inverse_dynamics.training
        image = obs["agentview"].unsqueeze(0).float().cuda() / 255.0
        action = self.forward(image).squeeze(0)
        return action.cpu()

    def get_current_and_next_embeddings(self, img):
        # Add context dim
        original_image = img
        img = img.unsqueeze(1)

        # Encode Image
        bsz, context, *image_dim = img.shape
        img = img.view(bsz * context, *image_dim)
        z_e = self.vqvae_encoder.encoder(img)
        z_e = self.vqvae_encoder.pre_quantization_conv(z_e)
        (
            embedding_loss,
            z_q,
            perplexity,
            _,
            min_encoding_indices,
        ) = self.vqvae_encoder.vector_quantization(z_e)
        curr_z_q = z_q
        embedding_idx = min_encoding_indices.view(bsz, context, -1)
        embedding_idx = embedding_idx.permute(1, 0, 2).contiguous()
        z_q = z_q.view(bsz, context, self.embedding_dim, 24, 24)
        z_q = z_q.permute(1, 0, 3, 4, 2).contiguous()
        targets = self.prediction_model.fc_in(z_q)
        targets = targets.view(context, bsz * 576, -1)

        # Add start embedding
        start_token = torch.zeros(targets.shape[1:]).unsqueeze(0).to(self.device)
        targets = torch.cat((start_token, targets), dim=0)
        memory = torch.zeros(*targets.shape).to(self.device)

        # Get prediction (codebook index)
        out = self.prediction_model(targets, memory)
        out = out.view(self.seq_len, bsz * 576, -1)
        out = out[-1]
        out_idxs = torch.argmax(out, dim=-1, keepdim=True)

        # Recover next embedding
        next_z_q = self.vqvae_encoder.vector_quantization.recover_embeddings(out_idxs)
        next_xhat = self.vqvae_encoder.decoder(next_z_q)
        curr_xhat = self.vqvae_encoder.decoder(curr_z_q)
        # plt.imshow((next_xhat[0]* 255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig('train_image.png')
        # return curr_z_q, next_z_q
        return original_image, next_xhat


class AutoRegressiveModel(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        embedding_dim,
        n_codes,
        seq_len,
        device="cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward),
            num_layers,
        ).to(self.device)
        self.fc_in = nn.Linear(embedding_dim, d_model, bias=False).to(self.device)
        self.fc_in.weight.data.normal_(std=0.02)
        self.fc_out = nn.Linear(d_model, n_codes, bias=False).to(self.device)
        self.mask = self.generate_square_subsequent_mask(seq_len, device)

    def forward(self, targets, memory):
        out = self.fc_out(self.decoder(targets, memory, tgt_mask=self.mask))
        return out

    def generate_square_subsequent_mask(self, sz: int, device: str = "cpu") -> torch.Tensor:
        """Generate the attention mask for causal decoding"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        ).to(device=device)
        return mask


def run_eval(
    env: PixelRobosuite,
    agent,
    num_game,
    seed,
    record_dir=None,
    verbose=True,
    save_frame=False,
    eval_mode=True,
):
    recorder = None if record_dir is None else Recorder(record_dir)

    scores = []
    lens = []
    with torch.no_grad(), utils.eval_mode(agent):
        for episode_idx in range(num_game):
            step = 0
            rewards = []
            np.random.seed(seed + episode_idx)
            obs, image_obs = env.reset()

            try:
                agent.reset()
            except AttributeError:
                pass

            terminal = False
            while not terminal:
                if recorder is not None:
                    recorder.add(image_obs)

                action = agent.act(obs).numpy()
                obs, reward, terminal, _, image_obs = env.step(action)
                rewards.append(reward)
                step += 1

            if verbose:
                print(
                    f"seed: {seed + episode_idx}, "
                    f"reward: {np.sum(rewards)}, len: {env.time_step}"
                )

            scores.append(np.sum(rewards))
            if scores[-1] > 0:
                lens.append(env.time_step)

            if recorder is not None:
                save_path = recorder.save(f"episode{episode_idx}", save_frame)
                reward_path = f"{save_path}.reward.pkl"
                print(f"saving reward to {reward_path}")
                pickle.dump(rewards, open(reward_path, "wb"))

    if verbose:
        print(f"num game: {len(scores)}, seed: {seed}, score: {np.mean(scores)}")
        print(f"average time for success games: {np.mean(lens)}")

    return scores


if __name__ == "__main__":
    import argparse
    import os
    import common_utils

    os.environ["MUJOCO_GL"] = "egl"
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--mode", type=str, default="bc", help="bc/rl")
    parser.add_argument("--num_game", type=int, default=10)
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--save_frame", type=int, default=0)
    parser.add_argument("--mp", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    common_utils.set_all_seeds(args.seed)

    model = torch.load("vqvae-weights/out_inverse_dynamics_3.pt")
    env = PixelRobosuite(
        "NutAssemblySquare",
        robots=["Panda"],
        episode_length=300,
    )

    scores = run_eval(
        env,
        model,
        args.num_game,
        args.seed,
        args.record_dir,
        save_frame=args.save_frame,
        verbose=args.verbose,
    )
    print(f"score: {np.mean(scores)}")
