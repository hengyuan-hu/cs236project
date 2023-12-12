import argparse
import os
import sys
import torch
from torch import nn
from torch.optim import Adam
import numpy as np

from train_bc import RobomimicDataset, DatasetConfig
import common_utils
from bc.resnet import ResNetEncoder, ResNetEncoderConfig
from bc.bc_policy import build_fc
import diffusion
from robosuite_wrapper import PixelRobosuite
from common_utils import ibrl_utils as utils
from common_utils import Recorder


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="exps/inv_dyna")
parser.add_argument("--use_wb", type=int, default=0)
parser.add_argument("--eval", type=int, default=0)
parser.add_argument("--env", type=str, default="lift")
parser.add_argument("--inv_dyna_path", type=str, default=None)
args = parser.parse_args()

common_utils.set_all_seeds(args.seed)
log_path = os.path.join(args.save_dir, "train.log")
sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)
stat = common_utils.MultiCounter(
    args.save_dir,
    bool(args.use_wb),
    wb_exp_name="cs236_diffusion",
    wb_run_name=args.save_dir.split("/")[-1],
)

if args.env == "lift":
    path="data/robomimic/lift/processed_data96.hdf5"
else:
    path="data/robomimic/square/processed_data96.hdf5"

cfg = DatasetConfig(
    path=path,
    rl_camera="agentview",
    num_data=-1,
)
dataset = RobomimicDataset(cfg)
device = "cuda" if torch.cuda.is_available() else "cpu"


class InverseDynamics(nn.Module):
    def __init__(self, num_action, action_dim):
        super().__init__()
        obs_shape = (6, 96, 96)
        resnet_cfg = ResNetEncoderConfig()
        self.encoder = ResNetEncoder(obs_shape, cfg=resnet_cfg)
        self.action_pred = build_fc(
            in_dim=self.encoder.repr_dim,
            hidden_dim=512,
            action_dim=num_action * action_dim,
            num_layer=3,
            layer_norm=True,
            dropout=0.0,
        )
        self.num_action = num_action
        self.action_dim = action_dim
        self.aug = common_utils.RandomShiftsAug(pad=4)
        self.diffusion_model = None

    def set_diffusion_model(self):
        self.diffusion_model = diffusion.get_model()
        self.diffusion_model.to("cuda")
        if args.env == "lift":
            model_path = "exps/lift_diffusion_conditional/robomimic_it_hand_10000_0.01.ckpt"
        else:
            model_path = "exps/square_diffusion_conditional/robomimic_it_hand_10000_0.01.ckpt"
        self.diffusion_model.load_state_dict(torch.load(model_path))

    def reset(self):
        self.past_obs = []

    def act(self, obs_dict: dict[str, torch.Tensor]):
        obs = obs_dict["agentview"]
        assert obs.dim() == 3
        obs = obs.unsqueeze(0)
        obs = obs / 255.0 * 2 - 1

        if len(self.past_obs) == 0:
            for _ in range(self.num_action):
                self.past_obs.append(obs)

        self.past_obs.append(obs)

        prev_cond = self.past_obs[-1 - self.num_action]
        curr_cond = self.past_obs[-1]
        cond = torch.cat([prev_cond, curr_cond], dim=1)
        sample: torch.Tensor = diffusion.sample_iadb(
            self.diffusion_model,
            torch.randn((1, 3 * 3, 96, 96)).to("cuda"),
            nb_step=100,
            condition=cond,
            unconditionally_trained=False,
        )
        sample = sample.view(1, 3, 3, 96, 96)
        sample = sample.clip(min=-1, max=1)
        image_pair = sample[:, 1:].flatten(1, 2)
        # print(">>> image pair size:", image_pair.size())
        # print(pred_img.size())

        # fig, ax = common_utils.generate_grid(rows=1, cols=3)
        # for i in range(3):
        #     img = sample[0, i].permute(1, 2, 0).detach().cpu().numpy()
        #     img = (img + 1) * 0.5
        #     img = np.clip(img, a_min=0, a_max=1)
        #     # print(f">>> {b=}, {i=}", img.min(), img.max())
        #     ax[i].imshow(img)
        # fig.savefig("eval_figs/test.png")
        # assert False

        action = self.forward(image_pair)
        action = action.view(self.num_action, self.action_dim)
        # print("action:", action.size())
        return action.cpu()

    def forward(self, obs: torch.Tensor):
        h = self.encoder(obs)
        action = self.action_pred(h)
        return action

    def loss(self, batch):
        image = batch.obs["agentview"]

        image = image.flatten(1, 2)
        image = self.aug(image.float())

        image = image / 255.0 * 2 - 1
        noise = (torch.rand((image.size(0), 3, 96, 96)) * 0.05).to("cuda")
        # add noise to the future frame
        image[:, 3:] += noise

        pred_action: torch.Tensor = self.forward(image).view(
            image.size(0), self.num_action, self.action_dim
        )
        target_action: torch.Tensor = batch.action["action"]
        valid_target = batch.obs["valid_target"]

        err = (target_action - pred_action).square().sum(2)
        err = (err * valid_target).sum(1) / valid_target.sum(1)
        return err.mean()


def train(dataset: RobomimicDataset, model: InverseDynamics, num_epoch, epoch_len):
    optim = Adam(model.parameters(), lr=1e-4)

    stat = common_utils.MultiCounter(
        args.save_dir,
        bool(args.use_wb),
        wb_exp_name="cs236_diffusion",
        wb_run_name=args.save_dir.split("/")[-1],
    )
    saver = common_utils.TopkSaver(args.save_dir, 3)

    for epoch in range(num_epoch):
        for _ in range(epoch_len):
            batch = dataset.sample_inv_dyna(256, 4, "cuda")
            loss = model.loss(batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            stat["loss"].append(loss.item())

        saver.save(model.state_dict(), -stat["loss"].mean())
        stat.summary(epoch)


def evaluate(
    agent,
    num_game,
    seed,
    record_dir=None,
    verbose=True,
):
    env_name = "Lift" if args.env == "lift" else "NutAssemblySquare"
    env = PixelRobosuite(
        env_name,
        robots=["Panda"],
        episode_length=300,
    )

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
                save_path = recorder.save(f"episode{episode_idx}", False)
                print(f"video saved to {save_path}")

    if verbose:
        print(f"num game: {len(scores)}, seed: {seed}, score: {np.mean(scores)}")
        print(f"average time for success games: {np.mean(lens)}")

    return scores


if args.eval:
    model = InverseDynamics(4, 7).cuda()
    model.load_state_dict(torch.load(args.inv_dyna_path))
    model.set_diffusion_model()
    record_dir = os.path.join(os.path.dirname(args.inv_dyna_path), "eval_vid")
    evaluate(model, args.eval, seed=1, record_dir=record_dir)
else:
    model = InverseDynamics(4, 7).cuda()
    train(dataset, model, 10, 10000)
