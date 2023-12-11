import argparse
import os
import sys
import torch
from torch.optim import Adam
import numpy as np
from diffusers import UNet2DModel

from train_bc import RobomimicDataset, DatasetConfig
import common_utils


def get_model():
    block_out_channels = (128, 128, 256, 256, 512, 512)
    down_block_types = (
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types = (
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    return UNet2DModel(
        block_out_channels=block_out_channels,
        out_channels=3 * 3,
        in_channels=3 * 3,
        up_block_types=up_block_types,
        down_block_types=down_block_types,
        add_attention=True,
    )


@torch.no_grad()
def sample_iadb(model, x0, nb_step, condition=None, unconditionally_trained=True):
    x_alpha = x0
    condition = condition.to(x0.device)
    # breakpoint()
    if condition is not None and unconditionally_trained:
        d_condition = condition.to(x0.device) - x0[:, : condition.shape[1]]
    for t in range(nb_step):
        alpha_start = t / nb_step
        alpha_end = (t + 1) / nb_step
        if condition is not None and not unconditionally_trained:
            x_alpha = torch.cat([condition.to(x0), x_alpha[:, condition.shape[1] :]], dim=1)

        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))["sample"]
        if condition is not None and unconditionally_trained:
            d = torch.cat([d_condition, d[:, d_condition.shape[1] :]], dim=1)
        x_alpha = x_alpha + (alpha_end - alpha_start) * d

    return x_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="exps/diffusion")
    parser.add_argument("--use_wb", type=int, default=0)
    parser.add_argument("--camera", type=str, default="agentview", help="agentview")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--conditional", action="store_true", help="condition on the frames before last"
    )
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

    cfg = DatasetConfig(
        path="data/robomimic/square/processed_data96.hdf5",
        # path="data/robomimic/lift/processed_data96.hdf5",
        rl_camera="agentview",
        num_data=-1,
    )
    dataset = RobomimicDataset(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model()
    model = model.to(device)

    if args.viz_ckpt is not None:
        # os.makedirs("outputs", exist_ok=True)
        state_dict = torch.load(args.viz_ckpt)
        model.load_state_dict(state_dict)
        bz = args.batch_size  # sample 100

        data = dataset.sample_diffusion(batchsize=bz, distance=4, device=device)[0][args.camera]
        data = data.flatten(1, 2) / 255.0
        data = (data * 2) - 1
        condition = data[:, :6]  # condition on first two frames. Unconditional if set None
        sample = sample_iadb(
            model,
            torch.randn((bz, 3 * 3, 96, 96)).to(device),
            nb_step=100,
            condition=condition,
            unconditionally_trained=not args.conditional,
        )
        sample = sample * 0.5 + 0.5
        # sample has shape [1, 3*3, 96, 96]
        # sample = sample.view(bz, 3, 3, 96, 96).permute(1, 3, 0, 4, 2).flatten(2,3).detach().cpu().numpy()
        sample = sample.view(bz, 3, 3, 96, 96)

        fig, ax = common_utils.generate_grid(rows=10, cols=3)
        for b in range(bz)[:10]:
            for i in range(3):
                img = sample[b, i].permute(1, 2, 0).detach().cpu().numpy()
                img = np.clip(img, a_min=0, a_max=1)
                print(f">>> {b=}, {i=}", img.min(), img.max())
                ax[b][i].imshow(img)

        fig_path = os.path.join(os.path.dirname(args.viz_ckpt), "viz.png")
        print(f"saving to {fig_path}")
        fig.tight_layout()
        fig.savefig(fig_path)
        exit()

    optimizer = Adam(model.parameters(), lr=1e-4)
    nb_iter = 0
    from tqdm import tqdm

    print("Start training")
    for current_epoch in tqdm(range(10000)):
        # print(f'Iter {current_epoch}')
        # for i, data in enumerate(dataloader):
        with torch.no_grad():
            data = dataset.sample_diffusion(
                batchsize=args.batch_size, distance=4, device=device
            ).obs
            data = data[args.camera]
            data = data.flatten(1, 2) / 255

            # flatten by batch and frames

        x1 = (data * 2) - 1

        alpha = torch.rand(x1.shape[0], device=device)
        if args.conditional:
            x1_condition = x1[:, :6]  # conditioned on first two frames, three channel each
            x1_gen = x1[:, 6:]
            x0_gen = torch.randn_like(x1_gen)
            x_alpha = alpha.view(-1, 1, 1, 1) * x1_gen + (1 - alpha).view(-1, 1, 1, 1) * x0_gen
            d = model(torch.cat([x1_condition, x_alpha], dim=1), alpha)["sample"][:, 6:]
            loss = torch.mean((d - (x1_gen - x0_gen)) ** 2)
        else:
            x0 = torch.randn_like(x1)
            x_alpha = alpha.view(-1, 1, 1, 1) * x1 + (1 - alpha).view(-1, 1, 1, 1) * x0

            d = model(x_alpha, alpha)["sample"]
            loss = torch.mean((d - (x1 - x0)) ** 2)

        stat["loss"].append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        nb_iter += 1

        if nb_iter % 100 == 0:
            stat.summary(nb_iter)

        if nb_iter % 1000 == 0:
            save_path = os.path.join(
                args.save_dir, f"robomimic_it_hand_{nb_iter}_{loss.item():.2f}.ckpt"
            )
            print(f"Save export {nb_iter}")
            torch.save(model.state_dict(), save_path)
