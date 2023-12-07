import os
import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam

import torch
from train_bc import RobomimicDataset, DatasetConfig
import numpy as np
from PIL import Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--viz_ckpt', type=str, default=None)
parser.add_argument('--conditional', action='store_true')
parser.add_argument('--camera', type=str, default='robot0_eye_in_hand', help='agentview')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--sample_steps', type=int, default=20, help='diffusion sampling steps. more step (20~100 +) better quality, fewer steps (5-20) faster sampling')
args = parser.parse_args()


cfg = DatasetConfig(
    # path='data/robomimic/square/processed_data96.hdf5',
    path='../cs236project-4188536d20d2c993039f620fa3451aadec12d1e8/data/robomimic/square/processed_data96.hdf5',
    rl_camera='agentview+robot0_eye_in_hand',
    num_data=-1
)
dataset = RobomimicDataset(cfg)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data = dataset.sample_diffusion(batchsize=32, distance=4, device=device)


def get_model():
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=3*3, in_channels=3*3, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

@torch.no_grad()
def sample_iadb(model, x0, nb_step, condition=None):
    x_alpha = x0
    # breakpoint()
    d_condition = condition.to(x0.device) - x0[:,:condition.shape[1]]
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        if condition is not None:
            d = torch.cat([d_condition, d[:,d_condition.shape[1]:]], dim=1)
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

model = get_model()
model = model.to(device)

if args.viz_ckpt is not None:
    os.makedirs('outputs', exist_ok=True)
    state_dict = torch.load(args.viz_ckpt)
    model.load_state_dict(state_dict)
    bz = args.batch_size  # sample 100
    data = dataset.sample_diffusion(batchsize=bz, distance=4, device=device)[0][args.camera].flatten(1, 2) / 255
    data = (data*2)-1
    condition = data[:,:6]  # condition on first two frames. Unconditional if set None
    sample = (sample_iadb(model, torch.randn((bz, 3*3, 96, 96)).to(device), nb_step=20,
                        condition=condition) * 0.5) + 0.5
        # sample has shape [1, 3*3, 96, 96]
        # sample = sample.view(bz, 3, 3, 96, 96).permute(1, 3, 0, 4, 2).flatten(2,3).detach().cpu().numpy()
    sample = sample.view(bz, 3, 3, 96, 96)
    for b in range(bz):
        for i in range(3):
            img = sample[b, i].permute(1,2,0)
            if i == 2:
                Image.fromarray(np.array(img.detach().cpu().numpy() * 255, dtype=np.uint8)).save(f'outputs/b{b}_gen_f{i+1}.png')
            elif i == 1:
                Image.fromarray(np.array(img.detach().cpu().numpy() * 255, dtype=np.uint8)).save(f'outputs/b{b}_ori_f{i+1}.png')
            elif i == 0:
                Image.fromarray(np.array(img.detach().cpu().numpy() * 255, dtype=np.uint8)).save(f'outputs/b{b}_ori_f{i+1}.png')

    # sample = sample.permute(0, 3, 1, 4, 2).flatten(2,3).flatten(0,1).detach().cpu().numpy()

    # arr = np.array(sample * 255, dtype=np.uint8)
    # Image.fromarray(arr).save('outputs/tmp_gen.png')
    # video = [Image.fromarray(a) for a in arr]
    # video[0].save('tmp_gen.gif', save_all=True, append_images=video[1:], loop=0, duration=100)
    exit()
    # torchvision.utils.save_image(sample, f'export_{str(nb_iter).zfill(8)}.png')



optimizer = Adam(model.parameters(), lr=1e-4)
nb_iter = 0
from tqdm import tqdm

print('Start training')
for current_epoch in tqdm(range(10000)):
    # print(f'Iter {current_epoch}')
    # for i, data in enumerate(dataloader):
    with torch.no_grad():
        data = dataset.sample_diffusion(batchsize=args.batch_size, distance=4, device=device)[0][args.camera].flatten(1, 2) / 255
        # flatten by batch and frames

    x1 = (data*2)-1
    x0 = torch.randn_like(x1)
    bs = x0.shape[0]

    alpha = torch.rand(bs, device=device)
    x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0
    
    d = model(x_alpha, alpha)['sample']
    loss = torch.mean((d - (x1-x0))**2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    nb_iter += 1

    if nb_iter % 1000 == 0:
        with torch.no_grad():
            print(f'Save export {nb_iter}')
            torch.save(model.state_dict(), f'robomimic_it_hand_{nb_iter}_{loss.item():.2f}.ckpt')