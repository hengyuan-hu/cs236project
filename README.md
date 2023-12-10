### Install packages (maybe incomplete)

```bash
pip install -r requirements.txt
```

### Dataset
The dataset is located at:
```
/iliad/u/hengyuan/vlm_robot/faster-q/data/robomimic/square/processed_data96.hdf5
```

Copy the dataset to your folder
```
mkdir -p data/robomimic
cp -r /iliad/u/hengyuan/vlm_robot/faster-q/data/robomimic/square data/robomimic
```

### Run

Before running any code, source the `set_env.sh` to add current folder to `PYTHONPATH`
```bash
source set_env.sh
```

Finally, train a bc model
```
# --use_wb 0 -> do not use wandb, you can turn it on to track learning curves
python train_bc.py --config_path cfgs/bc/square.yaml --use_wb 0
```
train a action-arnn autoregressive action prediction model
```
python train_bc.py --config_path cfgs/bc/square_arnn.yaml --use_wb 0
```

### Train an unconditional diffusion model

```
# unconditional training
CUDA_VISIBLE_DEVICES=3 python diffusion.py --batch_size 32 --camera agentview
# conditional training
CUDA_VISIBLE_DEVICES=4 python diffusion.py --batch_size 32 --camera agentview --conditional --save_dir exps/diffusion_conditional
```

Sample the diffusion model conditionally

```
# for unconditionally pretrained
CUDA_VISIBLE_DEVICES=3 python diffusion.py --batch_size 10 --viz_ckpt robomimic_it_hand_1000_0.13.ckpt --camera agentview
# for conditionally trained
CUDA_VISIBLE_DEVICES=5 python diffusion.py --batch_size 2 --viz_ckpt exps/diffusion_conditional/robomimic_it_hand_4000_0.02.ckpt --camera agentview --conditional --batch_size 10
```
Note: Replace the checkpoint with your checkpoint. After running the code, you can find the generated images in outputs/ folder.


### for VQVAE
```shell
# clone the submodule
git submodule init
git submodule update
# remember to run set_env to add stuff to PYTHONPATH
source set_env.sh

# evaluation
python evaluate/eval_vqvae.py
```