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
