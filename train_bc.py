from dataclasses import dataclass, field
import os
import sys
import pprint
from collections import namedtuple, defaultdict
import json
import yaml

import h5py
import pyrallis
import numpy as np
import torch

import common_utils
from bc.bc_policy import BcPolicy, BcPolicyConfig
from bc.action_rnn_policy import ARnnPolicy, ARnnPolicyConfig
from robosuite_wrapper import PixelRobosuite
from evaluate import run_eval, run_eval_mp


Batch = namedtuple("Batch", ["obs", "action"])


@dataclass
class DatasetConfig:
    path: str
    rl_camera: str = "robot0_eye_in_hand"
    num_data: int = -1
    max_len: int = -1
    eval_episode_len: int = 300

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")


class RobomimicDataset:
    def __init__(self, cfg: DatasetConfig):
        config_path = os.path.join(os.path.dirname(cfg.path), "env_cfg.json")
        self.env_config = json.load(open(config_path, "r"))
        self.task_name = self.env_config["env_name"]
        self.robot = self.env_config["env_kwargs"]["robots"]
        self.cfg = cfg

        self.data = []
        datafile = h5py.File(cfg.path)
        num_episode: int = len(list(datafile["data"].keys()))  # type: ignore
        print(f"Raw Dataset size (#episode): {num_episode}")

        self.ctrl_delta = self.env_config["env_kwargs"]["controller_configs"]["control_delta"]

        self.idx2entry = []  # store idx -> (episode_idx, timestep_idx)
        episode_lens = []
        all_actions = []  # for logging purpose
        for episode_id in range(num_episode):
            if cfg.num_data > 0 and len(episode_lens) >= cfg.num_data:
                break

            episode_tag = f"demo_{episode_id}"
            episode = datafile[f"data/{episode_tag}"]
            actions = np.array(episode["actions"]).astype(np.float32)  # type: ignore
            actions = torch.from_numpy(actions)
            all_actions.append(actions)
            episode_data: dict = {"action": actions}

            for camera in self.cfg.rl_cameras:
                obses: np.ndarray = episode[f"obs/{camera}_image"]  # type: ignore
                assert obses.shape[0] == actions.shape[0]
                episode_data[camera] = obses

            episode_len = actions.shape[0]
            if self.cfg.max_len > 0 and episode_len > self.cfg.max_len:
                print(f"removing {episode_tag} because it is too long {episode_len}")
                continue
            episode_lens.append(episode_len)

            # convert the data to list of dict
            episode_entries = []
            for i in range(episode_len):
                entry = {"action": episode_data["action"][i]}
                if self.env_config["env_kwargs"]["controller_configs"]["control_delta"]:
                    assert entry["action"].min() >= -1
                    assert entry["action"].max() <= 1

                for camera in cfg.rl_cameras:
                    entry[camera] = torch.from_numpy(episode_data[camera][i])

                self.idx2entry.append((len(self.data), len(episode_entries)))
                episode_entries.append(entry)
            self.data.append(episode_entries)
        datafile.close()

        self.obs_shape = self.data[-1][-1][cfg.rl_cameras[0]].size()
        self.action_dim = self.data[-1][-1]["action"].size()[0]
        print(f"Dataset size: {len(self.data)} episodes, {len(self.idx2entry)} steps")
        print(f"average length {np.mean(episode_lens):.1f}")
        print(f"obs shape:", self.obs_shape)
        all_actions = torch.cat(all_actions, dim=0)
        action_mins = all_actions.min(dim=0)[0]
        action_maxs = all_actions.max(dim=0)[0]
        for i in range(self.action_dim):
            print(f"action dim {i}: [{action_mins[i].item():.2f}, {action_maxs[i].item():.2f}]")

        self.env_params = dict(
            env_name=self.task_name,
            robots=self.robot,
            episode_length=cfg.eval_episode_len,
            reward_shaping=False,
            image_size=224,
            rl_image_size=self.obs_shape[-1],
            camera_names=cfg.rl_cameras,
            rl_cameras=cfg.rl_cameras,
            device="cuda",
            ctrl_delta=bool(self.env_config["env_kwargs"]["controller_configs"]["control_delta"]),
        )
        self.env = PixelRobosuite(**self.env_params)
        print(common_utils.wrap_ruler("config when the data was collected"))
        # check controller configs
        ref_controller_cfg = self.env_config["env_kwargs"]["controller_configs"]
        ref_controller_cfg["damping_ratio"] = ref_controller_cfg["damping"]
        ref_controller_cfg.pop("damping")
        ref_controller_cfg["damping_ratio_limits"] = ref_controller_cfg["damping_limits"]
        ref_controller_cfg.pop("damping_limits")
        # pprint.pprint(self.env.ctrl_config)
        # pprint.pprint(self.env_config["env_kwargs"]["controller_configs"])
        assert ref_controller_cfg == self.env.ctrl_config

        self.env_config["env_kwargs"].pop("controller_configs")
        pprint.pprint(self.env_config["env_kwargs"])
        assert self.env.env.control_freq == self.env_config["env_kwargs"]["control_freq"]
        print(common_utils.wrap_ruler(""))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _convert_to_batch(self, samples, device):
        batch = {}
        for k, v in samples.items():
            batch[k] = torch.stack(v).to(device)

        action = {"action": batch.pop("action")}
        ret = Batch(obs=batch, action=action)
        return ret

    def sample_bc(self, batchsize, device):
        indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.idx2entry[idx]
            entry = self.data[episode_idx][step_idx]
            for k, v in entry.items():
                samples[k].append(v)

        return self._convert_to_batch(samples, device)

    def _stack_actions(self, idx, begin, action_len):
        """stack actions in [begin, end)"""
        episode_idx, step_idx = self.idx2entry[idx]
        episode = self.data[episode_idx]

        actions = []
        valid_actions = []
        for action_idx in range(begin, begin + action_len):
            if action_idx < 0:
                actions.append(torch.zeros_like(episode[step_idx]["action"]))
                valid_actions.append(0)
            elif action_idx < len(episode):
                actions.append(episode[action_idx]["action"])
                valid_actions.append(1)
            else:
                actions.append(torch.zeros_like(actions[-1]))
                valid_actions.append(0)

        valid_actions = torch.tensor(valid_actions, dtype=torch.float32)
        actions = torch.stack(actions, dim=0)
        return actions, valid_actions

    def sample_arnn(self, batchsize, action_cond_horizon, action_pred_horizon, device):
        indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.idx2entry[idx]
            entry = self.data[episode_idx][step_idx]

            if action_cond_horizon:
                cond_actions, _ = self._stack_actions(
                    idx, step_idx - action_cond_horizon, action_cond_horizon
                )
                samples["cond_action"].append(cond_actions)

            actions, valid_actions = self._stack_actions(idx, step_idx - 1, action_pred_horizon + 1)
            input_actions = actions[:-1]
            input_actions[0] = 0
            target_actions = actions[1:]
            assert torch.equal(target_actions[0], entry["action"])

            samples["input_action"].append(input_actions)
            samples["valid_target"].append(valid_actions[1:])
            for k, v in entry.items():
                if k == "action":
                    samples[k].append(target_actions)
                else:
                    samples[k].append(v)

        return self._convert_to_batch(samples, device)

    def sample_diffusion(self, batchsize, distance, device):
        cond_indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for cond_idx in cond_indices:
            episode_idx, cond_step_idx = self.idx2entry[cond_idx]
            cond = self.data[episode_idx][cond_step_idx]
            prev_cond_step_idx = max(0, cond_step_idx - distance)
            prev_cond = self.data[episode_idx][prev_cond_step_idx]
            pred_step_idx = min(len(self.data[episode_idx]) - 1, cond_step_idx + distance)
            pred = self.data[episode_idx][pred_step_idx]

            combined = stack_dict([prev_cond, cond, pred])
            for k, v in combined.items():
                samples[k].append(v)

        return self._convert_to_batch(samples, device)

    def sample_inv_dyna(self, batchsize, distance, device):
        cond_indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for cond_idx in cond_indices:
            episode_idx, cond_step_idx = self.idx2entry[cond_idx]
            cond = self.data[episode_idx][cond_step_idx]
            pred_step_idx = min(len(self.data[episode_idx]) - 1, cond_step_idx + distance)
            pred = self.data[episode_idx][pred_step_idx]

            combined = stack_dict([cond, pred])
            actions, valid_actions = self._stack_actions(cond_idx, cond_step_idx, distance)
            for k, v in combined.items():
                if k == "action":
                    samples[k].append(actions)
                else:
                    samples[k].append(v)
            samples["valid_target"].append(valid_actions)

        return self._convert_to_batch(samples, device)


def stack_dict(list_dict):
    dict_list = defaultdict(list)
    for d in list_dict:
        for k, v in d.items():
            dict_list[k].append(v)

    stacked = {}
    for k, v in dict_list.items():
        stacked[k] = torch.stack(v, dim=0)
    return stacked


@dataclass
class MainConfig:
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(""))
    bc_policy: BcPolicyConfig = field(default_factory=lambda: BcPolicyConfig())
    arnn_policy: ARnnPolicyConfig = field(default_factory=lambda: ARnnPolicyConfig())
    seed: int = 1
    load_model: str = "none"
    method: str = "bc"  # bc/arnn/atransformer
    # training
    num_epoch: int = 20
    epoch_len: int = 10000
    batch_size: int = 256
    lr: float = 1e-4
    grad_clip: float = 5
    ema_method: str = "none"  # none, simple, complex
    ema_tau: float = 0.01
    cosine_schedule: int = 0
    lr_warm_up_steps: int = 0
    # log
    num_eval_episode: int = 50
    save_dir: str = "exps/bc/run1"
    use_wb: int = 0
    # to be overwritten by run() to facilitate model loading
    task_name: str = ""
    robots: list[str] = field(default_factory=lambda: [])
    image_size: int = -1
    rl_image_size: int = -1

    @property
    def wb_exp(self):
        return None if not self.use_wb else self.save_dir.split("/")[-2]

    @property
    def wb_run(self):
        return None if not self.use_wb else self.save_dir.split("/")[-1]

    @property
    def wb_group(self):
        if not self.use_wb:
            return None
        else:
            return "_".join([w for w in self.wb_run.split("_") if "seed" not in w])  # type: ignore

    @property
    def cfg_path(self):
        return os.path.join(self.save_dir, "cfg.yaml")


def run(cfg: MainConfig, policy):
    dataset = RobomimicDataset(cfg.dataset)
    cfg.task_name = dataset.task_name
    cfg.robots = dataset.robot
    cfg.image_size = dataset.env_params["image_size"]
    cfg.rl_image_size = dataset.env_params["rl_image_size"]

    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
    print(common_utils.wrap_ruler("config"))
    with open(cfg.cfg_path, "r") as f:
        print(f.read(), end="")
    cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

    if policy is None:
        if cfg.method == "arnn":
            policy = ARnnPolicy(
                dataset.obs_shape,
                dataset.action_dim,
                dataset.cfg.rl_cameras,
                cfg.arnn_policy,
            )
        elif cfg.method == "bc":
            policy = BcPolicy(
                dataset.obs_shape,
                dataset.action_dim,
                dataset.cfg.rl_cameras,
                cfg.bc_policy,
            )
        elif cfg.method == "diffusion":
            # TODO: change this to diffusion model
            policy = BcPolicy(
                dataset.obs_shape,
                dataset.action_dim,
                dataset.cfg.rl_cameras,
                cfg.bc_policy,
            )
    policy = policy.to("cuda")
    print(common_utils.wrap_ruler("policy weights"))
    print(policy)

    ema_policy = None
    if cfg.ema_method == "complex":
        ema_policy = common_utils.EMA(policy, power=3 / 4)
    elif cfg.ema_method == "simple":
        ema_policy = common_utils.SimpleEMA(policy, cfg.ema_tau)

    common_utils.count_parameters(policy)
    print("Using Adam optimzer")
    optim = torch.optim.Adam(policy.parameters(), cfg.lr)

    stat = common_utils.MultiCounter(
        cfg.save_dir,
        bool(cfg.use_wb),
        wb_exp_name=cfg.wb_exp,
        wb_run_name=cfg.wb_run,
        wb_group_name=cfg.wb_group,
        config=cfg_dict,
    )

    saver = common_utils.TopkSaver(cfg.save_dir, 3)
    stopwatch = common_utils.Stopwatch()
    best_score = 0
    optim_step = 0
    for epoch in range(cfg.num_epoch):
        stopwatch.reset()

        for _ in range(cfg.epoch_len):
            with stopwatch.time("sample"):
                if cfg.method == "arnn":
                    batch = dataset.sample_arnn(
                        cfg.batch_size,
                        cfg.arnn_policy.cond_horizon,
                        cfg.arnn_policy.pred_horizon,
                        "cuda:0",
                    )
                elif cfg.method == "diffusion":
                    batch = dataset.sample_diffusion(cfg.batch_size, 4, "cuda:0")
                    for k, v in batch.obs.items():
                        print(k, v.size())
                    assert False
                else:
                    batch = dataset.sample_bc(cfg.batch_size, "cuda:0")

            with stopwatch.time("train"):
                loss = policy.loss(batch)

                optim.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(  # type: ignore
                    policy.parameters(), max_norm=cfg.grad_clip
                )
                optim.step()
                stat["train/loss"].append(loss.item())
                stat["train/grad_norm"].append(grad_norm.item())
                optim_step += 1
                if ema_policy is not None:
                    decay = ema_policy.step(policy, optim_step=optim_step)
                    stat["other/decay"].append(decay)

        epoch_time = stopwatch.elapsed_time_since_reset
        with stopwatch.time("eval"):
            seed = epoch * 1991991991 % 9997
            eval_policy = policy if ema_policy is None else ema_policy.stable_model
            scores = evaluate(eval_policy, dataset, seed=seed, num_game=cfg.num_eval_episode)
            score = float(np.mean(scores))
            saved = saver.save(eval_policy.state_dict(), score)

        best_score = max(best_score, score)
        stat["score"].append(score)
        stat["score(best)"].append(best_score)
        stat["other/speed"].append(cfg.epoch_len / epoch_time)
        stat.summary(epoch)
        stopwatch.summary()
        if saved:
            print("model saved!")

    # eval the last checkpoint
    scores = evaluate(policy, dataset, num_game=50, seed=1)
    stat["last_ckpt_score"].append(np.mean(scores))
    # eval the best performing model again
    best_model = saver.get_best_model()
    policy.load_state_dict(torch.load(best_model))
    scores = evaluate(policy, dataset, num_game=50, seed=1)
    stat["final_score"].append(np.mean(scores))
    stat.summary(cfg.num_epoch)

    # quit!
    assert False


def evaluate(policy, dataset: RobomimicDataset, seed, num_game):
    if isinstance(policy, ARnnPolicy):
        scores = run_eval(dataset.env, policy, num_game=num_game, seed=seed, verbose=False)
    else:
        scores = run_eval_mp(
            dataset.env_params, policy, num_game=50, seed=seed, num_proc=10, verbose=False
        )
    return scores


# function to load bc models
def load_model(weight_file, device):
    cfg_path = os.path.join(os.path.dirname(weight_file), f"cfg.yaml")
    print(common_utils.wrap_ruler("config of loaded agent"))
    with open(cfg_path, "r") as f:
        print(f.read(), end="")
    print(common_utils.wrap_ruler(""))

    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    env_params = dict(
        env_name=cfg.task_name,
        robots=cfg.robots,
        episode_length=cfg.dataset.eval_episode_len,
        reward_shaping=False,
        image_size=cfg.image_size,
        rl_image_size=cfg.rl_image_size,
        camera_names=cfg.dataset.rl_cameras,
        rl_cameras=cfg.dataset.rl_cameras,
        device=device,
    )

    env = PixelRobosuite(**env_params)  # type: ignore
    print(f"observation shape: {env.observation_shape}")

    if cfg.method == "arnn":
        policy = ARnnPolicy(
            env.observation_shape,
            env.action_dim,
            env.rl_cameras,
            cfg.arnn_policy,
        )
    elif cfg.method == "bc":
        policy = BcPolicy(
            env.observation_shape,
            env.action_dim,
            env.rl_cameras,
            cfg.bc_policy,
        )
    else:
        assert False

    policy.load_state_dict(torch.load(weight_file))
    return policy.to(device), env, env_params


if __name__ == "__main__":
    import rich.traceback

    # make logging more beautiful
    rich.traceback.install()
    torch.set_printoptions(linewidth=100)

    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    common_utils.set_all_seeds(cfg.seed)
    log_path = os.path.join(cfg.save_dir, "train.log")
    sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)

    if cfg.load_model is not None and cfg.load_model != "none":
        policy = load_model(cfg.load_model, "cuda")[0]
    else:
        policy = None
    run(cfg, policy=policy)
