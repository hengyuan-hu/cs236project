from collections import defaultdict

import torch
import robosuite
from robosuite import load_controller_config
import numpy as np
from common_utils import ibrl_utils as utils


# all avail views:
# 'frontview', 'birdview', --> too far for this task
# 'agentview', 'robot0_robotview', --> same
# 'sideview', 'robot0_eye_in_hand'
GOOD_CAMERAS = {
    "Lift": ["agentview", "sideview", "robot0_eye_in_hand"],
    "PickPlaceCan": ["agentview", "robot0_eye_in_hand"],
    "NutAssemblySquare": ["agentview", "robot0_eye_in_hand"],
}
DEFAULT_CAMERA = "agentview"


STATE_KEYS = {
    "Lift": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
    "PickPlaceCan": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
    "NutAssemblySquare": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
}
STATE_SHAPE = {"Lift": (19,), "PickPlaceCan": (23,), "NutAssemblySquare": (23,)}
PROP_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


class PixelRobosuite:
    def __init__(
        self,
        env_name,
        robots,
        episode_length,
        *,
        reward_shaping=False,
        image_size=96,
        rl_image_size=None,
        device="cuda",
        camera_names=[DEFAULT_CAMERA],
        rl_cameras=["agentview"],
        env_reward_scale=1.0,
        end_on_success=True,
        use_state=False,
        obs_stack=1,
        prop_stack=1,
        flip_image=True,  # only false if using with eval_with_init_state
        ctrl_delta=True,
    ):
        assert isinstance(camera_names, list)
        self.camera_names = camera_names
        self.ctrl_config = load_controller_config(default_controller="OSC_POSE")
        self.ctrl_config["control_delta"] = ctrl_delta
        self.env = robosuite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=self.ctrl_config,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            reward_shaping=reward_shaping,
            camera_names=self.camera_names,
            camera_heights=image_size,
            camera_widths=image_size,
            horizon=episode_length,
        )
        self.rl_cameras = rl_cameras if isinstance(rl_cameras, list) else [rl_cameras]
        self.image_size = image_size
        self.rl_image_size = rl_image_size or image_size
        self.env_reward_scale = env_reward_scale
        self.end_on_success = end_on_success
        self.use_state = use_state
        self.state_keys = STATE_KEYS[env_name]
        self.prop_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        self.obs_stack = obs_stack
        self.prop_stack = prop_stack
        self.flip_image = flip_image

        self.resize_transform = None
        if self.rl_image_size != self.image_size:
            self.resize_transform = utils.get_rescale_transform(self.rl_image_size)

        self.num_action = len(self.env.action_spec[0])
        self.observation_shape = (3 * obs_stack, rl_image_size, rl_image_size)
        self.state_shape: tuple[int] = (STATE_SHAPE[env_name][0] * obs_stack,)
        self.prop_shape: tuple[int] = (9 * prop_stack,)
        self.device = device
        self.reward_model = None

        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = True

        self.obs_stack = obs_stack
        self.past_obses = defaultdict(list)

    @property
    def action_dim(self):
        return self.num_action

    def set_reward_model(self, reward_model):
        self.reward_model = reward_model

    def _extract_images(self, obs):
        # assert self.frame_stack == 1, "frame stack not supported"

        high_res_images = {}
        rl_obs = {}

        if self.use_state:
            states = []
            for key in self.state_keys:
                if key == "object":
                    key = "object-state"
                states.append(obs[key])
            state = torch.from_numpy(np.concatenate(states).astype(np.float32))
            # first append, then concat
            self.past_obses["state"].append(state)
            rl_obs["state"] = utils.concat_obs(
                len(self.past_obses["state"]) - 1, self.past_obses["state"], self.obs_stack
            ).to(self.device)

        props = []
        for key in self.prop_keys:
            props.append(obs[key])
        prop = torch.from_numpy(np.concatenate(props).astype(np.float32))
        # first append, then concat
        self.past_obses["prop"].append(prop)
        rl_obs["prop"] = utils.concat_obs(
            len(self.past_obses["prop"]) - 1, self.past_obses["prop"], self.prop_stack
        ).to(self.device)

        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            image_obs = obs[image_key]
            if self.flip_image:
                image_obs = image_obs[::-1]
            image_obs = torch.from_numpy(image_obs.copy()).permute([2, 0, 1])

            # keep the high-res version for rendering
            high_res_images[camera_name] = image_obs
            if camera_name not in self.rl_cameras:
                continue

            rl_image_obs = image_obs
            if self.resize_transform is not None:
                # set the device here because transform is 5x faster on GPU
                rl_image_obs = self.resize_transform(rl_image_obs.to(self.device))
            # first append, then concat
            self.past_obses[camera_name].append(rl_image_obs)
            rl_obs[camera_name] = utils.concat_obs(
                len(self.past_obses[camera_name]) - 1,
                self.past_obses[camera_name],
                self.obs_stack,
            )

        return rl_obs, high_res_images

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
        self.past_obses.clear()

        obs = self.env.reset()
        rl_obs, high_res_images = self._extract_images(obs)

        if self.reward_model is not None:
            self.reward_model.reset()
        return rl_obs, high_res_images

    def step(self, action):
        assert isinstance(action, np.ndarray)
        if len(action.shape) == 1:
            action = np.expand_dims(action, axis=0)
        num_action = action.shape[0]

        rl_obs = {}
        high_res_images = {}
        reward = 0
        success = False
        terminal = False
        for i in range(num_action):
            self.time_step += 1
            obs, step_reward, terminal, _ = self.env.step(action[i])
            # NOTE: extract images every step for potential obs stacking
            # this is not efficient
            rl_obs, high_res_images = self._extract_images(obs)

            reward += step_reward
            self.episode_reward += step_reward

            if step_reward == 1:
                success = True
                if self.end_on_success:
                    terminal = True

            if terminal:
                break

        reward = reward * self.env_reward_scale
        self.terminal = terminal
        return rl_obs, reward, terminal, success, high_res_images


if __name__ == "__main__":
    from torchvision.utils import save_image

    env = PixelRobosuite("Lift", "Panda", 200, image_size=256, camera_names=GOOD_CAMERAS["Lift"])
    x = env.reset()[0][GOOD_CAMERAS["Lift"][0]].float() / 255
    print(x.dtype)
    print(x.shape)
    save_image(x, "test_env.png")
