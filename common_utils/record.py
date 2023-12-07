import os
from collections import defaultdict

import imageio
import torch
import numpy as np


class Recorder:
    def __init__(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir
        self.frames = defaultdict(list)
        self.tensors = defaultdict(list)

    def add(self, camera_obses: dict[str, torch.Tensor]):
        for camera, obs in camera_obses.items():
            assert obs.dim() == 3 and obs.size(0) == 3
            assert obs.dtype == torch.uint8

            tensor = obs.cpu()
            self.tensors[camera].append(tensor)
            frame = tensor.permute([1, 2, 0]).numpy()
            self.frames[camera].append(frame)

    def save(self, name, save_images=False):
        for camera_name in self.frames:
            self._save(name, camera_name, save_images)
            break

        self.frames.clear()
        return f"{self.save_dir}/{name}"

    def _save(self, name, camera, save_images=False):
        path = os.path.join(self.save_dir, f"{name}-{camera}.gif")
        print(f"saving video to {path}")
        imageio.mimsave(path, self.frames[camera], duration=25)

        # tensor_path = os.path.join(self.save_dir, f"{name}-{camera}.pt")
        # print(f"saving tensor to {tensor_path}")
        # torch.save(self.tensors[camera], tensor_path)

        if save_images:
            for i, image in enumerate(self.frames[camera]):
                frame_dir = os.path.join(self.save_dir, f"{name}_frames")
                os.makedirs(frame_dir, exist_ok=True)
                frame_path = os.path.join(frame_dir, f"t{i}.png")
                imageio.imsave(frame_path, image)
        return path


if __name__ == "__main__":
    import numpy as np
    from robosuite_wrapper import PixelRobosuite, GOOD_CAMERAS

    recorder = Recorder("exps/rsuite/video_dev")
    env = PixelRobosuite("Lift", "Panda", 200, image_size=256, camera_names=GOOD_CAMERAS)
    _, image_obs = env.reset()
    recorder.add(image_obs)

    done = False
    while not done:
        action = np.random.randn(env.num_action)  # sample random action
        *_, image_obs = env.step(action)  # take action in the environment
        recorder.add(image_obs)
    recorder.save("video", False)
