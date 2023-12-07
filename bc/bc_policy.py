from dataclasses import dataclass, field
import torch
import torch.nn as nn

import common_utils
from common_utils import ibrl_utils as utils
from bc.multiview_encoder import MultiViewEncoder, MultiViewEncoderConfig


def build_fc(in_dim, hidden_dim, action_dim, num_layer, layer_norm, dropout):
    dims = [in_dim]
    dims.extend([hidden_dim for _ in range(num_layer)])

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if layer_norm == 2 and (i == num_layer - 1):
            layers.append(nn.LayerNorm(dims[i + 1]))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], action_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


@dataclass
class BcPolicyConfig:
    encoder: MultiViewEncoderConfig = field(default_factory=lambda: MultiViewEncoderConfig())
    hidden_dim: int = 1024
    num_layer: int = 1
    dropout: float = 0
    orth_init: int = 0


class BcPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim, rl_cameras, cfg: BcPolicyConfig):
        super().__init__()
        self.rl_cameras = rl_cameras
        self.cfg = cfg

        self.encoder = MultiViewEncoder(
            obs_shape=obs_shape,
            obs_horizon=1,
            prop_shape=0,
            rl_cameras=rl_cameras,
            use_prop=False,
            cfg=cfg.encoder,
        )

        self.policy = build_fc(
            in_dim=self.encoder.repr_dim,
            hidden_dim=cfg.hidden_dim,
            action_dim=action_dim,
            num_layer=cfg.num_layer,
            layer_norm=True,
            dropout=cfg.dropout,
        )
        self.aug = common_utils.RandomShiftsAug(pad=4)
        if self.cfg.orth_init:
            self.policy.apply(utils.orth_weight_init)

    def forward(self, obs: dict[str, torch.Tensor]):
        h = self.encoder(obs)
        mu = self.policy(h)  # policy contains tanh
        return mu

    def act(self, obs: dict[str, torch.Tensor], *, eval_mode=True):
        assert eval_mode
        assert not self.training

        unsqueezed = False
        if obs[self.rl_cameras[0]].dim() == 3:
            # add batch dim
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
            unsqueezed = True

        greedy_action = self.forward(obs)

        if unsqueezed:
            greedy_action = greedy_action.squeeze()
        return greedy_action.detach().cpu()

    def loss(self, batch):
        action = batch.action["action"]

        obs = {}
        for camera in self.rl_cameras:
            obs[camera] = batch.obs[camera].float()
            # obs[camera] = self.aug(batch.obs[camera].float())

        pred_action = self.forward(obs)
        loss = nn.functional.mse_loss(pred_action, action, reduction="none")
        loss = loss.sum(1).mean(0)
        return loss
