from dataclasses import dataclass, field
import torch
import torch.nn as nn

import common_utils
from common_utils import ibrl_utils as utils
from bc.multiview_encoder import MultiViewEncoder, MultiViewEncoderConfig


@dataclass
class ARnnPolicyConfig:
    encoder: MultiViewEncoderConfig = field(default_factory=lambda: MultiViewEncoderConfig())
    use_prop: int = 0
    hidden_dim: int = 1024
    num_layer: int = 1
    dropout: float = 0
    orth_init: int = 0
    cond_horizon: int = 0
    pred_horizon: int = 8
    noise_std: float = 0
    noise_unif: float = 0
    lstm_depth: int = 2
    use_pos_emb: int = 0
    shift_pad: int = 4


class ARnnPolicy(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        rl_cameras,
        cfg: ARnnPolicyConfig,
    ):
        super().__init__()
        self.rl_cameras = rl_cameras
        self.action_dim = action_dim
        self.cfg = cfg

        self.encoder = MultiViewEncoder(
            obs_shape=obs_shape,
            obs_horizon=1,
            prop_shape=0,
            rl_cameras=rl_cameras,
            use_prop=cfg.use_prop,
            cfg=cfg.encoder,
        )

        if cfg.use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.pred_horizon, action_dim))
            nn.init.normal_(self.pos_emb, std=0.1)
        else:
            self.pos_emb = None

        self.lstm_policy = nn.LSTM(
            input_size=self.encoder.repr_dim + action_dim * (cfg.cond_horizon + 1),
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.lstm_depth,
            batch_first=True,
        )

        policy: list[nn.Module] = [nn.Linear(cfg.hidden_dim, action_dim), nn.Tanh()]
        self.policy = nn.Sequential(*policy)

        self.aug = common_utils.RandomShiftsAug(pad=cfg.shift_pad)
        if self.cfg.orth_init:
            self.policy.apply(utils.orth_weight_init)

        # track actions for online interaction
        self.reset()

    def reset(self):
        self.past_actions = [
            torch.zeros(self.action_dim).cuda() for _ in range(self.cfg.cond_horizon)
        ]
        self.curr_actions = []

    def act(self, obs: dict[str, torch.Tensor], *, eval_mode=True):
        assert eval_mode
        assert not self.training

        if len(self.curr_actions) == 0:
            self.curr_actions = self._compute_action(obs)

        action = self.curr_actions.pop(0)

        if self.cfg.cond_horizon > 0:
            self.past_actions.pop(0)
            self.past_actions.append(action)

        return action.cpu()

    def _compute_action(self, obs: dict[str, torch.Tensor]):
        assert obs[self.rl_cameras[0]].dim() == 3
        # add batch dim
        for k, v in obs.items():
            obs[k] = v.unsqueeze(0)

        h = self.encoder(obs)
        if self.cfg.cond_horizon > 0:
            assert len(self.past_actions) == self.cfg.cond_horizon
            cond_action = torch.stack(self.past_actions).unsqueeze(0)
            cond_action = cond_action.flatten(1, 2)
            h = torch.cat([h, cond_action], dim=-1)

        lstm_hc = None
        pred_action = torch.zeros(1, self.action_dim).cuda()
        actions = []
        for i in range(self.cfg.pred_horizon):
            if self.pos_emb is not None:
                ha = torch.cat([h, self.pos_emb[:, i]], dim=-1)
            else:
                ha = torch.cat([h, pred_action], dim=-1)
            o, lstm_hc = self.lstm_policy(ha, lstm_hc)
            pred_action = self.policy(o)
            actions.append(pred_action.detach().squeeze(0))

        return actions

    def loss(self, batch):
        action = batch.action["action"]
        obs = {}

        for camera in self.rl_cameras:
            obs[camera] = self.aug(batch.obs[camera].float())

        h = self.encoder(obs)
        if self.cfg.cond_horizon > 0:
            cond_action = batch.obs["cond_action"]
            cond_action = cond_action.flatten(1, 2)
            # add noise
            if self.cfg.noise_std > 0:
                action_noise = torch.zeros_like(cond_action)
                action_noise.normal_(mean=0.0, std=self.cfg.noise_std)
                cond_action = cond_action + action_noise
            elif self.cfg.noise_unif:
                action_noise = torch.zeros_like(cond_action)
                action_noise.uniform_(-self.cfg.noise_unif, self.cfg.noise_unif)
                cond_action = cond_action + action_noise

            h = torch.cat([h, cond_action], dim=-1)

        h = h.unsqueeze(1).repeat(1, action.size(1), 1)
        input_action = batch.obs["input_action"]
        # add noise
        if self.cfg.noise_std > 0:
            action_noise = torch.zeros_like(input_action)
            action_noise.normal_(mean=0.0, std=self.cfg.noise_std)
            input_action = input_action + action_noise
        elif self.cfg.noise_unif > 0:
            action_noise = torch.zeros_like(input_action)
            action_noise.uniform_(-self.cfg.noise_unif, self.cfg.noise_unif)
            input_action = input_action + action_noise

        if self.pos_emb is not None:
            ha = torch.cat([h, self.pos_emb.repeat(action.size(0), 1, 1)], dim=-1)
        else:
            ha = torch.cat([h, input_action], dim=-1)
        o, _ = self.lstm_policy(ha)
        pred_action = self.policy(o)
        assert pred_action.size() == action.size()

        valid_action = batch.obs["valid_target"]
        loss = nn.functional.mse_loss(pred_action, action, reduction="none")
        loss = (loss.sum(2) * valid_action).sum() / valid_action.sum()
        return loss
