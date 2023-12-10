from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import einops
from icecream import ic

import common_utils
from bc.multiview_encoder import MultiViewEncoder, MultiViewEncoderConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        assert d_model % num_head == 0

        self.num_head = num_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask):
        """
        x: [batch, seq, d_model]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=self.num_head).unbind(1)
        # force flash attention, it will raise error if flash cannot be applied
        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, attn_mask=attn_mask
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_head, ff_factor, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_head)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_factor * d_model)
        self.linear2 = nn.Linear(ff_factor * d_model, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout = dropout

    def forward(self, x, attn_mask):
        x = x + nn.functional.dropout(self.mha(self.layer_norm1(x), attn_mask), p=self.dropout)
        x = x + nn.functional.dropout(self._ff_block(self.layer_norm2(x)), p=self.dropout)
        return x

    def _ff_block(self, x):
        x = nn.functional.dropout(nn.functional.relu(self.linear1(x)), self.dropout)
        x = self.linear2(x)
        return x


class ObsActionTransformer(nn.Module):
    def __init__(
        self,
        obs_dim,
        num_obs_token,
        action_dim,
        num_cond_action,
        num_pred_action,
        embed_dim,
        depth,
        num_head,
        dropout,
    ) -> None:
        super().__init__()

        self.num_obs_token = num_obs_token
        self.num_cond_action = num_cond_action
        self.num_pred_action = num_pred_action
        self.seq_len = num_obs_token + num_cond_action + num_pred_action
        self.cond_len = num_obs_token + num_cond_action
        self.embed_dim = embed_dim

        self.obs_proj = nn.Linear(obs_dim, num_obs_token * embed_dim)
        self.action_proj = nn.Linear(action_dim, embed_dim)

        # assume inputs are [batch, seq, embed_dim]
        self.positional_encoding = nn.Parameter(torch.empty(1, self.seq_len, embed_dim))
        nn.init.normal_(self.positional_encoding)

        self.transformer = nn.Sequential(
            *[TransformerLayer(embed_dim, num_head, 4, dropout) for _ in range(depth)]
        )

        self.attn_mask = torch.triu(
            torch.full((self.seq_len, self.seq_len), float("-inf")), diagonal=1
        )  # set lower_tri & diagnal = 0
        self.attn_mask[: self.cond_len, : self.cond_len] = 0

    def forward(self, obs_emb: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        if self.attn_mask.device != obs_emb.device:
            self.attn_mask = self.attn_mask.to(obs_emb.device)

        assert action_seq.size(1) == self.num_pred_action + self.num_cond_action

        bsize = obs_emb.size(0)
        obs_tokens = self.obs_proj(obs_emb).view(bsize, self.num_obs_token, self.embed_dim)
        action_tokens = self.action_proj(action_seq)
        input_tokens = torch.cat([obs_tokens, action_tokens], dim=1)
        x = input_tokens + self.positional_encoding

        for layer in self.transformer:
            x = layer(x, attn_mask=self.attn_mask)
        return x[:, -self.num_pred_action :]


@dataclass
class ATransformerConfig:
    encoder: MultiViewEncoderConfig = field(default_factory=lambda: MultiViewEncoderConfig())
    use_prop: int = 0
    cond_horizon: int = 0
    pred_horizon: int = 8
    transformer_obs_token: int = 4
    noise_std: float = 0
    noise_unif: float = 0
    transformer_embed_dim: int = 256
    transformer_depth: int = 3
    transformer_num_head: int = 8
    transformer_dropout: float = 0
    shift_pad: int = 4


class ATransformer(nn.Module):
    def __init__(
        self,
        obs_shape,
        prop_shape,
        action_dim,
        rl_cameras,
        cfg: ATransformerConfig,
    ):
        super().__init__()
        self.rl_cameras = rl_cameras
        self.action_dim = action_dim
        self.cfg = cfg

        self.action_transform = action_transform
        self.action_normalizer = action_normalizer
        if self.action_normalizer is not None:
            self.action_normalizer.to("cuda")

        self.encoder = MultiViewEncoder(
            obs_shape=obs_shape,
            obs_horizon=1,
            prop_shape=prop_shape,
            rl_cameras=rl_cameras,
            use_prop=cfg.use_prop,
            cfg=cfg.encoder,
        )

        self.transformer = ObsActionTransformer(
            obs_dim=self.encoder.repr_dim,
            num_obs_token=cfg.transformer_obs_token,
            action_dim=action_dim,
            num_cond_action=cfg.cond_horizon,
            num_pred_action=cfg.pred_horizon,
            embed_dim=cfg.transformer_embed_dim,
            depth=cfg.transformer_depth,
            num_head=cfg.transformer_num_head,
            dropout=cfg.transformer_dropout,
        )

        self.policy = nn.Sequential(nn.Linear(cfg.transformer_embed_dim, action_dim), nn.Tanh())
        self.aug = common_utils.RandomShiftsAug(pad=cfg.shift_pad)

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

        # add past action BEFORE denormalize & undo-transform
        if self.cfg.cond_horizon > 0:
            self.past_actions.pop(0)
            self.past_actions.append(action)

        if self.action_normalizer is not None:
            action = self.action_normalizer.denormalize(action)
        if self.action_transform is not None:
            action = self.action_transform.undo_transform_action(action)

        return action.cpu()

    def _compute_action(self, obs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        assert obs[self.rl_cameras[0]].dim() == 3
        # add batch dim
        for k, v in obs.items():
            obs[k] = v.unsqueeze(0)

        h = self.encoder(obs)

        cond_actions = None
        if self.cfg.cond_horizon > 0:
            assert len(self.past_actions) == self.cfg.cond_horizon
            cond_actions = torch.stack(self.past_actions).unsqueeze(0)

        input_pred_actions = torch.zeros(1, self.cfg.pred_horizon, self.action_dim).cuda()
        actions = []
        for i in range(self.cfg.pred_horizon):
            if cond_actions is not None:
                input_actions = torch.cat([cond_actions, input_pred_actions], dim=1)
            else:
                input_actions = input_pred_actions

            pred_actions = self.policy(self.transformer(h, input_actions))
            # ic(pred_actions.size())
            if i < self.cfg.pred_horizon - 1:
                input_pred_actions[:, i + 1] = pred_actions[:, i]

            actions.append(pred_actions[:, i].detach().squeeze(0))

        return actions

    def loss(self, batch):
        action = batch.action["action"]
        obs = {"prop": batch.obs["prop"]}

        for camera in self.rl_cameras:
            obs[camera] = self.aug(batch.obs[camera].float())

        # h: [batch, hid_dim]
        h = self.encoder(obs)

        all_input_actions = []
        if self.cfg.cond_horizon > 0:
            all_input_actions.append(batch.obs["cond_action"])
        all_input_actions.append(batch.obs["input_action"])

        # input_action: [batch, num_cond+num_pred, action_dim]
        input_action = torch.cat(all_input_actions, dim=1)

        # add noise to input action
        action_noise = torch.zeros_like(input_action)
        if self.cfg.noise_std > 0:
            action_noise.normal_(mean=0.0, std=self.cfg.noise_std)
        elif self.cfg.noise_unif > 0:
            action_noise.uniform_(-self.cfg.noise_unif, self.cfg.noise_unif)
        input_action = input_action + action_noise

        o = self.transformer(h, input_action)
        pred_action = self.policy(o)
        assert pred_action.size() == action.size()

        valid_action = batch.obs["valid_target"]
        loss = nn.functional.mse_loss(pred_action, action, reduction="none")
        loss = (loss.sum(2) * valid_action).sum() / valid_action.sum()
        return loss
