"""Continuous interaction points: joint regression, AR regression, tiny DDPM.

RadioDiff inspiration: treat *continuous* geometry as a conditional generative
problem rather than a purely discriminative next-step local prediction.
We denoise the 1-D wall parameters t ∈ (0,1)^K jointly, conditioned on the
discrete wall sequence + scene. This is NOT a reimplementation of RadioDiff.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.schema import MAX_BOUNCES, WALL_FEAT_DIM

from .encoder import (
    D_MODEL,
    SceneEncoder,
    SinusoidalTime,
    interaction_slot_features,
    wall_slot_features,
)


def _cond_vec(
    scene: torch.Tensor,
    wall_feats: torch.Tensor,
    wall_ids: torch.Tensor,
    bounce_mask: torch.Tensor,
    corner_feats: torch.Tensor | None = None,
    kinds: torch.Tensor | None = None,
) -> torch.Tensor:
    if corner_feats is not None and kinds is not None:
        slots = interaction_slot_features(wall_feats, corner_feats, wall_ids, kinds)
    else:
        slots = wall_slot_features(wall_feats, wall_ids)
    flat = torch.cat(
        [slots.reshape(slots.size(0), -1), bounce_mask, scene], dim=-1
    )
    return flat


class JointPointRegressor(nn.Module):
    """One-shot (non-AR) prediction of all bounce parameters t."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.encoder = SceneEncoder(d_model)
        in_dim = MAX_BOUNCES * WALL_FEAT_DIM + MAX_BOUNCES + d_model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, MAX_BOUNCES),
        )

    def forward(self, channels, tx, rx, wall_feats, wall_ids, bounce_mask,
                corner_feats=None, kinds=None, **_):
        scene, _, _ = self.encoder(channels, tx, rx, wall_feats, corner_feats)
        cond = _cond_vec(scene, wall_feats, wall_ids, bounce_mask, corner_feats, kinds)
        return torch.sigmoid(self.mlp(cond))


class ARPointModel(nn.Module):
    """Autoregressive scalar t_i | t_<i, walls, scene (GRU)."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.encoder = SceneEncoder(d_model)
        self.in_proj = nn.Linear(WALL_FEAT_DIM + 1 + d_model, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        channels,
        tx,
        rx,
        wall_feats,
        wall_ids,
        bounce_mask,
        t_prev: torch.Tensor,
        corner_feats=None,
        kinds=None,
        **_,
    ):
        """t_prev: [B, L] with t_prev[:,0]=0 and t_prev[:,i]=t_{i-1} (teacher force)."""
        scene, _, _ = self.encoder(channels, tx, rx, wall_feats, corner_feats)
        if corner_feats is not None and kinds is not None:
            slots = interaction_slot_features(wall_feats, corner_feats, wall_ids, kinds)
        else:
            slots = wall_slot_features(wall_feats, wall_ids)
        scene_exp = scene.unsqueeze(1).expand(-1, MAX_BOUNCES, -1)
        x = torch.cat([slots, t_prev.unsqueeze(-1), scene_exp], dim=-1)
        h, _ = self.gru(self.in_proj(x))
        return torch.sigmoid(self.head(h).squeeze(-1))

    @staticmethod
    def shift_prev(t: torch.Tensor) -> torch.Tensor:
        prev = torch.zeros_like(t)
        prev[:, 1:] = t[:, :-1]
        return prev

    @torch.no_grad()
    def free_run(
        self,
        channels,
        tx,
        rx,
        wall_feats,
        wall_ids,
        bounce_mask,
        t0_override: torch.Tensor | None = None,
        corner_feats=None,
        kinds=None,
        **_,
    ) -> torch.Tensor:
        b = channels.size(0)
        device = channels.device
        t_prev = torch.zeros(b, MAX_BOUNCES, device=device)
        preds = torch.zeros(b, MAX_BOUNCES, device=device)
        for i in range(MAX_BOUNCES):
            y = self.forward(
                channels,
                tx,
                rx,
                wall_feats,
                wall_ids,
                bounce_mask,
                t_prev=t_prev,
                corner_feats=corner_feats,
                kinds=kinds,
            )
            ti = y[:, i]
            if i == 0 and t0_override is not None:
                ti = t0_override
            preds[:, i] = ti
            if i + 1 < MAX_BOUNCES:
                t_prev = t_prev.clone()
                t_prev[:, i + 1] = ti
        return preds * bounce_mask


class TinyPointDDPM(nn.Module):
    """Joint DDPM over t ∈ R^K, conditioned on discrete structure + scene."""

    def __init__(self, n_steps: int = 40, d_model: int = D_MODEL):
        super().__init__()
        self.n_steps = n_steps
        self.encoder = SceneEncoder(d_model)
        self.time = SinusoidalTime(32)
        cond_dim = MAX_BOUNCES * WALL_FEAT_DIM + MAX_BOUNCES + d_model
        self.net = nn.Sequential(
            nn.Linear(MAX_BOUNCES + 32 + cond_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, MAX_BOUNCES),
        )
        betas = torch.linspace(1e-4, 0.06, n_steps)
        alphas = 1.0 - betas
        a_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("a_bar", a_bar)

    def _cond(self, channels, tx, rx, wall_feats, wall_ids, bounce_mask,
              corner_feats=None, kinds=None, **_):
        scene, _, _ = self.encoder(channels, tx, rx, wall_feats, corner_feats)
        return _cond_vec(scene, wall_feats, wall_ids, bounce_mask, corner_feats, kinds)

    def loss(self, x0, channels, tx, rx, wall_feats, wall_ids, bounce_mask,
             corner_feats=None, kinds=None, **_):
        """x0: [B,K] in [0,1]."""
        b = x0.size(0)
        t = torch.randint(0, self.n_steps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        a = self.a_bar[t].unsqueeze(-1)
        xt = a.sqrt() * x0 + (1.0 - a).sqrt() * noise
        cond = self._cond(
            channels, tx, rx, wall_feats, wall_ids, bounce_mask, corner_feats, kinds
        )
        temb = self.time(t)
        pred = self.net(torch.cat([xt, temb, cond], dim=-1))
        err = (pred - noise) ** 2
        denom = bounce_mask.sum().clamp(min=1.0)
        return (err * bounce_mask).sum() / denom

    @torch.no_grad()
    def sample(
        self,
        channels,
        tx,
        rx,
        wall_feats,
        wall_ids,
        bounce_mask,
        n_steps: int | None = None,
        freeze_t0: torch.Tensor | None = None,
        corner_feats=None,
        kinds=None,
        **_,
    ) -> torch.Tensor:
        """Ancestral sample. If freeze_t0 is set, replace x[:,0] each step
        (toy analogue of a constrained inverse / inpainting problem).
        """
        n_steps = n_steps or self.n_steps
        b = channels.size(0)
        x = torch.randn(b, MAX_BOUNCES, device=channels.device)
        cond = self._cond(
            channels, tx, rx, wall_feats, wall_ids, bounce_mask, corner_feats, kinds
        )
        # subsample timesteps if n_steps < trained
        idxs = torch.linspace(self.n_steps - 1, 0, n_steps, device=x.device).long()
        for t in idxs:
            temb = self.time(t.expand(b))
            eps = self.net(torch.cat([x, temb, cond], dim=-1))
            a = self.a_bar[t]
            a_prev = self.a_bar[t - 1] if int(t) > 0 else x.new_tensor(1.0)
            beta = self.betas[t]
            x0_hat = (x - (1.0 - a).sqrt() * eps) / a.sqrt().clamp(min=1e-6)
            x0_hat = x0_hat.clamp(0.0, 1.0)
            mean = a_prev.sqrt() * x0_hat + (1.0 - a_prev).sqrt() * eps
            if int(t) > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = x0_hat
            x = x.clamp(0.0, 1.0)
            if freeze_t0 is not None:
                x = x.clone()
                x[:, 0] = freeze_t0
        return x * bounce_mask
