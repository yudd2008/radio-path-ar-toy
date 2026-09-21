"""Scene CNN encoder (RadioUNet-style occupancy context, tiny)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.schema import MAX_BOUNCES, MAX_WALLS, VOCAB_WALL_OFFSET, WALL_FEAT_DIM

VOCAB_SIZE = VOCAB_WALL_OFFSET + MAX_WALLS  # PAD, TX, RX, walls...
D_MODEL = 64


class SceneEncoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.loc = nn.Linear(4, 16)
        self.out = nn.Linear(48 + 16, d_model)
        self.wall_mlp = nn.Sequential(
            nn.Linear(WALL_FEAT_DIM + 4, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        channels: torch.Tensor,
        tx: torch.Tensor,
        rx: torch.Tensor,
        wall_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (scene_vec [B,d], wall_embed [B,W,d])."""
        h = self.cnn(channels).flatten(1)
        loc = torch.cat([tx, rx], dim=-1)
        scene = self.out(torch.cat([h, self.loc(loc)], dim=-1))
        loc_exp = loc.unsqueeze(1).expand(-1, wall_feats.size(1), -1)
        wall_in = torch.cat([wall_feats, loc_exp], dim=-1)
        wall_embed = self.wall_mlp(wall_in)
        return scene, wall_embed


def wall_slot_features(
    wall_feats: torch.Tensor,
    wall_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather per-bounce wall geometry. wall_ids: [B, L] with -1 pad → zeros."""
    b, l = wall_ids.shape
    w = wall_feats.size(1)
    idx = wall_ids.clamp(min=0, max=w - 1)
    gather = wall_feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, wall_feats.size(-1)))
    valid = (wall_ids >= 0) & (wall_ids < w)
    return gather * valid.unsqueeze(-1).float()


class SinusoidalTime(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
        )
        ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)[:, : self.dim]
