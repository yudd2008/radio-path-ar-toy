"""Causal Transformer decoder for discrete interaction sequences.

Same conditioning and pointer head as ``ARPathTransformer`` (scene CNN, Tx/Rx,
wall/corner geometry, ``R_wall_k`` / ``D_corner_c`` / ``RX``). This baseline is
the stronger sequence-model foil: learned positions, pre-norm blocks, a wider
feed-forward, and a training schedule that is allowed to run to a validation
plateau. It does not replace the image-method ground truth.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data.schema import MAX_SEQ_LEN

from .encoder import SceneEncoder
from .sequence import TokenMixer, causal_mask, greedy_decode_interactions, pointer_logits

# Fixed recipe used by experiments/train_transformer.py. Checkpoint configs
# override these when a run is reloaded.
DEFAULT_CONFIG: dict[str, float | int] = {
    "d_model": 128,
    "nhead": 4,
    "nlayers": 4,
    "dropout": 0.1,
    "ff_mult": 4,
    "max_pos": MAX_SEQ_LEN + 4,
}


class SeriousPathTransformer(nn.Module):
    """Decoder-only Transformer: p(next interaction | prefix, scene, Tx, Rx)."""

    def __init__(
        self,
        d_model: int = int(DEFAULT_CONFIG["d_model"]),
        nhead: int = int(DEFAULT_CONFIG["nhead"]),
        nlayers: int = int(DEFAULT_CONFIG["nlayers"]),
        dropout: float = float(DEFAULT_CONFIG["dropout"]),
        ff_mult: int = int(DEFAULT_CONFIG["ff_mult"]),
        max_pos: int = int(DEFAULT_CONFIG["max_pos"]),
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.config = {
            "d_model": int(d_model),
            "nhead": int(nhead),
            "nlayers": int(nlayers),
            "dropout": float(dropout),
            "ff_mult": int(ff_mult),
            "max_pos": int(max_pos),
        }
        self.d_model = int(d_model)
        self.encoder = SceneEncoder(self.d_model)
        self.mixer = TokenMixer(self.d_model)
        self.pos = nn.Embedding(max_pos, self.d_model)
        self.scene_token = nn.Linear(self.d_model, self.d_model)
        self.embed_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.d_model)
        self.rx_head = nn.Linear(self.d_model, 1)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)

    def logits_from_hidden(
        self,
        h: torch.Tensor,
        wall_embed: torch.Tensor,
        wall_mask: torch.Tensor,
        corner_embed: torch.Tensor,
        corner_mask: torch.Tensor,
    ) -> torch.Tensor:
        return pointer_logits(
            h,
            self.rx_head(h).squeeze(-1),
            wall_embed,
            wall_mask,
            corner_embed,
            corner_mask,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        channels: torch.Tensor,
        tx: torch.Tensor,
        rx: torch.Tensor,
        wall_feats: torch.Tensor,
        wall_mask: torch.Tensor,
        corner_feats: torch.Tensor | None = None,
        corner_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scene, wall_embed, corner_embed = self.encoder(
            channels, tx, rx, wall_feats, corner_feats
        )
        if corner_mask is None:
            corner_mask = wall_mask.new_zeros(wall_mask.size(0), corner_embed.size(1))
        tok = self.mixer(tokens, wall_embed, corner_embed)
        prefix = self.scene_token(scene).unsqueeze(1)
        x = torch.cat([prefix, tok], dim=1)
        if x.size(1) > self.pos.num_embeddings:
            raise RuntimeError(
                f"sequence length {x.size(1)} exceeds max_pos={self.pos.num_embeddings}"
            )
        pos_ids = torch.arange(x.size(1), device=x.device)
        x = self.embed_drop(x + self.pos(pos_ids).unsqueeze(0))
        h = self.norm(self.tr(x, mask=causal_mask(x.size(1), x.device)))
        return self.logits_from_hidden(h[:, 1:, :], wall_embed, wall_mask, corner_embed, corner_mask)

    @torch.no_grad()
    def greedy_decode(
        self,
        channels: torch.Tensor,
        tx: torch.Tensor,
        rx: torch.Tensor,
        wall_feats: torch.Tensor,
        wall_mask: torch.Tensor,
        force_first: torch.Tensor | None = None,
        max_bounces: int = 3,
        corner_feats: torch.Tensor | None = None,
        corner_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return greedy_decode_interactions(
            self,
            channels,
            tx,
            rx,
            wall_feats,
            wall_mask,
            force_first=force_first,
            max_bounces=max_bounces,
            corner_feats=corner_feats,
            corner_mask=corner_mask,
        )


def build_serious_transformer(config: dict | None = None) -> SeriousPathTransformer:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    return SeriousPathTransformer(
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        nlayers=int(cfg["nlayers"]),
        dropout=float(cfg["dropout"]),
        ff_mult=int(cfg["ff_mult"]),
        max_pos=int(cfg["max_pos"]),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
