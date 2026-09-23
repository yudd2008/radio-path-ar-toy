"""Discrete interaction-sequence models: causal AR vs one-shot (non-AR)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.schema import (
    MAX_BOUNCES,
    MAX_CORNERS,
    MAX_SEQ_LEN,
    MAX_WALLS,
    PAD_ID,
    RX_ID,
    TX_ID,
    VOCAB_DIFFRACT_OFFSET,
    VOCAB_WALL_OFFSET,
)

from .encoder import D_MODEL, VOCAB_SIZE, SceneEncoder


def causal_mask(t: int, device: torch.device) -> torch.Tensor:
    m = torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)
    return m


def pointer_logits(
    h: torch.Tensor,
    rx_logits: torch.Tensor,
    wall_embed: torch.Tensor,
    wall_mask: torch.Tensor,
    corner_embed: torch.Tensor,
    corner_mask: torch.Tensor,
) -> torch.Tensor:
    """Score RX plus existing walls/corners. h [B,T,d] → logits [B,T,V]."""
    b, tlen, _ = h.shape
    logits = h.new_full((b, tlen, VOCAB_SIZE), -1e9)
    logits[:, :, RX_ID] = rx_logits
    wall_scores = torch.einsum("btd,bwd->btw", h, wall_embed)
    wall_scores = wall_scores.masked_fill(wall_mask.unsqueeze(1) < 0.5, -1e9)
    logits[:, :, VOCAB_WALL_OFFSET : VOCAB_WALL_OFFSET + MAX_WALLS] = wall_scores
    corner_scores = torch.einsum("btd,bcd->btc", h, corner_embed)
    corner_scores = corner_scores.masked_fill(corner_mask.unsqueeze(1) < 0.5, -1e9)
    logits[:, :, VOCAB_DIFFRACT_OFFSET : VOCAB_DIFFRACT_OFFSET + MAX_CORNERS] = corner_scores
    return logits


@torch.no_grad()
def greedy_decode_interactions(
    model: nn.Module,
    channels: torch.Tensor,
    tx: torch.Tensor,
    rx: torch.Tensor,
    wall_feats: torch.Tensor,
    wall_mask: torch.Tensor,
    force_first: torch.Tensor | None = None,
    max_bounces: int = MAX_BOUNCES,
    corner_feats: torch.Tensor | None = None,
    corner_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy AR rollout. ``force_first`` replaces the first interaction when >= 0."""
    b = channels.size(0)
    dev = channels.device
    seq = torch.full((b, MAX_SEQ_LEN), PAD_ID, device=dev, dtype=torch.long)
    seq[:, 0] = TX_ID
    finished = torch.zeros(b, dtype=torch.bool, device=dev)
    cur_len = 1
    for step in range(max_bounces + 1):
        logits = model(
            seq[:, :cur_len],
            channels,
            tx,
            rx,
            wall_feats,
            wall_mask,
            corner_feats,
            corner_mask,
        )
        last = logits[:, -1, :]
        if step == 0 and force_first is not None:
            use = force_first >= 0
            nxt = last.argmax(dim=-1)
            nxt = torch.where(use, force_first.to(dev), nxt)
        else:
            nxt = last.argmax(dim=-1)
        nxt = torch.where(finished, torch.full_like(nxt, PAD_ID), nxt)
        seq[:, cur_len] = nxt
        finished = finished | (nxt == RX_ID) | (nxt == PAD_ID)
        cur_len += 1
        if bool(finished.all()):
            break
    return seq


class TokenMixer(nn.Module):
    """Map a token id to a vector, using wall or corner geometry."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.special = nn.Embedding(VOCAB_WALL_OFFSET, d_model)  # PAD, TX, RX
        self.d_model = d_model

    def forward(
        self,
        tokens: torch.Tensor,
        wall_embed: torch.Tensor,
        corner_embed: torch.Tensor,
    ) -> torch.Tensor:
        """tokens [B,T] → [B,T,d]."""
        b, tlen = tokens.shape
        out = torch.zeros(b, tlen, self.d_model, device=tokens.device, dtype=wall_embed.dtype)
        special_mask = tokens < VOCAB_WALL_OFFSET
        out[special_mask] = self.special(tokens[special_mask].clamp(min=0, max=VOCAB_WALL_OFFSET - 1))
        refl = (tokens >= VOCAB_WALL_OFFSET) & (tokens < VOCAB_DIFFRACT_OFFSET)
        if refl.any():
            wid = (tokens - VOCAB_WALL_OFFSET).clamp(min=0, max=MAX_WALLS - 1)
            gathered = wall_embed.gather(1, wid.unsqueeze(-1).expand(-1, -1, self.d_model))
            out = torch.where(refl.unsqueeze(-1), gathered, out)
        diffr = tokens >= VOCAB_DIFFRACT_OFFSET
        if diffr.any():
            cid = (tokens - VOCAB_DIFFRACT_OFFSET).clamp(min=0, max=MAX_CORNERS - 1)
            gathered = corner_embed.gather(1, cid.unsqueeze(-1).expand(-1, -1, self.d_model))
            out = torch.where(diffr.unsqueeze(-1), gathered, out)
        return out


class ARPathTransformer(nn.Module):
    """Causal transformer: p(next wall / RX | prefix, scene, Tx, Rx)."""

    def __init__(self, d_model: int = D_MODEL, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.encoder = SceneEncoder(d_model)
        self.mixer = TokenMixer(d_model)
        self.scene_token = nn.Linear(d_model, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.rx_head = nn.Linear(d_model, 1)
        self.d_model = d_model

    def logits_from_hidden(
        self,
        h: torch.Tensor,
        wall_embed: torch.Tensor,
        wall_mask: torch.Tensor,
        corner_embed: torch.Tensor,
        corner_mask: torch.Tensor,
    ) -> torch.Tensor:
        """h [B,T,d] → vocab logits [B,T,V]."""
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
        ttot = x.size(1)
        mask = causal_mask(ttot, x.device)
        h = self.norm(self.tr(x, mask=mask))
        h_tok = h[:, 1:, :]
        return self.logits_from_hidden(h_tok, wall_embed, wall_mask, corner_embed, corner_mask)

    @torch.no_grad()
    def greedy_decode(
        self,
        channels: torch.Tensor,
        tx: torch.Tensor,
        rx: torch.Tensor,
        wall_feats: torch.Tensor,
        wall_mask: torch.Tensor,
        force_first: torch.Tensor | None = None,
        max_bounces: int = MAX_BOUNCES,
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


class OneShotPathModel(nn.Module):
    """Non-AR: jointly score every bounce slot from the scene (no prefix)."""

    def __init__(self, d_model: int = D_MODEL, n_slots: int = MAX_BOUNCES + 1):
        super().__init__()
        self.encoder = SceneEncoder(d_model)
        self.n_slots = n_slots
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.slot_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.rx_head = nn.Linear(d_model, 1)

    def forward(
        self,
        channels: torch.Tensor,
        tx: torch.Tensor,
        rx: torch.Tensor,
        wall_feats: torch.Tensor,
        wall_mask: torch.Tensor,
        tokens: torch.Tensor | None = None,
        corner_feats: torch.Tensor | None = None,
        corner_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del tokens
        scene, wall_embed, corner_embed = self.encoder(
            channels, tx, rx, wall_feats, corner_feats
        )
        if corner_mask is None:
            corner_mask = wall_mask.new_zeros(wall_mask.size(0), corner_embed.size(1))
        b = scene.size(0)
        q = self.slots.unsqueeze(0).expand(b, -1, -1)
        q = self.slot_mlp(torch.cat([q, scene.unsqueeze(1).expand(-1, self.n_slots, -1)], dim=-1))
        logits = q.new_full((b, self.n_slots, VOCAB_SIZE), -1e9)
        logits[:, :, RX_ID] = self.rx_head(q).squeeze(-1)
        wall_scores = torch.einsum("bld,bwd->blw", q, wall_embed)
        wall_scores = wall_scores.masked_fill(wall_mask.unsqueeze(1) < 0.5, -1e9)
        logits[:, :, VOCAB_WALL_OFFSET : VOCAB_WALL_OFFSET + MAX_WALLS] = wall_scores
        corner_scores = torch.einsum("bld,bcd->blc", q, corner_embed)
        corner_scores = corner_scores.masked_fill(corner_mask.unsqueeze(1) < 0.5, -1e9)
        logits[:, :, VOCAB_DIFFRACT_OFFSET : VOCAB_DIFFRACT_OFFSET + MAX_CORNERS] = corner_scores
        return logits

    @torch.no_grad()
    def decode(
        self,
        channels: torch.Tensor,
        tx: torch.Tensor,
        rx: torch.Tensor,
        wall_feats: torch.Tensor,
        wall_mask: torch.Tensor,
        force_first: torch.Tensor | None = None,
        corner_feats: torch.Tensor | None = None,
        corner_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward(
            channels, tx, rx, wall_feats, wall_mask,
            corner_feats=corner_feats, corner_mask=corner_mask,
        )
        pred_slots = logits.argmax(dim=-1)  # B, n_slots
        if force_first is not None:
            use = force_first >= 0
            pred_slots[:, 0] = torch.where(use, force_first, pred_slots[:, 0])
        b = pred_slots.size(0)
        seq = torch.full((b, MAX_SEQ_LEN), PAD_ID, device=pred_slots.device, dtype=torch.long)
        seq[:, 0] = TX_ID
        for i in range(self.n_slots):
            tok = pred_slots[:, i]
            seq[:, i + 1] = tok
        # zero out after first RX
        rx_hit = seq == RX_ID
        seen = rx_hit.cumsum(dim=1) > 1
        seq = seq.masked_fill(seen, PAD_ID)
        return seq
