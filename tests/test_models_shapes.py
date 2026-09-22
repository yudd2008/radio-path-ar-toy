"""Tiny shape / forward-pass smoke tests."""

from __future__ import annotations

import torch

from data.schema import (
    KIND_D_INT,
    KIND_R_INT,
    MAX_BOUNCES,
    MAX_CORNERS,
    MAX_SEQ_LEN,
    MAX_WALLS,
    RX_ID,
    TX_ID,
    VOCAB_DIFFRACT_OFFSET,
    VOCAB_SIZE,
    VOCAB_WALL_OFFSET,
)
from models.continuous import ARPointModel, JointPointRegressor, TinyPointDDPM
import torch.nn.functional as F

from experiments.train_transformer import lr_at_epoch
from models.sequence import ARPathTransformer, OneShotPathModel
from models.transformer_seq import SeriousPathTransformer


def _batch(b: int = 3):
    tokens = torch.zeros(b, MAX_SEQ_LEN, dtype=torch.long)
    tokens[:, 0] = TX_ID
    tokens[:, 1] = VOCAB_WALL_OFFSET + 0
    tokens[:, 2] = VOCAB_DIFFRACT_OFFSET + 1
    tokens[:, 3] = RX_ID
    wall_ids = torch.full((b, MAX_BOUNCES), -1, dtype=torch.long)
    wall_ids[:, 0] = 0
    wall_ids[:, 1] = 1
    kinds = torch.full((b, MAX_BOUNCES), -1, dtype=torch.long)
    kinds[:, 0] = KIND_R_INT
    kinds[:, 1] = KIND_D_INT
    bounce_mask = torch.zeros(b, MAX_BOUNCES)
    bounce_mask[:, 0] = 1  # only reflection t is free
    wall_mask = torch.zeros(b, MAX_WALLS)
    wall_mask[:, :8] = 1
    corner_mask = torch.zeros(b, MAX_CORNERS)
    corner_mask[:, :8] = 1
    return dict(
        tokens=tokens,
        channels=torch.rand(b, 3, 32, 32),
        tx=torch.rand(b, 2),
        rx=torch.rand(b, 2),
        wall_feats=torch.rand(b, MAX_WALLS, 6),
        wall_mask=wall_mask,
        corner_feats=torch.rand(b, MAX_CORNERS, 6),
        corner_mask=corner_mask,
        wall_ids=wall_ids,
        kinds=kinds,
        bounce_mask=bounce_mask,
        t_on_wall=torch.rand(b, MAX_BOUNCES) * bounce_mask,
    )


def test_ar_forward_and_decode():
    m = ARPathTransformer()
    m.eval()
    b = _batch()
    logits = m(
        b["tokens"][:, :-1],
        b["channels"],
        b["tx"],
        b["rx"],
        b["wall_feats"],
        b["wall_mask"],
        b["corner_feats"],
        b["corner_mask"],
    )
    assert logits.shape[0] == 3
    assert logits.shape[-1] == VOCAB_SIZE
    seq = m.greedy_decode(
        b["channels"],
        b["tx"],
        b["rx"],
        b["wall_feats"],
        b["wall_mask"],
        corner_feats=b["corner_feats"],
        corner_mask=b["corner_mask"],
    )
    assert seq.shape == b["tokens"].shape
    assert (seq[:, 0] == TX_ID).all()


def test_oneshot_and_continuous():
    b = _batch()
    os = OneShotPathModel()
    logits = os(
        b["channels"],
        b["tx"],
        b["rx"],
        b["wall_feats"],
        b["wall_mask"],
        corner_feats=b["corner_feats"],
        corner_mask=b["corner_mask"],
    )
    assert logits.dim() == 3
    assert logits.shape[-1] == VOCAB_SIZE
    j = JointPointRegressor()
    t = j(
        b["channels"],
        b["tx"],
        b["rx"],
        b["wall_feats"],
        b["wall_ids"],
        b["bounce_mask"],
        corner_feats=b["corner_feats"],
        kinds=b["kinds"],
    )
    assert t.shape == b["t_on_wall"].shape
    ar = ARPointModel()
    t2 = ar(
        b["channels"],
        b["tx"],
        b["rx"],
        b["wall_feats"],
        b["wall_ids"],
        b["bounce_mask"],
        t_prev=ARPointModel.shift_prev(b["t_on_wall"]),
        corner_feats=b["corner_feats"],
        kinds=b["kinds"],
    )
    assert t2.shape == b["t_on_wall"].shape
    d = TinyPointDDPM()
    loss = d.loss(
        b["t_on_wall"],
        b["channels"],
        b["tx"],
        b["rx"],
        b["wall_feats"],
        b["wall_ids"],
        b["bounce_mask"],
        corner_feats=b["corner_feats"],
        kinds=b["kinds"],
    )
    assert torch.isfinite(loss)
    sample = d.sample(
        b["channels"],
        b["tx"],
        b["rx"],
        b["wall_feats"],
        b["wall_ids"],
        b["bounce_mask"],
        n_steps=4,
        corner_feats=b["corner_feats"],
        kinds=b["kinds"],
    )
    assert sample.shape == b["t_on_wall"].shape


def _forward_serious(model, batch, tokens):
    return model(
        tokens,
        batch["channels"],
        batch["tx"],
        batch["rx"],
        batch["wall_feats"],
        batch["wall_mask"],
        batch["corner_feats"],
        batch["corner_mask"],
    )


def test_serious_transformer_causal_decode_and_step():
    torch.manual_seed(0)
    model = SeriousPathTransformer(d_model=32, nhead=4, nlayers=2, dropout=0.0, ff_mult=2)
    model.eval()
    batch = _batch()
    logits = _forward_serious(model, batch, batch["tokens"][:, :-1])
    assert logits.shape[0] == 3
    assert logits.shape[-1] == VOCAB_SIZE
    inp = batch["tokens"][:, :-1].clone()
    logits_a = _forward_serious(model, batch, inp)
    inp_b = inp.clone()
    inp_b[:, -1] = VOCAB_WALL_OFFSET + 5
    logits_b = _forward_serious(model, batch, inp_b)
    assert torch.allclose(logits_a[:, 0], logits_b[:, 0], atol=1e-5)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-5)
    seq = model.greedy_decode(
        batch["channels"],
        batch["tx"],
        batch["rx"],
        batch["wall_feats"],
        batch["wall_mask"],
        corner_feats=batch["corner_feats"],
        corner_mask=batch["corner_mask"],
    )
    assert seq.shape == batch["tokens"].shape
    assert (seq[:, 0] == TX_ID).all()
    forced = torch.full((3,), VOCAB_WALL_OFFSET + 3)
    forced_seq = model.greedy_decode(
        batch["channels"],
        batch["tx"],
        batch["rx"],
        batch["wall_feats"],
        batch["wall_mask"],
        force_first=forced,
        corner_feats=batch["corner_feats"],
        corner_mask=batch["corner_mask"],
    )
    assert torch.equal(forced_seq[:, 1], forced)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    train_logits = _forward_serious(model, batch, batch["tokens"][:, :-1])
    target = batch["tokens"][:, 1:]
    loss = F.cross_entropy(train_logits.reshape(-1, VOCAB_SIZE), target.reshape(-1), ignore_index=0)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()


def test_lr_schedule_warmup_then_decay():
    warmup = [lr_at_epoch(ep, 20, 4, 1e-3, 1e-5) for ep in range(1, 5)]
    assert warmup[0] < warmup[-1]
    mid = lr_at_epoch(12, 20, 4, 1e-3, 1e-5)
    end = lr_at_epoch(20, 20, 4, 1e-3, 1e-5)
    assert mid > end
    assert abs(end - 1e-5) < 1e-8


if __name__ == "__main__":
    test_ar_forward_and_decode()
    test_oneshot_and_continuous()
    test_serious_transformer_causal_decode_and_step()
    test_lr_schedule_warmup_then_decay()
    print("ok")
