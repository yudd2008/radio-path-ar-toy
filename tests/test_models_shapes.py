"""Tiny shape / forward-pass smoke tests."""

from __future__ import annotations

import torch

from data.schema import MAX_BOUNCES, MAX_SEQ_LEN, MAX_WALLS, RX_ID, TX_ID, VOCAB_WALL_OFFSET
from models.continuous import ARPointModel, JointPointRegressor, TinyPointDDPM
from models.sequence import ARPathTransformer, OneShotPathModel


def _batch(b: int = 3):
    tokens = torch.zeros(b, MAX_SEQ_LEN, dtype=torch.long)
    tokens[:, 0] = TX_ID
    tokens[:, 1] = VOCAB_WALL_OFFSET + 0
    tokens[:, 2] = VOCAB_WALL_OFFSET + 1
    tokens[:, 3] = RX_ID
    wall_ids = torch.full((b, MAX_BOUNCES), -1, dtype=torch.long)
    wall_ids[:, 0] = 0
    wall_ids[:, 1] = 1
    bounce_mask = torch.zeros(b, MAX_BOUNCES)
    bounce_mask[:, :2] = 1
    wall_mask = torch.zeros(b, MAX_WALLS)
    wall_mask[:, :8] = 1
    return dict(
        tokens=tokens,
        channels=torch.rand(b, 3, 32, 32),
        tx=torch.rand(b, 2),
        rx=torch.rand(b, 2),
        wall_feats=torch.rand(b, MAX_WALLS, 6),
        wall_mask=wall_mask,
        wall_ids=wall_ids,
        bounce_mask=bounce_mask,
        t_on_wall=torch.rand(b, MAX_BOUNCES) * bounce_mask,
    )


def test_ar_forward_and_decode():
    m = ARPathTransformer()
    m.eval()
    b = _batch()
    logits = m(b["tokens"][:, :-1], b["channels"], b["tx"], b["rx"], b["wall_feats"], b["wall_mask"])
    assert logits.shape[0] == 3
    seq = m.greedy_decode(b["channels"], b["tx"], b["rx"], b["wall_feats"], b["wall_mask"])
    assert seq.shape == b["tokens"].shape
    assert (seq[:, 0] == TX_ID).all()


def test_oneshot_and_continuous():
    b = _batch()
    os = OneShotPathModel()
    logits = os(b["channels"], b["tx"], b["rx"], b["wall_feats"], b["wall_mask"])
    assert logits.dim() == 3
    j = JointPointRegressor()
    t = j(b["channels"], b["tx"], b["rx"], b["wall_feats"], b["wall_ids"], b["bounce_mask"])
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
    )
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_ar_forward_and_decode()
    test_oneshot_and_continuous()
    print("ok")
