"""Train the ordinary AR architecture with the Transformer's loss weight.

``ARPathTransformer`` itself is unchanged. This run only changes the objective
and the schedule: interaction tokens are upweighted 3×, the learning rate
follows the same warmup + cosine, and the checkpoint is chosen by validation
teacher-forced accuracy on ``n_bounces≥2``. It is a control for the loss
weight, not a new architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PathNPZDataset
from experiments.common import device, seed_all
from experiments.train_transformer import (
    TRAIN_DEFAULTS,
    _hop2_plateaued,
    _loss_plateaued,
    eval_loader,
    lr_at_epoch,
    token_loss,
)
from models.sequence import ARPathTransformer


def _move(batch, dev):
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_training(npz: str, out_dir: str, seed: int = 0, max_epochs: int | None = None) -> dict:
    seed_all(seed)
    dev = device()
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bs = 32
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(PathNPZDataset(npz, split="train"), batch_size=bs, shuffle=True, generator=g)
    train_eval = DataLoader(PathNPZDataset(npz, split="train", min_bounces=2), batch_size=bs, shuffle=False)
    val_loader = DataLoader(PathNPZDataset(npz, split="val", min_bounces=2), batch_size=bs, shuffle=False)
    model = ARPathTransformer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(TRAIN_DEFAULTS["lr"]), weight_decay=1e-4)
    epochs = int(max_epochs if max_epochs is not None else 80)
    min_epochs, patience = 30, 16
    warmup = int(TRAIN_DEFAULTS["warmup_epochs"])
    max_lr, min_lr = float(TRAIN_DEFAULTS["lr"]), float(TRAIN_DEFAULTS["min_lr"])
    weight = float(TRAIN_DEFAULTS["interaction_token_weight"])
    best, best_epoch, best_state, stall = -1.0, 0, None, 0
    hist: list[dict] = []
    stop_reason = "max_epochs"
    for ep in tqdm(range(1, epochs + 1), desc="train[ar_weighted]"):
        lr = lr_at_epoch(ep, epochs, warmup, max_lr, min_lr)
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        total = n = 0.0
        for batch in train_loader:
            batch = _move(batch, dev)
            opt.zero_grad(set_to_none=True)
            tgt = batch["tokens"][:, 1:]
            logits = model(
                batch["tokens"][:, :-1],
                batch["channels"],
                batch["tx"],
                batch["rx"],
                batch["wall_feats"],
                batch["wall_mask"],
                batch["corner_feats"],
                batch["corner_mask"],
            )
            loss = token_loss(logits, tgt, weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * tgt.size(0)
            n += tgt.size(0)
        tr_loss = total / max(n, 1)
        tr = eval_loader(model, train_eval, dev)
        va = eval_loader(model, val_loader, dev)
        row = {
            "epoch": ep,
            "lr": lr,
            "train_loss": tr_loss,
            "train_tf_acc": tr["tf_acc"],
            "train_hop_acc": tr["hop_acc"],
            "val_loss": va["loss"],
            "val_tf_acc": va["tf_acc"],
            "val_tf_exact": va["tf_exact"],
            "val_hop_acc": va["hop_acc"],
            "val_hop2": va["hop_acc"][1],
        }
        hist.append(row)
        if va["tf_acc"] > best + 1e-6:
            best = float(va["tf_acc"])
            best_epoch = ep
            stall = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
        tqdm.write(
            f"epoch {ep:03d}  lr {lr:.2e}  train_loss {tr_loss:.4f}  "
            f"train_h2 {tr['hop_acc'][1]:.3f}  val_tf {va['tf_acc']:.3f}  "
            f"val_h2 {va['hop_acc'][1]:.3f}"
        )
        loss_flat = _loss_plateaued(hist, 8, 0.02)
        hop_flat = _hop2_plateaued(hist, 8)
        if ep >= min_epochs and stall >= patience and loss_flat and hop_flat:
            stop_reason = "val_patience_and_train_loss_plateau"
            break
        if ep >= min_epochs and stall >= patience * 2:
            stop_reason = "val_patience_exceeded"
            break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = hist[-1]["epoch"]
        best = hist[-1]["val_tf_acc"]
    payload = {
        "state_dict": best_state,
        "val_tf_acc": best,
        "best_epoch": best_epoch,
        "epochs_ran": hist[-1]["epoch"],
        "stop_reason": stop_reason,
        "train_loss_plateaued": _loss_plateaued(hist, 8, 0.02),
        "hist": hist,
        "interaction_token_weight": weight,
        "selection_metric": "val teacher-forced next-token accuracy on n_bounces>=2",
        "architecture": "ARPathTransformer",
        "lr": max_lr,
        "weight_decay": 1e-4,
        "schedule": "warmup+cosine",
    }
    out_path = out / "best.pt"
    torch.save(payload, out_path)
    public = {k: v for k, v in payload.items() if k != "state_dict"}
    (out / "history.json").write_text(json.dumps(public), encoding="utf-8")
    return {
        "ckpt": str(out_path),
        "val_tf_acc": best,
        "best_epoch": best_epoch,
        "epochs_ran": payload["epochs_ran"],
        "stop_reason": stop_reason,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="data/generated/dataset.npz")
    p.add_argument("--out", default="artifacts/ar_weighted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=80)
    args = p.parse_args()
    print(json.dumps(run_training(args.npz, args.out, seed=args.seed, max_epochs=args.max_epochs), indent=2))


if __name__ == "__main__":
    main()
