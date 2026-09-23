"""Train discrete sequence models (AR transformer + one-shot)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PathNPZDataset
from data.schema import PAD_ID
from experiments.common import device, seed_all
from models.sequence import ARPathTransformer, OneShotPathModel


def token_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """logits [B,T,V], target [B,T] with PAD ignored."""
    b, t, v = logits.shape
    return F.cross_entropy(
        logits.reshape(b * t, v),
        target.reshape(b * t),
        ignore_index=PAD_ID,
    )


def train_one(model, loader, opt, ar: bool, dev) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        if ar:
            inp = batch["tokens"][:, :-1]
            tgt = batch["tokens"][:, 1:]
            logits = model(
                inp,
                batch["channels"],
                batch["tx"],
                batch["rx"],
                batch["wall_feats"],
                batch["wall_mask"],
                batch["corner_feats"],
                batch["corner_mask"],
            )
        else:
            tgt = batch["tokens"][:, 1:]
            logits = model(
                batch["channels"],
                batch["tx"],
                batch["rx"],
                batch["wall_feats"],
                batch["wall_mask"],
                corner_feats=batch["corner_feats"],
                corner_mask=batch["corner_mask"],
            )
        loss = token_loss(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.item()) * tgt.size(0)
        n += tgt.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_tf_acc(model, loader, ar: bool, dev) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
        tgt = batch["tokens"][:, 1:]
        if ar:
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
        else:
            logits = model(
                batch["channels"],
                batch["tx"],
                batch["rx"],
                batch["wall_feats"],
                batch["wall_mask"],
                corner_feats=batch["corner_feats"],
                corner_mask=batch["corner_mask"],
            )
        pred = logits.argmax(dim=-1)
        mask = tgt != PAD_ID
        correct += int((pred[mask] == tgt[mask]).sum().item())
        total += int(mask.sum().item())
    return correct / max(total, 1)


def run_training(
    npz: str,
    out_dir: str,
    seed: int = 0,
    epochs: int = 18,
    batch_size: int = 32,
    lr: float = 2e-3,
    only: tuple[str, ...] | None = None,
) -> dict:
    seed_all(seed)
    dev = device()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_ds = PathNPZDataset(npz, split="train")
    val_ds = PathNPZDataset(npz, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    results = {}
    specs = [
        ("ar_transformer", ARPathTransformer(), True),
        ("oneshot", OneShotPathModel(), False),
    ]
    if only is not None:
        specs = [spec for spec in specs if spec[0] in only]
    for name, model, ar in specs:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best, best_state = -1.0, None
        hist = []
        for ep in tqdm(range(1, epochs + 1), desc=f"train[{name}]"):
            tr = train_one(model, train_loader, opt, ar, dev)
            va = eval_tf_acc(model, val_loader, ar, dev)
            hist.append({"epoch": ep, "train_loss": tr, "val_tf_acc": va})
            if va >= best:
                best = va
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        ckpt = out / f"{name}.pt"
        torch.save({"state_dict": best_state, "val_tf_acc": best, "hist": hist}, ckpt)
        results[name] = {"ckpt": str(ckpt), "val_tf_acc": best, "hist": hist}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/generated/dataset.npz")
    p.add_argument("--out", default="results/ckpts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=18)
    args = p.parse_args()
    run_training(args.npz, args.out, seed=args.seed, epochs=args.epochs)


if __name__ == "__main__":
    main()
