"""Train continuous interaction-point models (joint / AR / DDPM)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PathNPZDataset
from experiments.common import device, seed_all
from models.continuous import ARPointModel, JointPointRegressor, TinyPointDDPM


def _t_rmse(pred, target, mask) -> float:
    denom = mask.sum().clamp(min=1.0)
    return float(torch.sqrt(((pred - target) ** 2 * mask).sum() / denom).item())


def train_regressor(model, loader, opt, dev, ar: bool) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        kwargs = dict(
            channels=batch["channels"],
            tx=batch["tx"],
            rx=batch["rx"],
            wall_feats=batch["wall_feats"],
            wall_ids=batch["wall_ids"],
            bounce_mask=batch["bounce_mask"],
            corner_feats=batch["corner_feats"],
            kinds=batch["kinds"],
        )
        if ar:
            pred = model(**kwargs, t_prev=ARPointModel.shift_prev(batch["t_on_wall"]))
        else:
            pred = model(**kwargs)
        mask = batch["bounce_mask"]
        if mask.sum() < 1:
            continue
        loss = ((pred - batch["t_on_wall"]) ** 2 * mask).sum() / mask.sum()
        loss.backward()
        opt.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def train_ddpm(model: TinyPointDDPM, loader, opt, dev) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
        if batch["bounce_mask"].sum() < 1:
            continue
        opt.zero_grad(set_to_none=True)
        loss = model.loss(
            batch["t_on_wall"],
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_ids"],
            batch["bounce_mask"],
            corner_feats=batch["corner_feats"],
            kinds=batch["kinds"],
        )
        loss.backward()
        opt.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_regressor(model, loader, dev, ar: bool) -> float:
    model.eval()
    se, w = 0.0, 0.0
    for batch in loader:
        batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
        kwargs = dict(
            channels=batch["channels"],
            tx=batch["tx"],
            rx=batch["rx"],
            wall_feats=batch["wall_feats"],
            wall_ids=batch["wall_ids"],
            bounce_mask=batch["bounce_mask"],
            corner_feats=batch["corner_feats"],
            kinds=batch["kinds"],
        )
        if ar:
            pred = model.free_run(**kwargs)
        else:
            pred = model(**kwargs)
        mask = batch["bounce_mask"]
        se += float(((pred - batch["t_on_wall"]) ** 2 * mask).sum().item())
        w += float(mask.sum().item())
    return (se / max(w, 1.0)) ** 0.5


def run_training(
    npz: str,
    out_dir: str,
    seed: int = 0,
    epochs: int = 16,
    batch_size: int = 32,
    lr: float = 2e-3,
) -> dict:
    seed_all(seed)
    dev = device()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Continuous heads are conditioned on discrete structure; LoS (0 bounce)
    # contributes no t supervision, so keep bounce samples.
    train_ds = PathNPZDataset(npz, split="train", min_bounces=1)
    val_ds = PathNPZDataset(npz, split="val", min_bounces=1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    results = {}

    joint = JointPointRegressor()
    opt = torch.optim.AdamW(joint.parameters(), lr=lr)
    best, state, hist = 1e9, None, []
    for ep in tqdm(range(1, epochs + 1), desc="train[joint_t]"):
        tr = train_regressor(joint, train_loader, opt, dev, ar=False)
        va = eval_regressor(joint, val_loader, dev, ar=False)
        hist.append({"epoch": ep, "train": tr, "val_rmse": va})
        if va < best:
            best, state = va, {k: v.detach().cpu().clone() for k, v in joint.state_dict().items()}
    joint.load_state_dict(state)
    torch.save({"state_dict": state, "val_rmse": best, "hist": hist}, out / "joint_t.pt")
    results["joint_t"] = {"ckpt": str(out / "joint_t.pt"), "val_rmse": best}

    ar = ARPointModel()
    opt = torch.optim.AdamW(ar.parameters(), lr=lr)
    best, state, hist = 1e9, None, []
    for ep in tqdm(range(1, epochs + 1), desc="train[ar_t]"):
        tr = train_regressor(ar, train_loader, opt, dev, ar=True)
        va = eval_regressor(ar, val_loader, dev, ar=True)
        hist.append({"epoch": ep, "train": tr, "val_rmse": va})
        if va < best:
            best, state = va, {k: v.detach().cpu().clone() for k, v in ar.state_dict().items()}
    ar.load_state_dict(state)
    torch.save({"state_dict": state, "val_rmse": best, "hist": hist}, out / "ar_t.pt")
    results["ar_t"] = {"ckpt": str(out / "ar_t.pt"), "val_rmse": best}

    ddpm = TinyPointDDPM()
    opt = torch.optim.AdamW(ddpm.parameters(), lr=lr)
    best, state, hist = 1e9, None, []
    for ep in tqdm(range(1, epochs + 1), desc="train[ddpm_t]"):
        tr = train_ddpm(ddpm, train_loader, opt, dev)
        # cheap val: one-shot sample RMSE (stochastic)
        va = 0.0
        n = 0
        ddpm.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
                pred = ddpm.sample(
                    batch["channels"],
                    batch["tx"],
                    batch["rx"],
                    batch["wall_feats"],
                    batch["wall_ids"],
                    batch["bounce_mask"],
                    n_steps=12,
                    corner_feats=batch["corner_feats"],
                    kinds=batch["kinds"],
                )
                mask = batch["bounce_mask"]
                va += float(((pred - batch["t_on_wall"]) ** 2 * mask).sum().item())
                n += float(mask.sum().item())
        va = (va / max(n, 1.0)) ** 0.5
        hist.append({"epoch": ep, "train": tr, "val_rmse": va})
        if va < best:
            best, state = va, {k: v.detach().cpu().clone() for k, v in ddpm.state_dict().items()}
    ddpm.load_state_dict(state)
    torch.save({"state_dict": state, "val_rmse": best, "hist": hist}, out / "ddpm_t.pt")
    results["ddpm_t"] = {"ckpt": str(out / "ddpm_t.pt"), "val_rmse": best}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/generated/dataset.npz")
    p.add_argument("--out", default="results/ckpts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=16)
    args = p.parse_args()
    run_training(args.npz, args.out, seed=args.seed, epochs=args.epochs)


if __name__ == "__main__":
    main()
