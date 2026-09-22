"""Train the causal Transformer interaction baseline to a validation plateau.

Checkpoint selection matches the ordinary AR run: best validation
teacher-forced next-token accuracy. Training continues until that accuracy
has stopped improving and the training loss has flattened, or until
``max_epochs``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PathNPZDataset
from data.schema import PAD_ID
from experiments.common import device, seed_all
from models.transformer_seq import build_serious_transformer, count_parameters

# Documented recipe. Overrides belong on the CLI / saved config, not here.
TRAIN_DEFAULTS = {
    "lr": 2e-3,
    "min_lr": 1e-5,
    "weight_decay": 1e-4,
    "dropout": 0.05,
    "batch_size": 64,
    "max_epochs": 140,
    "min_epochs": 48,
    "patience": 24,
    "warmup_epochs": 5,
    "grad_clip": 1.0,
    "betas": (0.9, 0.98),
    "loss_plateau_tol": 0.02,
    "loss_plateau_window": 10,
    # Unweighted CE collapses to "emit RX" at hop 2, because one-bounce paths
    # (second token = RX) outnumber multi-bounce continuations. Upweight
    # R_wall / D_corner targets so those tokens are actually fit. RX stays 1.
    "interaction_token_weight": 3.0,
    "selection": "val teacher-forced next-token accuracy on n_bounces>=2",
}


def token_loss(logits: torch.Tensor, target: torch.Tensor, interaction_weight: float = 1.0) -> torch.Tensor:
    """Cross-entropy over non-PAD tokens. Interaction tokens can be upweighted."""
    b, t, v = logits.shape
    raw = F.cross_entropy(
        logits.reshape(b * t, v),
        target.reshape(b * t),
        ignore_index=PAD_ID,
        reduction="none",
    ).view(b, t)
    weights = torch.ones_like(raw)
    if interaction_weight != 1.0:
        weights = torch.where(target >= 3, torch.full_like(weights, float(interaction_weight)), weights)
    weights = torch.where(target == PAD_ID, torch.zeros_like(weights), weights)
    return (raw * weights).sum() / weights.sum().clamp(min=1.0)


def _move(batch: dict, dev: torch.device) -> dict:
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}


def sequence_logits(model, batch: dict) -> torch.Tensor:
    return model(
        batch["tokens"][:, :-1],
        batch["channels"],
        batch["tx"],
        batch["rx"],
        batch["wall_feats"],
        batch["wall_mask"],
        batch["corner_feats"],
        batch["corner_mask"],
    )


def adamw_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Decay matrix weights; skip biases, norms, and the position table."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        bare = name.endswith(".bias") or param.ndim == 1 or "pos" in name or "norm" in name
        (no_decay if bare else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at_epoch(epoch: int, max_epochs: int, warmup: int, max_lr: float, min_lr: float) -> float:
    """1-indexed epoch. Linear warmup, then cosine down to ``min_lr``."""
    if epoch <= warmup:
        return min_lr + (max_lr - min_lr) * (epoch / max(warmup, 1))
    span = max(max_epochs - warmup, 1)
    progress = min(max((epoch - warmup) / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def train_one_epoch(model, loader, opt, dev, interaction_weight: float) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = _move(batch, dev)
        opt.zero_grad(set_to_none=True)
        tgt = batch["tokens"][:, 1:]
        loss = token_loss(sequence_logits(model, batch), tgt, interaction_weight)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_DEFAULTS["grad_clip"])
        opt.step()
        total += float(loss.item()) * tgt.size(0)
        n += tgt.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_loader(model, loader, dev) -> dict:
    """Teacher-forced loss, token accuracy, exact path rate, and per-hop accuracy."""
    model.eval()
    loss_sum, n_seq = 0.0, 0
    correct, total = 0, 0
    exact, n_exact = 0, 0
    hop_hit = [0, 0, 0, 0]
    hop_tot = [0, 0, 0, 0]
    for batch in loader:
        batch = _move(batch, dev)
        tgt = batch["tokens"][:, 1:]
        logits = sequence_logits(model, batch)
        loss = token_loss(logits, tgt)
        pred = logits.argmax(dim=-1)
        mask = tgt != PAD_ID
        correct += int((pred[mask] == tgt[mask]).sum().item())
        total += int(mask.sum().item())
        loss_sum += float(loss.item()) * tgt.size(0)
        n_seq += tgt.size(0)
        tokens = batch["tokens"]
        for i in range(tokens.size(0)):
            gt = tokens[i]
            n_tgt = int((gt[1:] != PAD_ID).sum().item())
            if n_tgt <= 0:
                continue
            n_exact += 1
            exact += int(torch.equal(pred[i, :n_tgt], gt[1 : 1 + n_tgt]))
            for k in range(1, 5):
                if k >= gt.numel() or int(gt[k].item()) == PAD_ID:
                    continue
                hop_tot[k - 1] += 1
                hop_hit[k - 1] += int(int(pred[i, k - 1].item()) == int(gt[k].item()))
    hops = [hop_hit[k] / hop_tot[k] if hop_tot[k] else float("nan") for k in range(4)]
    return {
        "loss": loss_sum / max(n_seq, 1),
        "tf_acc": correct / max(total, 1),
        "tf_exact": exact / max(n_exact, 1),
        "hop_acc": hops,
    }


def _hop2_plateaued(hist: list[dict], window: int, tol: float = 0.01) -> bool:
    """True when train hop-2 accuracy on n_bounces≥2 has stopped rising."""
    if len(hist) < 2 * window:
        return False
    prev = max(row["train_hop_acc"][1] for row in hist[-2 * window : -window])
    recent = max(row["train_hop_acc"][1] for row in hist[-window:])
    return recent <= prev + tol


def _loss_plateaued(hist: list[dict], window: int, tol: float) -> bool:
    losses = [row["train_loss"] for row in hist]
    if len(losses) < 2 * window:
        return False
    prev = sum(losses[-2 * window : -window]) / window
    recent = sum(losses[-window:]) / window
    if prev <= 0:
        return recent <= prev
    return (prev - recent) / prev <= tol


def _save_curves(hist: list[dict], best_epoch: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in hist]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].plot(epochs, [row["train_loss"] for row in hist], label="train loss (weighted)")
    axes[0].plot(epochs, [row["val_loss"] for row in hist], label="val loss (unweighted, ≥2)")
    axes[0].axvline(best_epoch, color="0.4", ls="--", lw=1, label=f"best epoch {best_epoch}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("weighted cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Train loss (interaction-weighted)")
    axes[1].plot(epochs, [row["train_tf_acc"] for row in hist], label="train TF acc ≥2")
    axes[1].plot(epochs, [row["val_tf_acc"] for row in hist], label="val TF acc ≥2")
    axes[1].plot(epochs, [row.get("val_hop2", float("nan")) for row in hist], label="val hop2 ≥2")
    axes[1].axvline(best_epoch, color="0.4", ls="--", lw=1)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Teacher-forced accuracy")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_training(
    npz: str,
    out_dir: str,
    seed: int = 0,
    max_epochs: int | None = None,
    curve_path: str | None = None,
) -> dict:
    seed_all(seed)
    dev = device()
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_ds = PathNPZDataset(npz, split="train")
    val_ds = PathNPZDataset(npz, split="val", min_bounces=2)
    g = torch.Generator()
    g.manual_seed(seed)
    bs = int(TRAIN_DEFAULTS["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g)
    # Eval-mode passes so plotted accuracy is not depressed by dropout.
    train_eval_loader = DataLoader(
        PathNPZDataset(npz, split="train", min_bounces=2), batch_size=bs, shuffle=False
    )
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    interaction_weight = float(TRAIN_DEFAULTS["interaction_token_weight"])

    model = build_serious_transformer({"dropout": float(TRAIN_DEFAULTS["dropout"])}).to(dev)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(
        adamw_groups(model, float(TRAIN_DEFAULTS["weight_decay"])),
        lr=float(TRAIN_DEFAULTS["lr"]),
        betas=tuple(TRAIN_DEFAULTS["betas"]),
    )
    epochs = int(max_epochs if max_epochs is not None else TRAIN_DEFAULTS["max_epochs"])
    min_epochs = int(TRAIN_DEFAULTS["min_epochs"])
    patience = int(TRAIN_DEFAULTS["patience"])
    warmup = int(TRAIN_DEFAULTS["warmup_epochs"])
    max_lr = float(TRAIN_DEFAULTS["lr"])
    min_lr = float(TRAIN_DEFAULTS["min_lr"])

    best = -1.0
    best_epoch = 0
    best_state = None
    stall = 0
    hist: list[dict] = []
    stop_reason = "max_epochs"
    for ep in tqdm(range(1, epochs + 1), desc="train[serious_transformer]"):
        lr = lr_at_epoch(ep, epochs, warmup, max_lr, min_lr)
        _set_lr(opt, lr)
        tr_loss = train_one_epoch(model, train_loader, opt, dev, interaction_weight)
        tr_metrics = eval_loader(model, train_eval_loader, dev)
        va = eval_loader(model, val_loader, dev)
        row = {
            "epoch": ep,
            "lr": lr,
            "train_loss": tr_loss,
            "train_tf_acc": tr_metrics["tf_acc"],
            "train_tf_exact": tr_metrics["tf_exact"],
            "train_hop_acc": tr_metrics["hop_acc"],
            "val_loss": va["loss"],
            "val_tf_acc": va["tf_acc"],
            "val_tf_exact": va["tf_exact"],
            "val_hop_acc": va["hop_acc"],
            "val_hop2": va["hop_acc"][1],
        }
        hist.append(row)
        improved = va["tf_acc"] > best + 1e-6
        if improved:
            best = float(va["tf_acc"])
            best_epoch = ep
            stall = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
        hops = " ".join(f"h{k+1}={va['hop_acc'][k]:.3f}" for k in range(4))
        tqdm.write(
            f"epoch {ep:03d}  lr {lr:.2e}  train_loss {tr_loss:.4f}  "
            f"train_tf {tr_metrics['tf_acc']:.3f}  train_h2 {tr_metrics['hop_acc'][1]:.3f}  "
            f"val_tf {va['tf_acc']:.3f}  val_exact {va['tf_exact']:.3f}  {hops}"
        )
        loss_flat = _loss_plateaued(
            hist,
            int(TRAIN_DEFAULTS["loss_plateau_window"]),
            float(TRAIN_DEFAULTS["loss_plateau_tol"]),
        )
        hop2_flat = _hop2_plateaued(hist, int(TRAIN_DEFAULTS["loss_plateau_window"]))
        if ep >= min_epochs and stall >= patience and loss_flat and hop2_flat:
            stop_reason = "val_patience_and_train_loss_plateau"
            break
        if ep >= min_epochs and stall >= patience * 2:
            # Validation has stopped improving for a long stretch.
            stop_reason = "val_patience_exceeded"
            break

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    last_epoch = int(hist[-1]["epoch"]) if hist else 0
    if best_state is None:
        best_state = last_state
        best = float(hist[-1]["val_tf_acc"]) if hist else float("nan")
        best_epoch = last_epoch
    model.load_state_dict(best_state)
    loss_flat = _loss_plateaued(
        hist,
        int(TRAIN_DEFAULTS["loss_plateau_window"]),
        float(TRAIN_DEFAULTS["loss_plateau_tol"]),
    )
    ckpt_path = out / "best.pt"
    payload = {
        "state_dict": best_state,
        "config": model.config,
        "val_tf_acc": best,
        "best_epoch": best_epoch,
        "epochs_ran": int(hist[-1]["epoch"]) if hist else 0,
        "stop_reason": stop_reason,
        "train_loss_plateaued": loss_flat,
        "n_params": n_params,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_val_population": "n_bounces>=2",
        "seed": seed,
        "train_defaults": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in TRAIN_DEFAULTS.items()
        },
        "hist": hist,
        "selection_metric": "val teacher-forced next-token accuracy",
    }
    torch.save(payload, ckpt_path)
    torch.save(
        {"state_dict": last_state, "config": model.config, "epoch": last_epoch},
        out / "last.pt",
    )
    hist_path = out / "history.json"
    public = {k: v for k, v in payload.items() if k != "state_dict"}
    hist_path.write_text(json.dumps(public, indent=2), encoding="utf-8")
    if curve_path:
        _save_curves(hist, best_epoch, Path(curve_path))
    else:
        _save_curves(hist, best_epoch, out / "train_curves.png")
    return {
        "ckpt": str(ckpt_path),
        "history": str(hist_path),
        "val_tf_acc": best,
        "best_epoch": best_epoch,
        "epochs_ran": payload["epochs_ran"],
        "stop_reason": stop_reason,
        "train_loss_plateaued": loss_flat,
        "n_params": n_params,
        "final_train_loss": hist[-1]["train_loss"] if hist else None,
        "final_train_tf_acc": hist[-1]["train_tf_acc"] if hist else None,
        "final_val_tf_acc": hist[-1]["val_tf_acc"] if hist else None,
        "hist": hist,
        "config": model.config,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="data/generated/dataset.npz")
    p.add_argument("--out", default="artifacts/transformer_seq")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=TRAIN_DEFAULTS["max_epochs"])
    p.add_argument("--curve", default="results/figures/transformer_train_curves.png")
    args = p.parse_args()
    stats = run_training(args.npz, args.out, seed=args.seed, max_epochs=args.max_epochs, curve_path=args.curve)
    brief = {k: v for k, v in stats.items() if k != "hist"}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
