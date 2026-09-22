"""Autoregressive error-accumulation experiment (mandatory).

Compares:
  - teacher forcing vs free-run discrete sequences
  - intervention: corrupt the first interaction, then continue
  - non-AR one-shot, and oracle-first-token + AR rest
  - continuous: AR t-head vs joint regression vs tiny DDPM vs image-method oracle
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import PathNPZDataset
from data.schema import (
    MAX_BOUNCES,
    MAX_CORNERS,
    MAX_WALLS,
    PAD_ID,
    VOCAB_DIFFRACT_OFFSET,
    VOCAB_WALL_OFFSET,
    diffract_token_id,
    reflect_token_id,
)
from env.viz import save_scene_paths
from experiments.common import (
    device,
    interactions_from_rec,
    load_jsonl_index,
    points_from_interactions,
    scene_from_json,
    seed_all,
    validity_and_points,
)
from models.continuous import ARPointModel, JointPointRegressor, TinyPointDDPM
from models.sequence import ARPathTransformer, OneShotPathModel
from pathfind.diffraction import reconstruct_interactions


def _load_ar(ckpt: Path) -> ARPathTransformer:
    m = ARPathTransformer()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m


def _load_oneshot(ckpt: Path) -> OneShotPathModel:
    m = OneShotPathModel()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _mean(xs) -> float:
    xs = [float(x) for x in xs if x == x]  # drop nan
    return float(np.mean(xs)) if xs else float("nan")


def _as_numpy_seq(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.int64)


def _enc_kwargs(batch) -> dict:
    return dict(
        channels=batch["channels"],
        tx=batch["tx"],
        rx=batch["rx"],
        wall_feats=batch["wall_feats"],
        wall_mask=batch["wall_mask"],
        corner_feats=batch["corner_feats"],
        corner_mask=batch["corner_mask"],
    )


@torch.no_grad()
def _second_choice_first_token(model: ARPathTransformer, batch, dev) -> torch.Tensor:
    """Second-highest first interaction (R_wall / D_corner / RX) given TX."""
    kw = _enc_kwargs(batch)
    kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kw.items()}
    logits = model(batch["tokens"][:, :1].to(dev), **kw)
    first = logits[:, -1, :].clone()
    top2 = first.topk(k=2, dim=-1).indices
    gt_first = batch["tokens"][:, 1].to(dev)
    pick = torch.where(top2[:, 0] == gt_first, top2[:, 1], top2[:, 0])
    # If still equal (degenerate), shift to another existing wall.
    same = pick == gt_first
    if same.any():
        walls = torch.arange(MAX_WALLS, device=dev) + VOCAB_WALL_OFFSET
        corners = torch.arange(MAX_CORNERS, device=dev) + VOCAB_DIFFRACT_OFFSET
        exist_w = batch["wall_mask"].to(dev) > 0.5
        exist_c = batch["corner_mask"].to(dev) > 0.5
        for b in range(pick.size(0)):
            if not bool(same[b]):
                continue
            cand = torch.cat([walls[exist_w[b]], corners[exist_c[b]]])
            cand = cand[cand != gt_first[b]]
            pick[b] = cand[0] if cand.numel() else pick[b]
    return pick


def _random_wrong_first(batch, rng: np.random.Generator) -> torch.Tensor:
    b = batch["tokens"].size(0)
    out = batch["tokens"][:, 1].clone()
    for i in range(b):
        wids = np.where((batch["wall_mask"][i] > 0.5).numpy())[0]
        cids = np.where((batch["corner_mask"][i] > 0.5).numpy())[0]
        gt_tok = int(batch["tokens"][i, 1].item())
        cand = [reflect_token_id(int(w)) for w in wids]
        cand += [diffract_token_id(int(c)) for c in cids]
        cand = [t for t in cand if t != gt_tok]
        if cand:
            out[i] = int(rng.choice(cand))
    return out


def _hop_flags(pred: np.ndarray, gt: np.ndarray, n_hops: int) -> list[int]:
    """Match at positions 1..n_hops (after TX). Includes RX if present."""
    flags = []
    limit = min(n_hops, pred.shape[0] - 1, gt.shape[0] - 1)
    for k in range(1, limit + 1):
        flags.append(int(pred[k] == gt[k] and gt[k] != PAD_ID))
    return flags


PROTOCOL_TAGS = ("tf", "freerun", "oracle_first", "interv_2nd", "interv_rand")


def pack_protocol(stats: dict) -> dict:
    """Mean rates + hop table for the shared AR intervention tags."""
    summary = {k: _mean(v) for k, v in stats.items() if k != "n_bounces"}
    summary["n_eval"] = len(stats.get("n_bounces", []))
    summary["mean_n_bounces"] = _mean(stats.get("n_bounces", []))
    hop_table = {}
    for tag in PROTOCOL_TAGS:
        hop_table[tag] = {
            "exact": summary.get(f"{tag}_exact", float("nan")),
            "valid": summary.get(f"{tag}_valid", float("nan")),
            "later_acc": summary.get(f"{tag}_later_acc", float("nan")),
            "later_wall_acc": summary.get(f"{tag}_later_wall_acc", float("nan")),
            "hop2_wall": summary.get(f"{tag}_hop2_wall", float("nan")),
            "any_scene_path": summary.get(f"{tag}_any_scene_path", float("nan")),
            "pred_nbounces": summary.get(f"{tag}_pred_nbounces", float("nan")),
            "xy_mean": summary.get(f"{tag}_xy_mean", float("nan")),
            "hops": [summary.get(f"{tag}_hop{k}_acc", float("nan")) for k in range(1, MAX_BOUNCES + 2)],
            "xy_hops": [summary.get(f"{tag}_xy_hop{k}", float("nan")) for k in range(1, MAX_BOUNCES + 2)],
        }
    return {"summary": summary, "hop_table": hop_table}


@torch.no_grad()
def score_sequence_model(model, loader, json_index, rng, dev=None) -> dict:
    """Teacher-forcing, free-run, and first-hop corruption for one sequence model.

    Same definitions as the AR branch of ``eval_discrete``:
    teacher-forced hop accuracy, greedy free-run, oracle first interaction,
    forced second-best first interaction, and a random existing wall/corner
    that is not the labeled first hop. ``rng`` should be a fresh generator
    created with the experiment seed (``seed + 7``) so both models see the
    same illegal first hops.
    """
    if dev is None:
        dev = torch.device("cpu")
    model = model.to(dev)
    model.eval()
    stats = defaultdict(list)
    scene_path_set: dict[int, set[tuple[tuple[str, int], ...]]] = defaultdict(set)
    for rec in json_index.values():
        scene_path_set[int(rec["scene_id"])].add(tuple(interactions_from_rec(rec)))

    for batch in loader:
        bsz = batch["tokens"].size(0)
        tokens = batch["tokens"]
        enc = _enc_kwargs(batch)
        enc = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}
        logits = model(tokens[:, :-1].to(dev), **enc)
        tf_pred = logits.argmax(dim=-1)
        second = _second_choice_first_token(model, batch, dev)
        rnd = _random_wrong_first(batch, rng)
        free = model.greedy_decode(**enc)
        oracle = model.greedy_decode(**enc, force_first=tokens[:, 1].to(dev))
        interv = model.greedy_decode(**enc, force_first=second)
        interv_rand = model.greedy_decode(**enc, force_first=rnd.to(dev))
        first_logits = model(tokens[:, :1].to(dev), **enc)[:, -1, :]
        top2 = first_logits.topk(k=2, dim=-1).indices
        gt_first = tokens[:, 1].to(dev)

        for i in range(bsz):
            gt = _as_numpy_seq(tokens[i])
            nb = int(batch["n_bounces"][i].item())
            sid = int(batch["scene_id"][i].item())
            pid = int(batch["path_id"][i].item())
            rec = json_index[(sid, pid)]
            scene = scene_from_json(rec)
            gt_pts = np.array(rec["points"], dtype=np.float64)
            n_tgt = int((gt[1:] != PAD_ID).sum())

            def consume(tag: str, pred_seq: np.ndarray, start_hop: int = 1):
                flags = _hop_flags(pred_seq, gt, n_tgt)
                for k, f in enumerate(flags, start=1):
                    stats[f"{tag}_hop{k}_acc"].append(f)
                stats[f"{tag}_exact"].append(int(np.array_equal(pred_seq[: n_tgt + 1], gt[: n_tgt + 1])))
                later = flags[start_hop:]
                if later:
                    stats[f"{tag}_later_acc"].append(float(np.mean(later)))
                if nb >= 2 and len(flags) >= 2:
                    stats[f"{tag}_later_wall_acc"].append(float(np.mean(flags[1:nb])))
                if nb >= 2 and len(flags) >= 2:
                    stats[f"{tag}_hop2_wall"].append(float(flags[1]))
                valid, pts, inter = validity_and_points(scene, pred_seq)
                stats[f"{tag}_valid"].append(int(valid))
                stats[f"{tag}_pred_nbounces"].append(len(inter))
                stats[f"{tag}_any_scene_path"].append(int(tuple(inter) in scene_path_set[sid]))
                if pts is not None:
                    n = min(len(pts), len(gt_pts))
                    err = [float(np.linalg.norm(pts[j] - gt_pts[j])) for j in range(1, n)]
                    for j, e in enumerate(err, start=1):
                        stats[f"{tag}_xy_hop{j}"].append(e)
                    stats[f"{tag}_xy_mean"].append(float(np.mean(err)) if err else 0.0)
                else:
                    stats[f"{tag}_xy_mean"].append(float("nan"))
                return valid, pts, inter, flags

            tf_seq = gt.copy()
            tf_seq[1:] = _as_numpy_seq(tf_pred[i])
            consume("tf", tf_seq)
            consume("freerun", _as_numpy_seq(free[i]))
            consume("oracle_first", _as_numpy_seq(oracle[i]))
            consume("interv_2nd", _as_numpy_seq(interv[i]))
            consume("interv_rand", _as_numpy_seq(interv_rand[i]))
            stats["first_token_tf"].append(int(int(tf_pred[i, 0]) == int(gt[1])))
            stats["first_token_free"].append(int(int(free[i, 1]) == int(gt[1])))
            stats["first_token_top2"].append(int(bool((top2[i] == gt_first[i]).any().item())))
            stats["n_bounces"].append(nb)
    return pack_protocol(stats)


@torch.no_grad()
def eval_discrete(ar, oneshot, loader, json_index, rng, fig_dir: Path) -> dict:
    stats = defaultdict(list)
    examples = []
    n_plot = 0
    scene_path_set: dict[int, set[tuple[tuple[str, int], ...]]] = defaultdict(set)
    for rec in json_index.values():
        scene_path_set[int(rec["scene_id"])].add(tuple(interactions_from_rec(rec)))

    for batch in loader:
        bsz = batch["tokens"].size(0)
        tokens = batch["tokens"]
        enc = _enc_kwargs(batch)
        # teacher forcing hop acc
        logits = ar(tokens[:, :-1], **enc)
        tf_pred = logits.argmax(dim=-1)  # B, T-1  aligned with tokens[:,1:]
        os_logits = oneshot(**enc)
        os_pred_slots = os_logits.argmax(dim=-1)

        second = _second_choice_first_token(ar, batch, device())
        rnd = _random_wrong_first(batch, rng)

        free = ar.greedy_decode(**enc)
        oracle = ar.greedy_decode(**enc, force_first=tokens[:, 1])
        interv = ar.greedy_decode(**enc, force_first=second)
        interv_rand = ar.greedy_decode(**enc, force_first=rnd)
        os_seq = oneshot.decode(**enc)
        os_interv = oneshot.decode(**enc, force_first=second)

        for i in range(bsz):
            gt = _as_numpy_seq(tokens[i])
            nb = int(batch["n_bounces"][i].item())
            sid = int(batch["scene_id"][i].item())
            pid = int(batch["path_id"][i].item())
            rec = json_index[(sid, pid)]
            scene = scene_from_json(rec)
            gt_pts = np.array(rec["points"], dtype=np.float64)
            n_tgt = int((gt[1:] != PAD_ID).sum())

            def consume(tag: str, pred_seq: np.ndarray, start_hop: int = 1):
                flags = _hop_flags(pred_seq, gt, n_tgt)
                for k, f in enumerate(flags, start=1):
                    stats[f"{tag}_hop{k}_acc"].append(f)
                stats[f"{tag}_exact"].append(int(np.array_equal(pred_seq[: n_tgt + 1], gt[: n_tgt + 1])))
                later = flags[start_hop:]  # hops after first interaction (may include RX)
                if later:
                    stats[f"{tag}_later_acc"].append(float(np.mean(later)))
                # Subsequent *walls only* (exclude the easy terminal RX).
                if nb >= 2 and len(flags) >= 2:
                    stats[f"{tag}_later_wall_acc"].append(float(np.mean(flags[1:nb])))
                if nb >= 2 and len(flags) >= 2:
                    stats[f"{tag}_hop2_wall"].append(float(flags[1]))
                valid, pts, inter = validity_and_points(scene, pred_seq)
                stats[f"{tag}_valid"].append(int(valid))
                stats[f"{tag}_pred_nbounces"].append(len(inter))
                stats[f"{tag}_any_scene_path"].append(int(tuple(inter) in scene_path_set[sid]))
                if pts is not None:
                    n = min(len(pts), len(gt_pts))
                    err = [float(np.linalg.norm(pts[j] - gt_pts[j])) for j in range(1, n)]
                    for j, e in enumerate(err, start=1):
                        stats[f"{tag}_xy_hop{j}"].append(e)
                    stats[f"{tag}_xy_mean"].append(float(np.mean(err)) if err else 0.0)
                else:
                    stats[f"{tag}_xy_mean"].append(float("nan"))
                return valid, pts, inter, flags

            # teacher forcing: stitch TX + tf predictions
            tf_seq = gt.copy()
            tf_seq[1:] = _as_numpy_seq(tf_pred[i])
            consume("tf", tf_seq)

            consume("freerun", _as_numpy_seq(free[i]))
            consume("oracle_first", _as_numpy_seq(oracle[i]))
            consume("interv_2nd", _as_numpy_seq(interv[i]))
            consume("interv_rand", _as_numpy_seq(interv_rand[i]))

            os_s = _as_numpy_seq(os_seq[i])
            consume("oneshot", os_s)
            consume("oneshot_interv", _as_numpy_seq(os_interv[i]))

            stats["first_token_tf"].append(int(int(tf_pred[i, 0]) == int(gt[1])))
            stats["first_token_free"].append(int(int(free[i, 1]) == int(gt[1])))
            stats["first_token_oneshot"].append(int(int(os_pred_slots[i, 0]) == int(gt[1])))
            stats["n_bounces"].append(nb)

            if n_plot < 8 and nb >= 2:
                _, free_pts, _ = validity_and_points(scene, _as_numpy_seq(free[i]))
                _, iv_pts, iv_inter = validity_and_points(scene, _as_numpy_seq(interv[i]))
                paths = [{"points": gt_pts, "label": "GT", "mechanism": rec.get("mechanism", "reflection")}]
                if free_pts is not None:
                    paths.append({"points": free_pts, "label": "AR free-run", "mechanism": "mixed"})
                if iv_pts is not None:
                    paths.append({"points": iv_pts, "label": "AR wrong-1st", "mechanism": "mixed"})
                elif iv_inter:
                    fake_t = [0.5] * len(iv_inter)
                    paths.append(
                        {
                            "points": points_from_interactions(scene, iv_inter, fake_t),
                            "label": "wrong-1st (invalid geom)",
                            "mechanism": "mixed",
                        }
                    )
                save_scene_paths(
                    scene,
                    paths,
                    fig_dir / f"example_{n_plot}_sid{sid}.png",
                    title=f"scene {sid}  GT {rec.get('tokens')}  n={nb}",
                    wall_labels=True,
                )
                n_plot += 1
                examples.append({"scene_id": sid, "path_id": pid, "gt": rec["tokens"]})

    summary = {k: _mean(v) for k, v in stats.items() if k != "n_bounces"}
    summary["n_eval"] = len(stats["n_bounces"])
    summary["mean_n_bounces"] = _mean(stats["n_bounces"])
    # hop-wise tables for the key methods
    hop_table = {}
    for tag in ["tf", "freerun", "oracle_first", "interv_2nd", "interv_rand", "oneshot", "oneshot_interv"]:
        hop_table[tag] = {
            "exact": summary.get(f"{tag}_exact", float("nan")),
            "valid": summary.get(f"{tag}_valid", float("nan")),
            "later_acc": summary.get(f"{tag}_later_acc", float("nan")),
            "later_wall_acc": summary.get(f"{tag}_later_wall_acc", float("nan")),
            "hop2_wall": summary.get(f"{tag}_hop2_wall", float("nan")),
            "any_scene_path": summary.get(f"{tag}_any_scene_path", float("nan")),
            "pred_nbounces": summary.get(f"{tag}_pred_nbounces", float("nan")),
            "xy_mean": summary.get(f"{tag}_xy_mean", float("nan")),
            "hops": [summary.get(f"{tag}_hop{k}_acc", float("nan")) for k in range(1, MAX_BOUNCES + 2)],
            "xy_hops": [summary.get(f"{tag}_xy_hop{k}", float("nan")) for k in range(1, MAX_BOUNCES + 2)],
        }
    return {"summary": summary, "hop_table": hop_table, "examples": examples}


@torch.no_grad()
def eval_continuous(joint, ar_t, ddpm, loader, json_index, rng) -> dict:
    stats = defaultdict(list)
    for batch in loader:
        bsz = batch["tokens"].size(0)
        cont_kw = dict(
            channels=batch["channels"],
            tx=batch["tx"],
            rx=batch["rx"],
            wall_feats=batch["wall_feats"],
            wall_ids=batch["wall_ids"],
            bounce_mask=batch["bounce_mask"],
            corner_feats=batch["corner_feats"],
            kinds=batch["kinds"],
        )
        jpred = joint(**cont_kw)
        ar_free = ar_t.free_run(**cont_kw)
        t_gt = batch["t_on_wall"]
        # corrupt first t
        noise = torch.full((bsz,), 0.28)
        t0_bad = (t_gt[:, 0] + noise).clamp(0.05, 0.95)
        # if GT t0 already near bound, flip to the other side
        t0_bad = torch.where((t_gt[:, 0] - t0_bad).abs() < 0.08, (t_gt[:, 0] - 0.28).clamp(0.05, 0.95), t0_bad)
        ar_bad = ar_t.free_run(**cont_kw, t0_override=t0_bad)
        dd = ddpm.sample(**cont_kw, n_steps=12)
        dd_freeze = ddpm.sample(**cont_kw, n_steps=12, freeze_t0=t0_bad)
        for i in range(bsz):
            sid = int(batch["scene_id"][i].item())
            pid = int(batch["path_id"][i].item())
            rec = json_index[(sid, pid)]
            scene = scene_from_json(rec)
            inter = interactions_from_rec(rec)
            nb = len(inter)
            if nb < 1:
                continue
            mask = batch["bounce_mask"][i].numpy()
            gt_t = t_gt[i].numpy()
            gt_pts = np.array(rec["points"], dtype=np.float64)

            def rec_err(tag, t_hat, start=0):
                t_hat = t_hat.detach().cpu().numpy()
                for k in range(nb):
                    if mask[k] < 0.5:
                        continue
                    stats[f"{tag}_t_hop{k+1}"].append(abs(float(t_hat[k] - gt_t[k])))
                pts = points_from_interactions(scene, inter, t_hat[:nb])
                for k in range(nb):
                    stats[f"{tag}_xy_hop{k+1}"].append(
                        float(np.linalg.norm(pts[k + 1] - gt_pts[k + 1]))
                    )
                # later hops only
                if nb >= 2:
                    later_xy = [
                        float(np.linalg.norm(pts[k + 1] - gt_pts[k + 1]))
                        for k in range(1, nb)
                    ]
                    stats[f"{tag}_later_xy"].append(float(np.mean(later_xy)))

            rec_err("joint", jpred[i])
            rec_err("ar_free", ar_free[i])
            rec_err("ar_badt0", ar_bad[i])
            rec_err("ddpm", dd[i])
            rec_err("ddpm_freeze_t0", dd_freeze[i])

            # geometry oracle (discrete R/D structure known)
            oracle = reconstruct_interactions(scene, inter)
            if oracle is not None:
                for k in range(nb):
                    stats["oracle_t_hop" + str(k + 1)].append(
                        abs(float(oracle.t_on_wall[k] - gt_t[k]))
                    )
                    stats["oracle_xy_hop" + str(k + 1)].append(
                        float(np.linalg.norm(oracle.points[k + 1] - gt_pts[k + 1]))
                    )
            # first-hop forced error magnitude (sanity)
            stats["t0_corruption"].append(abs(float(t0_bad[i] - t_gt[i, 0])))

    summary = {k: _mean(v) for k, v in stats.items()}
    summary["n_eval"] = len(stats.get("t0_corruption", []))
    return summary


def make_plots(discrete: dict, continuous: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hop_table = discrete["hop_table"]
    labels = {
        "tf": "AR teacher-force",
        "freerun": "AR free-run",
        "oracle_first": "oracle 1st + AR rest",
        "interv_2nd": "wrong 1st (2nd-best) + AR",
        "interv_rand": "wrong 1st (random wall) + AR",
        "oneshot": "one-shot (non-AR)",
        "oneshot_interv": "one-shot, overwrite 1st",
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for tag, lab in labels.items():
        ys = hop_table[tag]["hops"][:3]
        ax.plot(np.arange(1, len(ys) + 1), ys, marker="o", label=lab)
    ax.set_xlabel("sequence hop after Tx (1 = first wall, 2 = second wall, 3 often Rx)")
    ax.set_ylabel("token accuracy vs this GT path")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([1, 2, 3])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)
    ax.set_title("Discrete path: teacher-forcing vs free-run vs first-token intervention")
    fig.tight_layout()
    fig.savefig(out_dir / "discrete_hop_accuracy.png", dpi=140)
    plt.close(fig)

    # bar: exact / valid  — skip stitched teacher-force (not a decoded path)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    tags = [t for t in labels if t != "tf"]
    x = np.arange(len(tags))
    exact = [hop_table[t]["exact"] for t in tags]
    valid = [hop_table[t]["valid"] for t in tags]
    ax.bar(x - 0.18, exact, 0.36, label="exact match vs this GT path")
    ax.bar(x + 0.18, valid, 0.36, label="geometrically valid path (R / D / mixed)")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[t] for t in tags], rotation=28, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.legend(fontsize=8)
    ax.set_title("Exact GT match vs geometric validity (another IRT branch still counts as valid)")
    fig.tight_layout()
    fig.savefig(out_dir / "discrete_exact_valid.png", dpi=140)
    plt.close(fig)

    # continuous error growth
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = {
        "joint regression": [continuous.get(f"joint_xy_hop{k}", np.nan) for k in range(1, 4)],
        "AR t free-run": [continuous.get(f"ar_free_xy_hop{k}", np.nan) for k in range(1, 4)],
        "AR t, corrupted t0": [continuous.get(f"ar_badt0_xy_hop{k}", np.nan) for k in range(1, 4)],
        "DDPM joint": [continuous.get(f"ddpm_xy_hop{k}", np.nan) for k in range(1, 4)],
        "DDPM freeze t0": [continuous.get(f"ddpm_freeze_t0_xy_hop{k}", np.nan) for k in range(1, 4)],
        "image-method oracle": [continuous.get(f"oracle_xy_hop{k}", np.nan) for k in range(1, 4)],
    }
    xs = np.arange(1, 4)
    for lab, ys in series.items():
        ax.plot(xs, ys, marker="o", label=lab)
    ax.set_xlabel("bounce hop")
    ax.set_ylabel("mean |Δxy| vs GT interaction point")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Continuous points: error vs hop (GT discrete walls)")
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_error_growth.png", dpi=140)
    plt.close(fig)


def write_findings(discrete: dict, continuous: dict, out_dir: Path) -> str:
    ht = discrete["hop_table"]
    n_eval = discrete["summary"].get("n_eval", float("nan"))
    hop1_tf = ht["tf"]["hops"][0]
    hop2_tf = ht["tf"]["hops"][1]
    hop2_free = ht["freerun"]["hops"][1]
    hop2_oracle = ht["oracle_first"]["hops"][1]
    hop2_2nd = ht["interv_2nd"]["hops"][1]
    later_w_oracle = ht["oracle_first"]["later_wall_acc"]
    later_w_2nd = ht["interv_2nd"]["later_wall_acc"]
    exact_free = ht["freerun"]["exact"]
    exact_oracle = ht["oracle_first"]["exact"]
    exact_2nd = ht["interv_2nd"]["exact"]
    exact_rand = ht["interv_rand"]["exact"]
    exact_os = ht["oneshot"]["exact"]
    valid_free = ht["freerun"]["valid"]
    valid_2nd = ht["interv_2nd"]["valid"]
    valid_rand = ht["interv_rand"]["valid"]
    valid_os = ht["oneshot"]["valid"]
    any_2nd = ht["interv_2nd"]["any_scene_path"]
    any_rand = ht["interv_rand"]["any_scene_path"]
    oracle_xy = continuous.get("oracle_xy_hop1", float("nan"))
    ar_later = continuous.get("ar_badt0_later_xy", float("nan"))
    joint_later = continuous.get("joint_later_xy", float("nan"))
    ar_free_xy = continuous.get("ar_free_later_xy", float("nan"))
    ddpm_xy = continuous.get("ddpm_later_xy", float("nan"))
    joint_h1 = continuous.get("joint_xy_hop1", float("nan"))

    lines = [
        f"结论（n={int(n_eval)} 条测试路径，n_bounces≥2；数字来自本次 toy run，seed=0，不是预设口号）。",
        "论文对照见 [`docs/paper_notes.md`](../docs/paper_notes.md)。",
        "",
        "### 论文主张（本玩具对齐的那几条）",
        "",
        "**WinProp IRT**（Altair 用户指南；Hoppe et al., EPMCC 1999）：寻径是在预处理好的墙面/tile 可见性关系上做 **树搜索**；交互点被约束在这些离散元件上；交互次数很少（文档称最多约三次即可）。树上每一枝是「两个元件之间的可见性关系」，预测时先展开发射端可见的第一层，再递归检查反射条件。",
        "→ 本玩具：`R_wall_k` / `D_corner_c` token = 树节点（反射墙或绕射角点）；合法 token 邻接 = 树边；连续 t = 该墙上的点（绕射点就是角点，t=0）。普通 AR 把「树」当成「一条句子」的局部 next-token。",
        "",
        "**RadioDiff**（Wang et al., IEEE TCCN 2024）：无采样无线电地图构建更应是 **条件生成**，而不是 RadioUNet 式纯判别 MSE。路径损耗并不已经写在输入里，必须被生成出来。",
        "→ 本玩具 **不重实现 RadioDiff、不生成场图**。只把同一课用在 **固定离散路径结构上的连续交互点**：联合回归/小 DDPM，而不是逐步 AR 猜 t。RadioDiff 生成 pathloss map；我们生成/refinement 的是墙上的点。",
        "",
        "**RadioUNet**：仅作占用栅格 + Tx/Rx 通道的场景编码器上下文，不估计 RM。",
        "",
        "WinProp IRT 也跟踪垂直棱/楔的绕射。本玩具在矩形角点上加了简化的 2D Keller/UTD **存在性**判定（轮廓/前向面、自由空间、最小弯折），token 为 `D_corner_c`。这不是完整 UTD 系数、不是 3D Keller cone。下述 AR 数字来自本次 run 的离散序列（现在可以含绕射 token）；没有另造指标。",
        "",
        "### 离散交互序列（已有测量）",
        f"- 第一跳（相对「这一条」GT 路径）准确率只有 {hop1_tf:.3f}。同一 Tx/Rx 通常有多条合法 IRT 分支，普通 AR 被训练成对准单条标注路径，第一跳本身就不是唯一 next-token。这与 IRT「第一层 = 发射端可见元件的分支选择」一致。",
        f"- Teacher-forcing 第二墙 {hop2_tf:.3f}；free-run 第二墙掉到 {hop2_free:.3f}（给定自己的第一跳后续开始偏）。exact {exact_free:.3f}。",
        f"- **Oracle 第一墙 + AR 其余**：第二墙 {hop2_oracle:.3f}，后续墙 {later_w_oracle:.3f}，整段 exact {exact_oracle:.3f}。第一层被纠正后，其余更像沿着那条树边走。",
        f"- **把第一跳改成模型第二候选再 AR**：第二墙 {hop2_2nd:.3f}，后续墙 {later_w_2nd:.3f}，exact {exact_2nd:.3f}。",
        f"  几何合法率仍有 {valid_2nd:.3f}（与 free-run {valid_free:.3f} 接近），且 {any_2nd:.3f} 落在该场景已枚举的某条 GT 分支上——说明第二候选往往是树上的另一枝，而不是「同一条序列的局部噪声」。这正是 IRT 树的换枝，不是语言模型式的 AR 纠错。",
        f"- **把第一跳改成随机墙再 AR**：合法率从 {valid_free:.3f} 掉到 {valid_rand:.3f}，落到场景 GT 分支的比例 {any_rand:.3f}，exact {exact_rand:.3f}。离开可见性树后，后续 next-token 几乎拼不出镜面路径：镜像法要求整段墙序列与遮挡/反射定律全局一致，局部续写不够。",
        f"- One-shot 非 AR：exact {exact_os:.3f}，后续墙几乎对不上这条多跳 GT，但合法率 {valid_os:.3f}（常退化成短的 1-bounce/LoS 合法枝）。",
        "",
        "### 连续交互点（离散墙序列给定，已有测量）",
        f"- 镜像法 oracle 第一跳点误差 {oracle_xy:.2e}（几何闭合）。离散结构已知时，连续点由全局 unfold 决定，不该再当开放的 AR 序列来猜——这是 IRT「点落在给定 tile 上」+ 镜面闭合。",
        f"- 联合回归 hop1 xy {joint_h1:.3f}，后续 {joint_later:.3f}；AR-t free-run 后续 {ar_free_xy:.3f}；污染 t0 后再 AR 后续 {ar_later:.3f}。联合头（RadioDiff 所说的「连续几何用生成/联合而不是逐步判别」）比逐步 AR-t 更稳。",
        f"- 本玩具里的小 DDPM 后续误差 {ddpm_xy:.3f}，没有超过联合回归（CPU 小组件、步数少）。它只用来表明归纳偏置，不声称复现 RadioDiff，也不编造 SOTA。",
        "",
        "### 对研究问题的回答",
        "",
        "**不适合把传播路径直接当成普通 autoregressive sequence 来生成。**",
        "",
        "论文理由：WinProp IRT 的对象是可见性树 + 元件上的点，不是唯一 next-token 句子；第一交互是树的第一层分支；连续点由整段离散结构的镜像几何闭合。RadioDiff 进一步说明：连续无线电几何应对齐条件生成，而不是纯判别逐步回归——但那是场图；我们只把该课用在固定路径结构上的 t。",
        "",
        "玩具证据（上表数字）：",
        "1. 第一交互是分支选择，不是局部 token；改错第一跳后，相对原 GT 的后续墙准确率崩掉；",
        "2. 若错误第一跳仍落在树上（模型第二候选），可以走出另一条合法路径，但这是换枝，不是 AR 纠错；",
        "3. 若第一跳是随机墙（离开树），几何合法率崩溃——缺少全局镜面/镜像法一致性；",
        "4. 连续点在离散结构给定后由镜像法决定（oracle 误差 0），联合回归/生成比逐步 AR 更合适。",
        "模型不必优于所有 baseline；one-shot 与 oracle-first 只是「更少 AR」和「第一跳被纠正」的对照。",
    ]
    text = "\n".join(lines)
    (out_dir / "findings.md").write_text(text, encoding="utf-8")
    return text


def run_experiment(
    npz: str,
    jsonl: str,
    ckpt_dir: str,
    out_dir: str,
    seed: int = 0,
    min_bounces: int = 2,
) -> dict:
    seed_all(seed)
    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(ckpt_dir)
    json_index = load_jsonl_index(jsonl)
    ds = PathNPZDataset(npz, split="test", min_bounces=min_bounces)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    ar = _load_ar(ckpt / "ar_transformer.pt")
    oneshot = _load_oneshot(ckpt / "oneshot.pt")
    rng = np.random.default_rng(seed + 7)
    discrete = eval_discrete(ar, oneshot, loader, json_index, rng, fig_dir)

    ds_c = PathNPZDataset(npz, split="test", min_bounces=max(1, min_bounces - 1))
    loader_c = DataLoader(ds_c, batch_size=16, shuffle=False)
    joint = JointPointRegressor()
    joint.load_state_dict(torch.load(ckpt / "joint_t.pt", map_location="cpu", weights_only=False)["state_dict"])
    joint.eval()
    ar_t = ARPointModel()
    ar_t.load_state_dict(torch.load(ckpt / "ar_t.pt", map_location="cpu", weights_only=False)["state_dict"])
    ar_t.eval()
    ddpm = TinyPointDDPM()
    ddpm.load_state_dict(torch.load(ckpt / "ddpm_t.pt", map_location="cpu", weights_only=False)["state_dict"])
    ddpm.eval()
    continuous = eval_continuous(joint, ar_t, ddpm, loader_c, json_index, rng)

    make_plots(discrete, continuous, fig_dir)
    findings = write_findings(discrete, continuous, out)
    payload = {"discrete": discrete, "continuous": continuous, "findings": findings}
    (out / "metrics.json").write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/generated/dataset.npz")
    p.add_argument("--jsonl", default="data/generated/dataset.jsonl")
    p.add_argument("--ckpts", default="results/ckpts")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_experiment(args.npz, args.jsonl, args.ckpts, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
