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


def _select_example_keys(json_index: dict) -> set[tuple[int, int]]:
    """Stable mix of reflection, diffraction, and mixed test paths (n_bounces≥2)."""
    buckets: dict[str, list[tuple[int, int]]] = {
        "reflection": [],
        "diffraction": [],
        "mixed": [],
    }
    for (sid, pid), rec in sorted(json_index.items()):
        if rec.get("split") != "test":
            continue
        if int(rec.get("n_bounces", 0)) < 2:
            continue
        mech = str(rec.get("mechanism") or "")
        if mech in buckets:
            buckets[mech].append((int(sid), int(pid)))
    chosen: list[tuple[int, int]] = []
    for mech, quota in (("reflection", 3), ("diffraction", 3), ("mixed", 2)):
        chosen.extend(buckets[mech][:quota])
    return set(chosen)


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
    example_keys = _select_example_keys(json_index)
    for stale in fig_dir.glob("example_*.png"):
        stale.unlink()
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

            if (sid, pid) in example_keys:
                _, free_pts, _ = validity_and_points(scene, _as_numpy_seq(free[i]))
                _, iv_pts, iv_inter = validity_and_points(scene, _as_numpy_seq(interv[i]))
                mech = rec.get("mechanism", "reflection")
                paths = [{"points": gt_pts, "label": f"GT {mech}", "mechanism": mech}]
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
                tok_s = " → ".join(rec.get("tokens") or [])
                save_scene_paths(
                    scene,
                    paths,
                    fig_dir / f"example_{n_plot}_sid{sid}.png",
                    title=f"scene {sid}  {tok_s}  n={nb}",
                    wall_labels=True,
                )
                n_plot += 1
                examples.append(
                    {
                        "scene_id": sid,
                        "path_id": pid,
                        "mechanism": rec.get("mechanism"),
                        "gt": rec["tokens"],
                    }
                )

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
        "interv_rand": "wrong 1st (random R/D) + AR",
        "oneshot": "one-shot (non-AR)",
        "oneshot_interv": "one-shot, overwrite 1st",
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for tag, lab in labels.items():
        ys = hop_table[tag]["hops"][:3]
        ax.plot(np.arange(1, len(ys) + 1), ys, marker="o", label=lab)
    ax.set_xlabel("sequence hop after Tx (1 = first R_wall/D_corner, 2 = second, 3 often Rx)")
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
    n_cont = continuous.get("n_eval")
    ax.set_title(
        "Continuous points: error vs hop"
        + (f" (n={int(n_cont)}, GT discrete R/D)" if isinstance(n_cont, (int, float)) and n_cont == n_cont else "")
    )
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_error_growth.png", dpi=140)
    plt.close(fig)


def _fmt(x, digits: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v or v in (float("inf"), float("-inf")):
        return "n/a"
    return f"{v:.{digits}f}"


def _hop_line(ht: dict, tag: str) -> str:
    hops = ht[tag]["hops"]
    return " / ".join(_fmt(h) for h in hops[:4])


def write_findings(
    discrete: dict,
    continuous: dict,
    out_dir: Path,
    continuous_ge2: dict | None = None,
) -> str:
    ht = discrete["hop_table"]
    n_eval = int(discrete["summary"].get("n_eval", 0))
    n_wide = int(continuous.get("n_eval", 0))
    hop1_tf = ht["tf"]["hops"][0]
    hop2_tf = ht["tf"]["hops"][1]
    hop2_free = ht["freerun"]["hops"][1]
    hop2_oracle = ht["oracle_first"]["hops"][1]
    hop2_2nd = ht["interv_2nd"]["hops"][1]
    later_w_oracle = ht["oracle_first"]["later_wall_acc"]
    later_w_2nd = ht["interv_2nd"]["later_wall_acc"]
    exact_tf = ht["tf"]["exact"]
    exact_free = ht["freerun"]["exact"]
    exact_oracle = ht["oracle_first"]["exact"]
    exact_2nd = ht["interv_2nd"]["exact"]
    exact_rand = ht["interv_rand"]["exact"]
    exact_os = ht["oneshot"]["exact"]
    valid_tf = ht["tf"]["valid"]
    valid_free = ht["freerun"]["valid"]
    valid_2nd = ht["interv_2nd"]["valid"]
    valid_rand = ht["interv_rand"]["valid"]
    valid_os = ht["oneshot"]["valid"]
    any_2nd = ht["interv_2nd"]["any_scene_path"]
    any_rand = ht["interv_rand"]["any_scene_path"]
    os_hop2 = ht["oneshot"]["hops"][1]
    os_h1 = ht["oneshot"]["hops"][0]
    oracle_xy = continuous.get("oracle_xy_hop1", float("nan"))
    ar_later = continuous.get("ar_badt0_later_xy", float("nan"))
    joint_later = continuous.get("joint_later_xy", float("nan"))
    ar_free_xy = continuous.get("ar_free_later_xy", float("nan"))
    ddpm_xy = continuous.get("ddpm_later_xy", float("nan"))
    joint_h1 = continuous.get("joint_xy_hop1", float("nan"))

    if float(hop2_tf) < 0.05:
        hop2_sentence = (
            f"- 不加权小 AR 的 teacher-forcing 第二交互只有 {_fmt(hop2_tf)}，free-run {_fmt(hop2_free)}，"
            f"整段 exact {_fmt(exact_free)}（TF exact {_fmt(exact_tf)}）。"
            "1-bounce 路径的第二 token 就是 RX，不加权交叉熵把 hop2 收成 RX，所以这条原配方几乎没学到第二跳的 `R_wall_k` / `D_corner_c`。"
        )
    else:
        hop2_sentence = (
            f"- Teacher-forcing 第二交互 {_fmt(hop2_tf)}；free-run 掉到 {_fmt(hop2_free)}。"
            f"free-run exact {_fmt(exact_free)}（TF exact {_fmt(exact_tf)}）。"
        )

    if float(exact_2nd) < 0.02 and float(any_2nd) > 0.4:
        second_note = (
            f"exact {_fmt(exact_2nd)}，但合法率 {_fmt(valid_2nd)}（free-run {_fmt(valid_free)}），"
            f"其中 {_fmt(any_2nd)} 落在该场景已枚举的某条 GT 枝上：第二候选多半是换枝，不是把原序列修补回来。"
        )
    else:
        second_note = (
            f"exact {_fmt(exact_2nd)}，合法率 {_fmt(valid_2nd)}（free-run {_fmt(valid_free)}），"
            f"落在场景已枚举 GT 枝的比例 {_fmt(any_2nd)}。"
        )

    if float(ddpm_xy) > float(joint_later) + 0.01:
        ddpm_sentence = (
            f"小 DDPM 后续 xy {_fmt(ddpm_xy)}，高于联合回归的 {_fmt(joint_later)}（CPU 小组件、12 步采样）。"
            "它只标出「连续几何可以联合生成」这个归纳，不声称复现 RadioDiff。"
        )
    elif float(ddpm_xy) + 0.01 < float(joint_later):
        ddpm_sentence = (
            f"小 DDPM 后续 xy {_fmt(ddpm_xy)}，低于联合回归的 {_fmt(joint_later)}。"
            "这是这次 CPU 小组件的测量，不声称复现 RadioDiff，也不把它写成 SOTA。"
        )
    else:
        ddpm_sentence = (
            f"小 DDPM 后续 xy {_fmt(ddpm_xy)}，联合回归 {_fmt(joint_later)}，两者接近。"
            "不声称复现 RadioDiff。"
        )

    if float(joint_later) + 0.005 < float(ar_free_xy):
        joint_vs_ar = "联合回归的后续 xy 低于逐步 AR-t。"
    elif float(ar_free_xy) + 0.005 < float(joint_later):
        joint_vs_ar = "逐步 AR-t 的后续 xy 低于联合回归；以表中的数字为准，不把这个差写成方法结论。"
    else:
        joint_vs_ar = "联合回归与逐步 AR-t 的后续 xy 接近。"

    ge2_lines: list[str] = []
    if continuous_ge2 is not None:
        ge2_lines = [
            f"- 同一批 n_bounces≥2 路径（n={int(continuous_ge2.get('n_eval', 0))}，与离散 / Transformer 同一条条路径）上的误差：",
            (
                f"  联合 hop1 xy {_fmt(continuous_ge2.get('joint_xy_hop1'))}，后续 {_fmt(continuous_ge2.get('joint_later_xy'))}；"
                f"AR-t free-run 后续 {_fmt(continuous_ge2.get('ar_free_later_xy'))}；"
                f"污染 t0 后再 AR 后续 {_fmt(continuous_ge2.get('ar_badt0_later_xy'))}；"
                f"DDPM 后续 {_fmt(continuous_ge2.get('ddpm_later_xy'))}；"
                f"oracle hop1 xy {_fmt(continuous_ge2.get('oracle_xy_hop1'), 2)}。"
            ),
            "  逐跳 xy（hop1 / hop2 / hop3）：",
            (
                f"  联合 {_fmt(continuous_ge2.get('joint_xy_hop1'))} / {_fmt(continuous_ge2.get('joint_xy_hop2'))} / {_fmt(continuous_ge2.get('joint_xy_hop3'))}；"
                f"AR-t {_fmt(continuous_ge2.get('ar_free_xy_hop1'))} / {_fmt(continuous_ge2.get('ar_free_xy_hop2'))} / {_fmt(continuous_ge2.get('ar_free_xy_hop3'))}；"
                f"AR-t 污染 t0 {_fmt(continuous_ge2.get('ar_badt0_xy_hop1'))} / {_fmt(continuous_ge2.get('ar_badt0_xy_hop2'))} / {_fmt(continuous_ge2.get('ar_badt0_xy_hop3'))}；"
                f"DDPM {_fmt(continuous_ge2.get('ddpm_xy_hop1'))} / {_fmt(continuous_ge2.get('ddpm_xy_hop2'))} / {_fmt(continuous_ge2.get('ddpm_xy_hop3'))}；"
                f"DDPM 冻 t0 {_fmt(continuous_ge2.get('ddpm_freeze_t0_xy_hop1'))} / {_fmt(continuous_ge2.get('ddpm_freeze_t0_xy_hop2'))} / {_fmt(continuous_ge2.get('ddpm_freeze_t0_xy_hop3'))}。"
            ),
        ]

    lines = [
        f"结论（离散 n={n_eval} 条测试路径，n_bounces≥2；与 `results/transformer_comparison.json` 同一 seed=0 划分。数字来自衍射 token 重跑，不是预设口号）。",
        "论文对照见 [`docs/paper_notes.md`](../docs/paper_notes.md)。",
        "",
        "### 论文主张（本玩具对齐的那几条）",
        "",
        "**WinProp IRT**（Altair 用户指南；Hoppe et al., EPMCC 1999）：寻径是在预处理好的墙面/tile 可见性关系上做 **树搜索**；交互点被约束在这些离散元件上；交互次数很少（文档称最多约三次即可）。树上每一枝是「两个元件之间的可见性关系」，预测时先展开发射端可见的第一层，再递归检查反射条件。",
        "→ 本玩具：`R_wall_k` / `D_corner_c` token = 树节点（反射墙或绕射角点）；合法 token 邻接 = 树边；反射的连续 t 落在该墙上，绕射点就是角点（t 记 0）。普通 AR 把「树」当成「一条句子」的局部 next-token。",
        "",
        "**RadioDiff**（Wang et al., IEEE TCCN 2024）：无采样无线电地图构建更应是 **条件生成**，而不是 RadioUNet 式纯判别 MSE。路径损耗并不已经写在输入里，必须被生成出来。",
        "→ 本玩具 **不重实现 RadioDiff、不生成场图**。只把同一课用在 **固定离散路径结构上的连续交互点**：联合回归/小 DDPM，而不是逐步 AR 猜 t。RadioDiff 生成 pathloss map；我们生成/refinement 的是墙上的点。",
        "",
        "**RadioUNet**：仅作占用栅格 + Tx/Rx 通道的场景编码器上下文，不估计 RM。",
        "",
        "WinProp IRT 也跟踪垂直棱/楔的绕射。本玩具在矩形角点上加了简化的 2D Keller/UTD **存在性**判定（轮廓/前向面、自由空间、最小弯折），token 为 `D_corner_c`。这不是完整 UTD 系数、不是 3D Keller cone。下面的数字就是这次 `R_wall_k` / `D_corner_c` 词表上的测量，没有另造指标。",
        "",
        "### 划分",
        "",
        "seed=0，`generate_dataset` 的 240/50/70 个场景，每个 split 再加一个反射+绕射 showcase 场景。离散表、one-shot、以及下方 Transformer 一节用的都是 test 里 **n_bounces≥2** 的路径，n="
        f"{n_eval}。",
        f"连续点的原实验筛选是 **n_bounces≥1**（LoS 没有交互坐标 t），同一批 test 场景，n={n_wide}，不是 {n_eval}。"
        "t 误差只平均反射跳（`bounce_mask` 丢掉绕射，因为角点坐标已经由 `D_corner_c` 钉死）；xy 误差包含绕射跳，离散结构给定时该跳应为 0。"
        "与离散同一条条路径的连续误差记在 `continuous_n_bounces_ge_2`。",
        "",
        "### 离散交互序列（不加权小 AR + one-shot）",
        "",
        f"- 第一跳（相对「这一条」GT 路径）teacher-forcing 准确率 {_fmt(hop1_tf)}。"
        "同一 Tx/Rx 通常有多条合法反射/绕射枝，单序列监督把树压成一句话，第一跳不是唯一 next-token。",
        hop2_sentence,
        f"- **Oracle 第一交互 + AR 其余**：第二交互 {_fmt(hop2_oracle)}，后续交互 {_fmt(later_w_oracle)}，整段 exact {_fmt(exact_oracle)}。",
        f"- **第一跳改成模型第二候选再 AR**：第二交互 {_fmt(hop2_2nd)}，后续交互 {_fmt(later_w_2nd)}。{second_note}",
        f"- **第一跳改成随机已有墙或角点再 AR**：合法率从 {_fmt(valid_free)} 到 {_fmt(valid_rand)}，"
        f"落到场景 GT 枝的比例 {_fmt(any_rand)}，exact {_fmt(exact_rand)}。",
        f"- One-shot（非 AR）：hop1 {_fmt(os_h1)}，hop2 {_fmt(os_hop2)}，exact {_fmt(exact_os)}，合法率 {_fmt(valid_os)}。",
        f"- Teacher-forcing 拼出来的 token 串合法率 {_fmt(valid_tf)}（这不是一条解码路径，只是逐位 argmax）。",
        "",
        "不加权小 AR，逐跳 token accuracy（hop1 / hop2 / hop3 / hop4）：",
        "",
        "| 设定 | hop1 | hop2 | hop3 | hop4 | exact | valid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| teacher-force | {_hop_line(ht, 'tf').replace(' / ', ' | ')} | {_fmt(ht['tf']['exact'])} | {_fmt(ht['tf']['valid'])} |",
        f"| free-run | {_hop_line(ht, 'freerun').replace(' / ', ' | ')} | {_fmt(ht['freerun']['exact'])} | {_fmt(ht['freerun']['valid'])} |",
        f"| oracle 第一交互 | {_hop_line(ht, 'oracle_first').replace(' / ', ' | ')} | {_fmt(ht['oracle_first']['exact'])} | {_fmt(ht['oracle_first']['valid'])} |",
        f"| 第二候选第一跳 | {_hop_line(ht, 'interv_2nd').replace(' / ', ' | ')} | {_fmt(ht['interv_2nd']['exact'])} | {_fmt(ht['interv_2nd']['valid'])} |",
        f"| 随机第一跳 | {_hop_line(ht, 'interv_rand').replace(' / ', ' | ')} | {_fmt(ht['interv_rand']['exact'])} | {_fmt(ht['interv_rand']['valid'])} |",
        f"| one-shot | {_hop_line(ht, 'oneshot').replace(' / ', ' | ')} | {_fmt(ht['oneshot']['exact'])} | {_fmt(ht['oneshot']['valid'])} |",
        f"| one-shot 改第一跳 | {_hop_line(ht, 'oneshot_interv').replace(' / ', ' | ')} | {_fmt(ht['oneshot_interv']['exact'])} | {_fmt(ht['oneshot_interv']['valid'])} |",
        "",
        "交互加权的小 AR 和 `SeriousPathTransformer` 用同一 n、同一第一跳协议，数字在下面的 Transformer 一节（checkpoint 未重训，只复核）。",
        "",
        "### 连续交互点（离散 R/D 序列给定）",
        "",
        f"- 筛选 n_bounces≥1，n={n_wide}。镜像法 oracle 第一跳 xy {_fmt(oracle_xy, 2)}（几何闭合）。",
        f"- 联合回归 hop1 xy {_fmt(joint_h1)}，后续 {_fmt(joint_later)}；AR-t free-run 后续 {_fmt(ar_free_xy)}；污染 t0 后再 AR 后续 {_fmt(ar_later)}。{joint_vs_ar}",
        f"- {ddpm_sentence}",
        "逐跳 xy（hop1 / hop2 / hop3），n_bounces≥1：",
        (
            f"- 联合 {_fmt(continuous.get('joint_xy_hop1'))} / {_fmt(continuous.get('joint_xy_hop2'))} / {_fmt(continuous.get('joint_xy_hop3'))}；"
            f"AR-t {_fmt(continuous.get('ar_free_xy_hop1'))} / {_fmt(continuous.get('ar_free_xy_hop2'))} / {_fmt(continuous.get('ar_free_xy_hop3'))}；"
            f"AR-t 污染 t0 {_fmt(continuous.get('ar_badt0_xy_hop1'))} / {_fmt(continuous.get('ar_badt0_xy_hop2'))} / {_fmt(continuous.get('ar_badt0_xy_hop3'))}；"
            f"DDPM {_fmt(continuous.get('ddpm_xy_hop1'))} / {_fmt(continuous.get('ddpm_xy_hop2'))} / {_fmt(continuous.get('ddpm_xy_hop3'))}；"
            f"DDPM 冻 t0 {_fmt(continuous.get('ddpm_freeze_t0_xy_hop1'))} / {_fmt(continuous.get('ddpm_freeze_t0_xy_hop2'))} / {_fmt(continuous.get('ddpm_freeze_t0_xy_hop3'))}；"
            f"oracle {_fmt(continuous.get('oracle_xy_hop1'), 2)} / {_fmt(continuous.get('oracle_xy_hop2'), 2)} / {_fmt(continuous.get('oracle_xy_hop3'), 2)}。"
        ),
        *ge2_lines,
        "",
        "### 对研究问题的回答",
        "",
        "**不适合把传播路径直接当成普通 autoregressive sequence 来生成。**",
        "",
        "论文理由：WinProp IRT 的对象是可见性树 + 元件上的点，不是唯一 next-token 句子；第一交互是树的第一层分支；连续点由整段离散结构的镜像几何闭合。绕射顶点没有自由 t。RadioDiff 说明的是场图应对齐条件生成；本玩具只把该课用在固定路径结构上的反射 t。",
        "",
        "这次测量：",
        f"1. 不加权小 AR 的第一跳准确率 {_fmt(hop1_tf)}，第二跳 {_fmt(hop2_tf)}。第一交互是分支，而且不加权损失学不会 `R_wall_k` / `D_corner_c` 的第二跳。",
        f"2. 第一跳换成第二候选后 exact {_fmt(exact_2nd)}，合法率 {_fmt(valid_2nd)}，场景 GT 枝 {_fmt(any_2nd)}。",
        f"3. 第一跳换成随机已有墙/角点后合法率 {_fmt(valid_rand)}（free-run {_fmt(valid_free)}），exact {_fmt(exact_rand)}。",
        f"4. 离散结构给定后，镜像法 oracle 的第一跳 xy 是 {_fmt(oracle_xy, 2)}。连续点不该再当开放 AR 序列来猜。",
        "交互加权小 AR 与 4 层 Transformer 是否改变这个结论，看下面同一 n 上的对照，不在这里另写一套数字。",
        "模型不必优于所有 baseline；one-shot 与 oracle-first 只是「更少 AR」和「第一跳被纠正」的对照。",
    ]
    text = "\n".join(lines) + "\n"
    path = out_dir / "findings.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    begin = "<!-- TRANSFORMER_BASELINE_BEGIN -->"
    end = "<!-- TRANSFORMER_BASELINE_END -->"
    if begin in existing and end in existing:
        section = existing[existing.index(begin) : existing.index(end) + len(end)]
        file_text = text.rstrip() + "\n\n" + section.rstrip() + "\n"
    else:
        file_text = text
    path.write_text(file_text, encoding="utf-8")
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
    # Historical continuous filter: n_bounces>=1 on the same test scenes.
    # Torch RNG is whatever discrete eval left, matching the original call order.
    continuous = eval_continuous(joint, ar_t, ddpm, loader_c, json_index, rng)
    continuous["path_filter"] = "split=test, n_bounces>=1"
    continuous["same_scenes_as_discrete"] = True
    continuous["differs_from_discrete_n"] = (
        "LoS paths have no interaction coordinate. This block keeps every test path "
        "with at least one interaction. It is not the n_bounces>=2 path list."
    )
    # Same paths as the discrete / Transformer table. Reseed so this DDPM draw
    # does not depend on how many torch ops the wider eval consumed.
    seed_all(seed)
    ds_same = PathNPZDataset(npz, split="test", min_bounces=min_bounces)
    loader_same = DataLoader(ds_same, batch_size=16, shuffle=False)
    continuous_ge2 = eval_continuous(joint, ar_t, ddpm, loader_same, json_index, rng)
    continuous_ge2["path_filter"] = "split=test, n_bounces>=2"
    continuous_ge2["same_paths_as_discrete_and_transformer"] = True
    continuous_ge2["ddpm_seed"] = (
        "seed_all(seed) immediately before this eval; "
        "the n_bounces>=1 block above is not reseeded"
    )

    make_plots(discrete, continuous_ge2, fig_dir)
    findings = write_findings(discrete, continuous, out, continuous_ge2=continuous_ge2)
    payload = {
        "protocol": {
            "seed": seed,
            "scenes": "generate_dataset 240/50/70 plus one showcase scene per split",
            "tokens": "TX / R_wall_k / D_corner_c / RX",
            "discrete_filter": "split=test, n_bounces>=2",
            "discrete_n_eval": discrete["summary"].get("n_eval"),
            "same_paths_as": "results/transformer_comparison.json",
            "continuous_filter": "split=test, n_bounces>=1",
            "continuous_n_eval": continuous.get("n_eval"),
            "continuous_same_paths_filter": "split=test, n_bounces>=2",
            "continuous_same_paths_n_eval": continuous_ge2.get("n_eval"),
            "continuous_t_error": "reflection hops only (bounce_mask drops diffraction)",
            "continuous_xy_error": "every interaction hop; diffraction xy is the corner once the discrete token is given",
            "figure_continuous_error_growth": "continuous_n_bounces_ge_2 (same paths as discrete)",
            "sequence_epochs": 16,
            "continuous_epochs": 14,
            "ar_checkpoint": "results/ckpts/ar_transformer.pt (existing weights; a fresh 16-epoch train matched bitwise)",
            "oneshot_checkpoint": "results/ckpts/oneshot.pt trained this run",
            "continuous_checkpoints": "joint_t.pt, ar_t.pt, ddpm_t.pt trained this run",
        },
        "discrete": discrete,
        "continuous": continuous,
        "continuous_n_bounces_ge_2": continuous_ge2,
        "findings": findings,
    }
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
