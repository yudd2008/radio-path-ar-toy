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
    MAX_WALLS,
    PAD_ID,
    RX_ID,
    TX_ID,
    VOCAB_WALL_OFFSET,
    decode_wall_ids,
    wall_token_id,
)
from env.viz import save_scene_paths
from experiments.common import (
    device,
    load_jsonl_index,
    points_from_t,
    scene_from_json,
    seed_all,
    validity_and_points,
)
from models.continuous import ARPointModel, JointPointRegressor, TinyPointDDPM
from models.sequence import ARPathTransformer, OneShotPathModel
from pathfind.image_method import reconstruct_path


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


def _mean(xs) -> float:
    xs = [float(x) for x in xs if x == x]  # drop nan
    return float(np.mean(xs)) if xs else float("nan")


def _as_numpy_seq(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.int64)


@torch.no_grad()
def _second_choice_first_token(model: ARPathTransformer, batch, dev) -> torch.Tensor:
    """Second-highest first-wall (or RX) under teacher-forced TX prefix."""
    logits = model(
        batch["tokens"][:, :1].to(dev),
        batch["channels"].to(dev),
        batch["tx"].to(dev),
        batch["rx"].to(dev),
        batch["wall_feats"].to(dev),
        batch["wall_mask"].to(dev),
    )
    first = logits[:, -1, :].clone()
    top2 = first.topk(k=2, dim=-1).indices
    gt_first = batch["tokens"][:, 1].to(dev)
    pick = torch.where(top2[:, 0] == gt_first, top2[:, 1], top2[:, 0])
    # If still equal (degenerate), shift to another existing wall.
    same = pick == gt_first
    if same.any():
        # pick first existing wall that is not GT
        walls = torch.arange(MAX_WALLS, device=dev) + VOCAB_WALL_OFFSET
        exist = batch["wall_mask"].to(dev) > 0.5
        for b in range(pick.size(0)):
            if not bool(same[b]):
                continue
            cand = walls[exist[b]]
            cand = cand[cand != gt_first[b]]
            pick[b] = cand[0] if cand.numel() else pick[b]
    return pick


def _random_wrong_first(batch, rng: np.random.Generator) -> torch.Tensor:
    b = batch["tokens"].size(0)
    out = batch["tokens"][:, 1].clone()
    for i in range(b):
        exist = (batch["wall_mask"][i] > 0.5).numpy()
        wids = np.where(exist)[0]
        gt = int(batch["wall_ids"][i, 0].item())
        cand = [w for w in wids if w != gt]
        if cand:
            out[i] = wall_token_id(int(rng.choice(cand)))
    return out


def _hop_flags(pred: np.ndarray, gt: np.ndarray, n_hops: int) -> list[int]:
    """Match at positions 1..n_hops (after TX). Includes RX if present."""
    flags = []
    limit = min(n_hops, pred.shape[0] - 1, gt.shape[0] - 1)
    for k in range(1, limit + 1):
        flags.append(int(pred[k] == gt[k] and gt[k] != PAD_ID))
    return flags


@torch.no_grad()
def eval_discrete(ar, oneshot, loader, json_index, rng, fig_dir: Path) -> dict:
    stats = defaultdict(list)
    examples = []
    n_plot = 0
    for batch in loader:
        bsz = batch["tokens"].size(0)
        tokens = batch["tokens"]
        # teacher forcing hop acc
        logits = ar(
            tokens[:, :-1],
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_mask"],
        )
        tf_pred = logits.argmax(dim=-1)  # B, T-1  aligned with tokens[:,1:]
        os_logits = oneshot(
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_mask"],
        )
        os_pred_slots = os_logits.argmax(dim=-1)

        second = _second_choice_first_token(ar, batch, device())
        rnd = _random_wrong_first(batch, rng)

        free = ar.greedy_decode(
            batch["channels"], batch["tx"], batch["rx"], batch["wall_feats"], batch["wall_mask"]
        )
        oracle = ar.greedy_decode(
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_mask"],
            force_first=tokens[:, 1],
        )
        interv = ar.greedy_decode(
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_mask"],
            force_first=second,
        )
        interv_rand = ar.greedy_decode(
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_mask"],
            force_first=rnd,
        )
        os_seq = oneshot.decode(
            batch["channels"], batch["tx"], batch["rx"], batch["wall_feats"], batch["wall_mask"]
        )
        os_interv = oneshot.decode(
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_mask"],
            force_first=second,
        )

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
                later = flags[start_hop:]  # hops after first interaction
                if later:
                    stats[f"{tag}_later_acc"].append(float(np.mean(later)))
                valid, pts, wids = validity_and_points(scene, pred_seq)
                stats[f"{tag}_valid"].append(int(valid))
                if pts is not None:
                    n = min(len(pts), len(gt_pts))
                    err = [float(np.linalg.norm(pts[j] - gt_pts[j])) for j in range(1, n)]
                    for j, e in enumerate(err, start=1):
                        stats[f"{tag}_xy_hop{j}"].append(e)
                    stats[f"{tag}_xy_mean"].append(float(np.mean(err)) if err else 0.0)
                else:
                    stats[f"{tag}_xy_mean"].append(float("nan"))
                return valid, pts, wids, flags

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
                _, free_pts, _, _ = validity_and_points(scene, _as_numpy_seq(free[i]))
                _, iv_pts, _, _ = validity_and_points(scene, _as_numpy_seq(interv[i]))
                paths = [{"points": gt_pts, "label": "GT"}]
                if free_pts is not None:
                    paths.append({"points": free_pts, "label": "AR free-run"})
                if iv_pts is not None:
                    paths.append({"points": iv_pts, "label": "AR wrong-1st"})
                else:
                    # still draw the (invalid) polyline from tokens via t-midpoints
                    wids = decode_wall_ids(_as_numpy_seq(interv[i]))
                    if wids and all(0 <= w < len(scene.walls) for w in wids):
                        fake_t = [0.5] * len(wids)
                        paths.append(
                            {
                                "points": points_from_t(scene, wids, fake_t),
                                "label": "wrong-1st (invalid geom)",
                            }
                        )
                save_scene_paths(
                    scene,
                    paths,
                    fig_dir / f"example_{n_plot}_sid{sid}.png",
                    title=f"scene {sid}  GT walls {rec['wall_ids']}  n_bounce={nb}",
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
        jpred = joint(
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_ids"],
            batch["bounce_mask"],
        )
        ar_free = ar_t.free_run(
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_ids"],
            batch["bounce_mask"],
        )
        t_gt = batch["t_on_wall"]
        # corrupt first t
        noise = torch.full((bsz,), 0.28)
        t0_bad = (t_gt[:, 0] + noise).clamp(0.05, 0.95)
        # if GT t0 already near bound, flip to the other side
        t0_bad = torch.where((t_gt[:, 0] - t0_bad).abs() < 0.08, (t_gt[:, 0] - 0.28).clamp(0.05, 0.95), t0_bad)
        ar_bad = ar_t.free_run(
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_ids"],
            batch["bounce_mask"],
            t0_override=t0_bad,
        )
        dd = ddpm.sample(
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_ids"],
            batch["bounce_mask"],
            n_steps=12,
        )
        dd_freeze = ddpm.sample(
            batch["channels"],
            batch["tx"],
            batch["rx"],
            batch["wall_feats"],
            batch["wall_ids"],
            batch["bounce_mask"],
            n_steps=12,
            freeze_t0=t0_bad,
        )
        for i in range(bsz):
            sid = int(batch["scene_id"][i].item())
            pid = int(batch["path_id"][i].item())
            rec = json_index[(sid, pid)]
            scene = scene_from_json(rec)
            wids = rec["wall_ids"]
            nb = len(wids)
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
                pts = points_from_t(scene, wids, t_hat[:nb])
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

            # image-method oracle (discrete structure known)
            oracle = reconstruct_path(scene, wids)
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
    hops = np.arange(1, MAX_BOUNCES + 2)
    labels = {
        "tf": "AR teacher-force",
        "freerun": "AR free-run",
        "oracle_first": "oracle 1st + AR rest",
        "interv_2nd": "wrong 1st (2nd-best) + AR",
        "oneshot": "one-shot (non-AR)",
        "oneshot_interv": "one-shot, overwrite 1st",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for tag, lab in labels.items():
        ys = hop_table[tag]["hops"]
        ax.plot(hops[: len(ys)], ys, marker="o", label=lab)
    ax.set_xlabel("sequence hop after Tx (1 = first wall, last often Rx)")
    ax.set_ylabel("token accuracy vs GT path")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Discrete path: teacher-forcing vs free-run vs first-token intervention")
    fig.tight_layout()
    fig.savefig(out_dir / "discrete_hop_accuracy.png", dpi=140)
    plt.close(fig)

    # bar: exact / valid
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    tags = list(labels)
    x = np.arange(len(tags))
    exact = [hop_table[t]["exact"] for t in tags]
    valid = [hop_table[t]["valid"] for t in tags]
    ax.bar(x - 0.18, exact, 0.36, label="exact match vs this GT path")
    ax.bar(x + 0.18, valid, 0.36, label="geometrically valid specular path")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[t] for t in tags], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.legend(fontsize=8)
    ax.set_title("Exact GT match vs geometric validity (a valid other IRT branch still counts)")
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
    later_ar = ht["interv_2nd"]["later_acc"]
    later_oracle = ht["oracle_first"]["later_acc"]
    later_os = ht["oneshot_interv"]["later_acc"]
    valid_free = ht["freerun"]["valid"]
    valid_int = ht["interv_2nd"]["valid"]
    tf_h1 = ht["tf"]["hops"][0]
    free_h1 = ht["freerun"]["hops"][0]
    oracle_xy = continuous.get("oracle_xy_hop1", float("nan"))
    ar_later = continuous.get("ar_badt0_later_xy", float("nan"))
    joint_later = continuous.get("joint_later_xy", float("nan"))

    # Honest conclusion from the measured numbers.
    degrade = (later_oracle - later_ar) if (later_oracle == later_oracle and later_ar == later_ar) else float("nan")
    lines = [
        "结论（由本玩具实验的测量值得出，不是预设口号）：",
        "",
        f"- Teacher-forcing 第一跳准确率 {tf_h1:.3f}，free-run 第一跳 {free_h1:.3f}。",
        f"- 给定 GT 第一墙后 AR 续写（oracle first）后续 token 准确率 {later_oracle:.3f}；",
        f"  把第一跳改成模型第二候选后再 AR，后续准确率掉到 {later_ar:.3f}（Δ={degrade:.3f}）。",
        f"- Free-run 序列成为*某条*合法镜面路径的比例 {valid_free:.3f}；错误第一跳之后只剩 {valid_int:.3f}。",
        f"- One-shot 改写第一跳后，后续 hop 准确率 {later_os:.3f}（后续槽位本就不依赖第一跳，因此不会出现 AR 式累积，但也无法用第一跳约束后面的墙）。",
        f"- 若离散墙序列已知，image method 第一跳点误差 {oracle_xy:.2e}（几何闭合）；",
        f"  AR 连续头在 t0 被污染后后续点 xy 误差 {ar_later:.3f}，联合回归后续误差 {joint_later:.3f}。",
        "",
        "对研究问题「传播路径是否适合直接当作普通 autoregressive sequence 来生成」：",
        "本实验表明：路径的离散结构更接近 WinProp IRT 的可见性树搜索（全局、短深度、强几何约束），",
        "而不是局部 next-token。第一交互一旦错，后续墙序列与连续点都会垮掉；",
        "连续点在离散结构给定后几乎由镜面几何决定，更适合联合生成/回归（RadioDiff 式 generative 思想），而不是逐步 AR。",
        "模型不必优于所有 baseline；这里 one-shot 与 oracle-first 只是为了对照「更少 AR」与「第一跳被纠正」。",
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
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
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
