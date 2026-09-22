"""Re-evaluate the ordinary AR model and the Transformer baseline.

Uses the same test split and the same first-hop corruption protocol as
``experiments.ar_intervention.score_sequence_model`` (teacher forcing,
free-run, oracle first hop, second-best first hop, random existing
wall/corner). Writes metrics, a hop figure, and a marked section in
``results/findings.md``. Does not train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import PathNPZDataset
from experiments.ar_intervention import _jsonable, score_sequence_model
from experiments.common import load_jsonl_index, seed_all
from models.sequence import ARPathTransformer
from models.transformer_seq import build_serious_transformer

BEGIN = "<!-- TRANSFORMER_BASELINE_BEGIN -->"
END = "<!-- TRANSFORMER_BASELINE_END -->"

# Recipe of the paired ordinary-AR run (experiments/train_sequence.py defaults,
# with the epoch count used by run_transformer_baseline).
AR_RECIPE = {
    "name": "ARPathTransformer",
    "d_model": 64,
    "nhead": 4,
    "nlayers": 2,
    "dropout": 0.1,
    "dim_feedforward": 128,
    "positional_encoding": False,
    "norm": "post-norm",
    "optimizer": "AdamW",
    "lr": 2e-3,
    "schedule": "constant",
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 16,
    "selection": "best validation teacher-forced next-token accuracy",
}


def _fmt(x, digits: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v or v in (float("inf"), float("-inf")):
        return "n/a"
    return f"{v:.{digits}f}"


def _load_ar(path: Path) -> tuple[ARPathTransformer, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = ARPathTransformer()
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def _load_transformer(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = build_serious_transformer(blob.get("config"))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def _train_brief_ar(blob: dict) -> dict:
    hist = list(blob.get("hist") or [])
    return {
        "recipe": AR_RECIPE,
        "epochs_ran": len(hist),
        "val_tf_acc": blob.get("val_tf_acc"),
        "best_epoch": (
            [row["epoch"] for row in hist if row.get("val_tf_acc") == blob.get("val_tf_acc")] or [len(hist)]
        )[-1],
        "train_loss_first": hist[0]["train_loss"] if hist else None,
        "train_loss_last": hist[-1]["train_loss"] if hist else None,
        "val_tf_acc_first": hist[0]["val_tf_acc"] if hist else None,
        "val_tf_acc_last": hist[-1]["val_tf_acc"] if hist else None,
        "hist": hist,
    }


def _train_brief_tfm(blob: dict) -> dict:
    hist = list(blob.get("hist") or [])
    return {
        "config": blob.get("config"),
        "train_defaults": blob.get("train_defaults"),
        "n_params": blob.get("n_params"),
        "best_epoch": blob.get("best_epoch"),
        "epochs_ran": blob.get("epochs_ran", len(hist)),
        "stop_reason": blob.get("stop_reason"),
        "train_loss_plateaued": blob.get("train_loss_plateaued"),
        "val_tf_acc": blob.get("val_tf_acc"),
        "selection_metric": blob.get("selection_metric"),
        "n_train": blob.get("n_train"),
        "n_val": blob.get("n_val"),
        "train_loss_first": hist[0]["train_loss"] if hist else None,
        "train_loss_last": hist[-1]["train_loss"] if hist else None,
        "train_tf_acc_last": hist[-1].get("train_tf_acc") if hist else None,
        "val_tf_acc_first": hist[0]["val_tf_acc"] if hist else None,
        "val_tf_acc_last": hist[-1]["val_tf_acc"] if hist else None,
        "val_loss_best": next(
            (row["val_loss"] for row in hist if row.get("epoch") == blob.get("best_epoch")),
            None,
        ),
        "hist": hist,
    }


def render_findings_section(payload: dict) -> str:
    ar = payload["ar"]["hop_table"]
    tfm = payload["transformer"]["hop_table"]
    ar_s = payload["ar"]["summary"]
    tf_s = payload["transformer"]["summary"]
    ar_tr = payload["ar"]["train"]
    tf_tr = payload["transformer"]["train"]
    n_eval = payload["n_eval"]
    cfg = tf_tr.get("config") or {}

    def row(name: str, ht: dict, summary: dict) -> str:
        return (
            f"| {name} | {_fmt(ht['tf']['hops'][0])} | {_fmt(ht['tf']['hops'][1])} | "
            f"{_fmt(ht['tf']['exact'])} | {_fmt(ht['freerun']['hops'][1])} | "
            f"{_fmt(ht['freerun']['exact'])} | {_fmt(ht['freerun']['valid'])} | "
            f"{_fmt(ht['oracle_first']['hops'][1])} | {_fmt(ht['oracle_first']['exact'])} | "
            f"{_fmt(ht['interv_2nd']['hops'][1])} | {_fmt(ht['interv_2nd']['exact'])} | "
            f"{_fmt(ht['interv_2nd']['valid'])} | {_fmt(ht['interv_2nd']['any_scene_path'])} | "
            f"{_fmt(ht['interv_rand']['valid'])} | {_fmt(ht['interv_rand']['exact'])} | "
            f"{_fmt(summary.get('first_token_top2'))} |"
        )

    hop_header = "| 模型 | " + " | ".join(f"hop{k}" for k in range(1, 5)) + " |"
    hop_sep = "| --- | " + " | ".join("---:" for _ in range(4)) + " |"

    def hop_row(name: str, ht: dict, tag: str) -> str:
        hops = ht[tag]["hops"][:4]
        return f"| {name} | " + " | ".join(_fmt(h) for h in hops) + " |"

    tf_exact = float(tfm["freerun"]["exact"])
    ar_exact = float(ar["freerun"]["exact"])
    tf_gain = tf_exact - ar_exact
    both_branch = (
        float(ar["interv_2nd"]["exact"]) < 0.05
        and float(tfm["interv_2nd"]["exact"]) < 0.05
        and float(ar["interv_2nd"]["valid"]) > 0.7
        and float(tfm["interv_2nd"]["valid"]) > 0.7
    )
    both_offtree = (
        float(ar["interv_rand"]["valid"]) + 0.25 < float(ar["freerun"]["valid"])
        and float(tfm["interv_rand"]["valid"]) + 0.25 < float(tfm["freerun"]["valid"])
    )
    if both_branch and both_offtree and tf_gain < 0.2:
        verdict_en = (
            "A larger Transformer trained to a validation plateau does not remove "
            "the two failure modes that make ordinary autoregressive free-run the "
            "wrong object for these multipath sequences: a wrong-but-plausible first "
            "hop switches branches, and an off-tree first hop collapses geometric validity. "
            "Free-run exact match stays far from solving the labeled path."
        )
        verdict_zh = (
            "把因果 Transformer 训到验证集平台期，并没有消掉普通 AR free-run 不适合多径序列的两种失败："
            "第一跳若仍像树上的另一枝，后续是换枝而不是把原 GT 续回来；第一跳离开树，几何合法率就掉下去。"
            "Free-run 对上这一条标注路径的比例仍然低。"
        )
    elif tf_gain >= 0.2 and float(tfm["freerun"]["exact"]) >= 0.5 and not both_branch:
        verdict_en = (
            "On this split the Transformer materially raises free-run exact match "
            "and does not show the same first-hop branch-switch failure. That narrows "
            "the claim: capacity and training were part of the gap for the small AR. "
            "Read the intervention rows before treating sequential free-run as solved."
        )
        verdict_zh = (
            "在这一划分上，Transformer 明显提高了 free-run exact，而且没有表现出同样的第一跳换枝失败。"
            "这说明小 AR 的差距里有容量和训练不足的成分。是否因此就适合把路径当成普通 AR，要看干预行，不能直接当成已经解决。"
        )
    else:
        verdict_en = (
            "Read the table rather than a slogan. Where free-run or a corrupted first "
            "hop still fails, a stronger sequence model has not turned the visibility "
            "tree into an ordinary sentence. Where it gains, the gain is the measured "
            "delta above, not a claim that the Transformer generates image-method paths."
        )
        verdict_zh = (
            "以表里的测量为准。Free-run 或改错第一跳仍然失败的地方，更强的序列模型并没有把可见性树变成普通句子；"
            "有提高的地方，提高量就是上表的差，不是 Transformer 生成了镜像法路径。"
        )

    lines = [
        BEGIN,
        "",
        "### Transformer 序列基线（与普通 AR 同数据、同第一跳干预）",
        "",
        f"配对测量：seed={payload['seed']}，测试路径 n={int(n_eval)}（`n_bounces≥2`）。"
        "划分是 `generate_dataset` 的 240/50/70 个场景再加每个 split 一个 showcase 场景，与 `experiments.run_all` 相同。"
        "Ground truth 仍是镜像法 + 已有角点绕射规则枚举出的路径。普通 AR 和 Transformer 都不是 GT 生成器。",
        "",
        "**训练（验证集 teacher-forced next-token accuracy 选 checkpoint）。**",
        (
            f"- 普通 AR（`ARPathTransformer`）：d={AR_RECIPE['d_model']}，"
            f"{AR_RECIPE['nlayers']} 层 post-norm，无位置编码，FFN {AR_RECIPE['dim_feedforward']}，"
            f"dropout {AR_RECIPE['dropout']}，AdamW lr={AR_RECIPE['lr']} 常数，"
            f"weight decay {AR_RECIPE['weight_decay']}，batch {AR_RECIPE['batch_size']}，"
            f"跑了 {ar_tr.get('epochs_ran')} epoch（本配对脚本指定 {AR_RECIPE['epochs']}）。"
            f"最佳验证 TF acc {_fmt(ar_tr.get('val_tf_acc'))}。"
            f"训练 loss {_fmt(ar_tr.get('train_loss_first'), 4)} → {_fmt(ar_tr.get('train_loss_last'), 4)}。"
        ),
        (
            f"- Transformer（`SeriousPathTransformer`）：d={cfg.get('d_model')}，"
            f"{cfg.get('nlayers')} 层 pre-norm 因果 decoder，学习位置编码，"
            f"FFN 倍数 {cfg.get('ff_mult')}，dropout {cfg.get('dropout')}，"
            f"参数量 {tf_tr.get('n_params')}。"
            f"AdamW（矩阵权重 weight decay 0.01；bias / LayerNorm / 位置表不衰减），"
            f"线性 warmup {((tf_tr.get('train_defaults') or {}).get('warmup_epochs', 'n/a'))} epoch "
            f"后 cosine 降到 min lr，梯度裁剪 1。"
            f"最佳验证 TF acc {_fmt(tf_tr.get('val_tf_acc'))} 出现在 epoch {tf_tr.get('best_epoch')}，"
            f"共跑 {tf_tr.get('epochs_ran')} epoch，停止原因 `{tf_tr.get('stop_reason')}`，"
            f"训练 loss 平台标记 {tf_tr.get('train_loss_plateaued')}。"
            f"训练 loss {_fmt(tf_tr.get('train_loss_first'), 4)} → {_fmt(tf_tr.get('train_loss_last'), 4)}；"
            f"结束时 train TF acc {_fmt(tf_tr.get('train_tf_acc_last'))}，"
            f"该 epoch 的 val TF acc {_fmt(tf_tr.get('val_tf_acc_last'))}。"
            "曲线：`results/figures/transformer_train_curves.png`。"
        ),
        "",
        "协议与 `score_sequence_model` 一致（也就是 `eval_discrete` 里 AR 的那一支）："
        "teacher-forcing；greedy free-run；oracle 第一交互再自由生成；"
        "第一交互换成模型第二候选再自由生成；第一交互换成随机的、已存在的墙或角点 token（不等于标注第一跳）再自由生成。"
        "随机干预的 RNG 是 `np.random.default_rng(seed+7)`，两个模型各用一次相同的种子，因此非法第一跳相同。"
        "exact = 整段 token 对上这一条标注路径。valid = 镜像法/绕射重建成功。any GT branch = 预测交互序列出现在该场景已枚举路径里。",
        "",
        "| 模型 | TF hop1 | TF hop2 | TF exact | free hop2 | free exact | free valid | oracle hop2 | oracle exact | 2nd hop2 | 2nd exact | 2nd valid | 2nd 落在场景某条 GT 枝 | rand valid | rand exact | 第一跳 top-2 含 GT |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        row("普通 AR", ar, ar_s),
        row("Transformer", tfm, tf_s),
        "",
        "Teacher-forcing 与 free-run 的逐跳 token accuracy（hop1 = 第一交互，之后常含第二交互与 RX）：",
        "",
        hop_header,
        hop_sep,
        hop_row("AR teacher-force", ar, "tf"),
        hop_row("AR free-run", ar, "freerun"),
        hop_row("Transformer teacher-force", tfm, "tf"),
        hop_row("Transformer free-run", tfm, "freerun"),
        hop_row("AR wrong-1st (2nd) ", ar, "interv_2nd"),
        hop_row("Transformer wrong-1st (2nd)", tfm, "interv_2nd"),
        hop_row("AR wrong-1st (random)", ar, "interv_rand"),
        hop_row("Transformer wrong-1st (random)", tfm, "interv_rand"),
        "",
        f"图：`results/figures/transformer_vs_ar_hop_accuracy.png`。原始数字：`results/transformer_comparison.json`。",
        "",
        f"**这一对照说明什么。** {verdict_zh}",
        (
            f"具体差：Transformer free-run exact − AR free-run exact = {_fmt(tf_gain)}；"
            f"第二候选后的后续墙准确率 AR {_fmt(ar['interv_2nd']['later_wall_acc'])} / "
            f"Transformer {_fmt(tfm['interv_2nd']['later_wall_acc'])}；"
            f"随机第一跳后的合法率 AR {_fmt(ar['interv_rand']['valid'])} "
            f"（free-run {_fmt(ar['freerun']['valid'])}）/ "
            f"Transformer {_fmt(tfm['interv_rand']['valid'])} "
            f"（free-run {_fmt(tfm['freerun']['valid'])}）。"
        ),
        "误差是否沿跳累积，看上表 TF hop2 与 free hop2 的差，不另定义指标。",
        "",
        "### Transformer sequence baseline (same data, same first-hop protocol)",
        "",
        (
            f"Paired measurement, seed={payload['seed']}, n={int(n_eval)} test paths with "
            "n_bounces≥2. Splits match `experiments.run_all` (240/50/70 scenes plus one "
            "showcase scene per split, seed 0). Ground truth is still the image-method "
            "enumerator plus the repo's corner-diffraction rules. Neither model generates that ground truth. "
            "Epoch budgets differ on purpose: the ordinary AR keeps this repo's 16-epoch "
            "constant-lr recipe; the Transformer is trained until validation teacher-forced "
            "accuracy stops improving and the training loss flattens, or until the epoch cap."
        ),
        "",
        (
            f"Ordinary AR: best validation teacher-forced accuracy {_fmt(ar_tr.get('val_tf_acc'))} "
            f"after {ar_tr.get('epochs_ran')} epochs at the published small-model recipe "
            f"(d=64, 2 layers, constant AdamW lr=2e-3). "
            f"Transformer: {tf_tr.get('n_params')} parameters, best validation teacher-forced "
            f"accuracy {_fmt(tf_tr.get('val_tf_acc'))} at epoch {tf_tr.get('best_epoch')} "
            f"of {tf_tr.get('epochs_ran')} ({tf_tr.get('stop_reason')}; "
            f"train-loss plateau flag {tf_tr.get('train_loss_plateaued')}). "
            f"Train loss {_fmt(tf_tr.get('train_loss_first'), 4)} → {_fmt(tf_tr.get('train_loss_last'), 4)}."
        ),
        "",
        verdict_en,
        (
            f"Free-run exact: AR {_fmt(ar_exact)}, Transformer {_fmt(tf_exact)} "
            f"(delta {_fmt(tf_gain)}). "
            f"After the model's own second-best first hop, later-wall accuracy is "
            f"AR {_fmt(ar['interv_2nd']['later_wall_acc'])}, "
            f"Transformer {_fmt(tfm['interv_2nd']['later_wall_acc'])}; "
            f"exact match is AR {_fmt(ar['interv_2nd']['exact'])}, "
            f"Transformer {_fmt(tfm['interv_2nd']['exact'])}; "
            f"geometric validity stays AR {_fmt(ar['interv_2nd']['valid'])}, "
            f"Transformer {_fmt(tfm['interv_2nd']['valid'])}, "
            f"of which a scene-enumerated branch covers "
            f"AR {_fmt(ar['interv_2nd']['any_scene_path'])} and "
            f"Transformer {_fmt(tfm['interv_2nd']['any_scene_path'])}. "
            f"After a random existing wall/corner first hop, validity is "
            f"AR {_fmt(ar['interv_rand']['valid'])} and "
            f"Transformer {_fmt(tfm['interv_rand']['valid'])} "
            f"(exact {_fmt(ar['interv_rand']['exact'])} / {_fmt(tfm['interv_rand']['exact'])}). "
            f"First-hop top-2 contains the labeled token for "
            f"AR {_fmt(ar_s.get('first_token_top2'))} and "
            f"Transformer {_fmt(tf_s.get('first_token_top2'))} of paths."
        ),
        "Hop-wise teacher-forcing versus free-run is the table above and "
        "`results/figures/transformer_vs_ar_hop_accuracy.png`. "
        "No metric in this section was filled in by hand.",
        "",
        END,
        "",
    ]
    return "\n".join(lines)


def upsert_findings(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if BEGIN in existing and END in existing:
        pre, rest = existing.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        # section already includes the markers
        text = pre.rstrip() + "\n\n" + section.rstrip() + "\n" + post.lstrip("\n")
    else:
        text = existing.rstrip() + "\n\n" + section.rstrip() + "\n"
    path.write_text(text, encoding="utf-8")


def make_figure(payload: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar = payload["ar"]["hop_table"]
    tfm = payload["transformer"]["hop_table"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax = axes[0]
    xs = np.arange(1, 4)
    series = [
        ("AR teacher-force", ar["tf"]["hops"][:3]),
        ("AR free-run", ar["freerun"]["hops"][:3]),
        ("Transformer teacher-force", tfm["tf"]["hops"][:3]),
        ("Transformer free-run", tfm["freerun"]["hops"][:3]),
    ]
    for lab, ys in series:
        ax.plot(xs, ys, marker="o", label=lab)
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("hop after Tx")
    ax.set_ylabel("token accuracy vs this GT path")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)
    ax.set_title("Teacher-forcing vs free-run")

    ax = axes[1]
    tags = ["freerun", "interv_2nd", "interv_rand"]
    labels = ["free-run", "wrong 1st\n(2nd-best)", "wrong 1st\n(random)"]
    x = np.arange(len(tags))
    width = 0.18
    ax.bar(x - 1.5 * width, [ar[t]["exact"] for t in tags], width, label="AR exact")
    ax.bar(x - 0.5 * width, [ar[t]["valid"] for t in tags], width, label="AR valid")
    ax.bar(x + 0.5 * width, [tfm[t]["exact"] for t in tags], width, label="TFM exact")
    ax.bar(x + 1.5 * width, [tfm[t]["valid"] for t in tags], width, label="TFM valid")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Exact GT vs geometric validity")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_comparison(
    npz: str,
    jsonl: str,
    ar_ckpt: str,
    tfm_ckpt: str,
    out_dir: str,
    seed: int = 0,
    min_bounces: int = 2,
    findings_path: str | None = None,
) -> dict:
    seed_all(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_index = load_jsonl_index(jsonl)
    ds = PathNPZDataset(npz, split="test", min_bounces=min_bounces)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    ar_model, ar_blob = _load_ar(Path(ar_ckpt))
    tfm_model, tfm_blob = _load_transformer(Path(tfm_ckpt))
    # Independent generators so both models see the same illegal first hops.
    ar_scored = score_sequence_model(
        ar_model, loader, json_index, np.random.default_rng(seed + 7)
    )
    tfm_scored = score_sequence_model(
        tfm_model, loader, json_index, np.random.default_rng(seed + 7)
    )
    train_ds = PathNPZDataset(npz, split="train")
    val_ds = PathNPZDataset(npz, split="val")
    payload = {
        "seed": seed,
        "min_bounces": min_bounces,
        "n_eval": ar_scored["summary"]["n_eval"],
        "n_train_paths": len(train_ds),
        "n_val_paths": len(val_ds),
        "n_test_paths_all": len(PathNPZDataset(npz, split="test")),
        "protocol": (
            "score_sequence_model: teacher-forcing, greedy free-run, "
            "oracle first interaction, second-best first interaction, "
            "random existing wall/corner ≠ labeled first hop; "
            "rng = np.random.default_rng(seed+7) reset per model"
        ),
        "ar_ckpt": str(ar_ckpt),
        "transformer_ckpt": str(tfm_ckpt),
        "ar": {**ar_scored, "train": _train_brief_ar(ar_blob)},
        "transformer": {**tfm_scored, "train": _train_brief_tfm(tfm_blob)},
    }
    if payload["transformer"]["summary"]["n_eval"] != payload["ar"]["summary"]["n_eval"]:
        raise RuntimeError("AR and Transformer were not scored on the same number of paths")
    fig = out / "figures" / "transformer_vs_ar_hop_accuracy.png"
    make_figure(payload, fig)
    section = render_findings_section(payload)
    findings = Path(findings_path) if findings_path else out / "findings.md"
    upsert_findings(findings, section)
    # Drop per-epoch hist from the comparison file? Keep it: it is the plateau evidence.
    (out / "transformer_comparison.json").write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="data/generated/dataset.npz")
    p.add_argument("--jsonl", default="data/generated/dataset.jsonl")
    p.add_argument("--ar-ckpt", default="results/ckpts/ar_transformer.pt")
    p.add_argument("--ckpt", default="artifacts/transformer_seq/best.pt")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    payload = run_comparison(
        args.npz, args.jsonl, args.ar_ckpt, args.ckpt, args.out, seed=args.seed
    )
    print(
        "n_eval",
        payload["n_eval"],
        "AR free exact",
        round(payload["ar"]["hop_table"]["freerun"]["exact"], 3),
        "Transformer free exact",
        round(payload["transformer"]["hop_table"]["freerun"]["exact"], 3),
    )
    print("wrote", Path(args.out) / "transformer_comparison.json")


if __name__ == "__main__":
    main()
