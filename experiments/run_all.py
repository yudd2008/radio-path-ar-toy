"""End-to-end demo: generate data → train tiny models → AR intervention experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generate import generate_dataset
from env.scene import Scene, random_scene
from env.viz import save_scene_paths
from experiments.ar_intervention import run_experiment
from experiments.common import seed_all
from experiments.train_continuous import run_training as train_cont
from experiments.train_sequence import run_training as train_seq
from pathfind.image_method import find_paths


def demo_gt_figure(seed: int, out: Path) -> None:
    rng = __import__("numpy").random.default_rng(seed)
    for k in range(40):
        scene = random_scene(rng, n_rects=3, scene_id=k)
        if scene is None:
            continue
        paths = find_paths(scene, max_bounces=3, max_paths=6)
        bounce = [p for p in paths if p.n_bounces >= 1]
        if not bounce:
            continue
        recs = [
            {"points": p.points, "label": f"{p.n_bounces}-bounce {p.tokens}"}
            for p in paths[:4]
        ]
        save_scene_paths(
            scene,
            recs,
            out / "gt_paths_example.png",
            title="Exact image-method paths (GT)",
            wall_labels=True,
        )
        return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data/generated")
    p.add_argument("--out", default="results")
    p.add_argument("--n-train", type=int, default=240)
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--n-test", type=int, default=70)
    p.add_argument("--seq-epochs", type=int, default=16)
    p.add_argument("--cont-epochs", type=int, default=14)
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    seed_all(args.seed)
    data_dir = Path(args.data_dir)
    out = Path(args.out)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    if not args.skip_data:
        counts = generate_dataset(
            data_dir,
            seed=args.seed,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
        )
        print("dataset:", json.dumps(counts, indent=2))
    demo_gt_figure(args.seed, fig)

    ckpt = out / "ckpts"
    if not args.skip_train:
        seq_stats = train_seq(str(data_dir / "dataset.npz"), str(ckpt), seed=args.seed, epochs=args.seq_epochs)
        cont_stats = train_cont(str(data_dir / "dataset.npz"), str(ckpt), seed=args.seed, epochs=args.cont_epochs)
        print("seq val:", {k: v.get("val_tf_acc") for k, v in seq_stats.items()})
        print("cont val:", {k: v.get("val_rmse") for k, v in cont_stats.items()})

    payload = run_experiment(
        str(data_dir / "dataset.npz"),
        str(data_dir / "dataset.jsonl"),
        str(ckpt),
        str(out),
        seed=args.seed,
    )
    print(payload["findings"])
    print("wrote", out / "metrics.json")


if __name__ == "__main__":
    main()
