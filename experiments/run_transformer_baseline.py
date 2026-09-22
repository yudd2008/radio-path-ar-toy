"""One command: same seed-0 split, ordinary AR, serious Transformer, comparison.

Does not replace ``experiments.run_all``. That entrypoint stays the short demo
(small AR, one-shot, continuous heads). This entrypoint is the paired sequence
comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generate import generate_dataset
from experiments.common import seed_all
from experiments.eval_transformer import run_comparison
from experiments.train_matched_ar import run_training as train_weighted_ar
from experiments.train_sequence import run_training as train_ar
from experiments.train_transformer import run_training as train_transformer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data/generated")
    p.add_argument("--out", default="results")
    p.add_argument("--n-train", type=int, default=240)
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--n-test", type=int, default=70)
    p.add_argument("--ar-epochs", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-ar", action="store_true")
    p.add_argument("--skip-transformer", action="store_true")
    p.add_argument("--skip-weighted-ar", action="store_true")
    args = p.parse_args()

    seed_all(args.seed)
    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_data:
        counts = generate_dataset(
            data_dir,
            seed=args.seed,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
        )
        print("dataset:", json.dumps(counts))

    ckpt = out / "ckpts"
    if not args.skip_ar:
        ar_stats = train_ar(
            str(data_dir / "dataset.npz"),
            str(ckpt),
            seed=args.seed,
            epochs=args.ar_epochs,
            only=("ar_transformer",),
        )
        print("AR val_tf_acc", ar_stats["ar_transformer"]["val_tf_acc"])

    weighted_dir = Path("artifacts/ar_weighted")
    if not args.skip_weighted_ar:
        w_stats = train_weighted_ar(str(data_dir / "dataset.npz"), str(weighted_dir), seed=args.seed)
        print("weighted AR:", json.dumps(w_stats))

    tfm_dir = Path("artifacts/transformer_seq")
    if not args.skip_transformer:
        tfm_stats = train_transformer(
            str(data_dir / "dataset.npz"),
            str(tfm_dir),
            seed=args.seed,
            max_epochs=args.max_epochs,
            curve_path=str(out / "figures" / "transformer_train_curves.png"),
        )
        brief = {k: v for k, v in tfm_stats.items() if k != "hist"}
        print("transformer:", json.dumps(brief))

    payload = run_comparison(
        str(data_dir / "dataset.npz"),
        str(data_dir / "dataset.jsonl"),
        str(ckpt / "ar_transformer.pt"),
        str(tfm_dir / "best.pt"),
        str(out),
        seed=args.seed,
        ar_weighted_ckpt=str(weighted_dir / "best.pt"),
    )
    print("wrote", out / "transformer_comparison.json")
    print("n_eval", payload["n_eval"])


if __name__ == "__main__":
    main()
