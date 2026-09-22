"""Light demo: plot specular reflection AND corner diffraction on one scene.

Does not train models. Geometry only.

    PYTHONPATH=. python3 -m experiments.demo_reflect_diffract --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.scene import showcase_reflect_diffract_scene
from env.viz import MECHANISM_LABEL, save_scene_paths
from pathfind.image_method import find_paths


def _token_str(path) -> str:
    return " → ".join(path.tokens())


def pick_legend_paths(paths, per_mech: int = 2) -> list[dict]:
    groups: dict[str, list] = {}
    for p in sorted(paths, key=lambda q: (q.n_interactions, q.length)):
        groups.setdefault(p.mechanism(), []).append(p)
    recs: list[dict] = []
    for mech in ("los", "reflection", "diffraction", "mixed"):
        chosen = groups.get(mech, [])[:per_mech]
        # Prefer showing a 1-bounce reflection if one exists.
        if mech == "reflection":
            one = [p for p in groups.get(mech, []) if p.n_interactions == 1]
            two = [p for p in groups.get(mech, []) if p.n_interactions >= 2]
            chosen = (one[:1] + two[:1]) or chosen
        for p in chosen:
            recs.append(
                {
                    "points": p.points,
                    "label": f"{MECHANISM_LABEL[mech]}  {_token_str(p)}",
                    "mechanism": mech,
                }
            )
    return recs


def save_showcase_figure(out_dir: str | Path, nlos: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = showcase_reflect_diffract_scene(nlos=nlos, scene_id=0)
    paths = find_paths(scene, max_bounces=2, max_paths=12)
    recs = pick_legend_paths(paths)
    title = (
        "NLOS urban toy: reflection (solid orange) + diffraction (dashed purple)"
        if nlos
        else "Partial-LoS urban toy: LoS / reflection / diffraction"
    )
    fig_path = out / "reflect_diffract_showcase.png"
    save_scene_paths(
        scene,
        recs,
        fig_path,
        title=title,
        wall_labels=True,
        corner_labels=True,
    )
    summary = {
        "figure": str(fig_path),
        "nlos": nlos,
        "n_paths": len(paths),
        "mechanisms": sorted({p.mechanism() for p in paths}),
        "paths": [
            {
                "mechanism": p.mechanism(),
                "tokens": p.tokens(),
                "length": float(p.length),
                "n_interactions": p.n_interactions,
            }
            for p in paths
        ],
    }
    (out / "reflect_diffract_showcase.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/figures")
    p.add_argument("--los", action="store_true", help="use the partial-LoS layout")
    args = p.parse_args()
    summary = save_showcase_figure(args.out, nlos=not args.los)
    print(json.dumps({k: summary[k] for k in ("figure", "nlos", "n_paths", "mechanisms")}, indent=2))
    for rec in summary["paths"]:
        print(f"  {rec['mechanism']:12s}  {' → '.join(rec['tokens'])}  L={rec['length']:.3f}")
    mechs = set(summary["mechanisms"])
    if "reflection" not in mechs or "diffraction" not in mechs:
        raise SystemExit("showcase did not yield both reflection and diffraction")


if __name__ == "__main__":
    main()
