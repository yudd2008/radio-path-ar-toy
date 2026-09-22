"""Scene / path visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .scene import Scene

# Distinct styles for the four path mechanisms (legend: LoS / reflection /
# diffraction / mixed). Diffraction uses dashed + squares so it is readable
# even if a color print collapses.
MECHANISM_STYLE = {
    "los": dict(color="#2ca02c", ls="-", marker="o", lw=1.9),
    "reflection": dict(color="#ff7f0e", ls="-", marker="o", lw=1.9),
    "diffraction": dict(color="#9467bd", ls=(0, (6, 2.4)), marker="s", lw=2.1),
    "mixed": dict(color="#17becf", ls="-.", marker="D", lw=2.0),
}

MECHANISM_LABEL = {
    "los": "LoS",
    "reflection": "reflection",
    "diffraction": "diffraction",
    "mixed": "mixed",
}

_FALLBACK_PALETTE = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]


def draw_scene(ax, scene: Scene, wall_labels: bool = False, corner_labels: bool = False) -> None:
    ax.set_xlim(0, scene.width)
    ax.set_ylim(0, scene.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f4ee")
    for rect in scene.rects:
        ax.add_patch(
            Rectangle(
                (rect.xmin, rect.ymin),
                rect.xmax - rect.xmin,
                rect.ymax - rect.ymin,
                facecolor="#6d7a8a",
                edgecolor="#222",
                linewidth=1.0,
                alpha=0.92,
            )
        )
    if wall_labels:
        for w in scene.walls:
            m = w.midpoint
            ax.text(
                m[0],
                m[1],
                f"w{w.wall_id}",
                fontsize=7,
                ha="center",
                va="center",
                color="#111",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
            )
    if corner_labels:
        for c in scene.corners:
            ax.scatter(c.x, c.y, s=18, c="#111", marker="x", zorder=4)
            ax.text(
                c.x,
                c.y,
                f"c{c.corner_id}",
                fontsize=6,
                ha="left",
                va="bottom",
                color="#4a148c",
            )
    ax.scatter(*scene.tx, s=55, c="#d62728", zorder=5, label="Tx")
    ax.scatter(*scene.rx, s=55, c="#1f77b4", zorder=5, label="Rx")
    ax.annotate("Tx", scene.tx, textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.annotate("Rx", scene.rx, textcoords="offset points", xytext=(6, 6), fontsize=8)


def draw_path(
    ax,
    points: Sequence[np.ndarray] | np.ndarray,
    color: str = "#2ca02c",
    ls: str = "-",
    lw: float = 1.8,
    label: Optional[str] = None,
    marker: str = "o",
) -> None:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return
    ax.plot(pts[:, 0], pts[:, 1], color=color, ls=ls, lw=lw, marker=marker, ms=5, label=label)


def _style_for(rec: dict, index: int) -> dict:
    mech = rec.get("mechanism")
    if mech in MECHANISM_STYLE:
        return dict(MECHANISM_STYLE[mech])
    return dict(
        color=_FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)],
        ls="-",
        marker="o",
        lw=1.8,
    )


def save_scene_paths(
    scene: Scene,
    paths: Iterable[dict],
    out_path: str | Path,
    title: str = "",
    wall_labels: bool = False,
    corner_labels: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    draw_scene(ax, scene, wall_labels=wall_labels, corner_labels=corner_labels)
    for i, rec in enumerate(paths):
        if isinstance(rec, dict):
            pts = rec["points"]
            label = rec.get("label")
            style = _style_for(rec, i)
        else:
            pts = rec
            label = None
            style = _style_for({}, i)
        draw_path(ax, pts, label=label, **style)
    if title:
        ax.set_title(title, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best", fontsize=7, framealpha=0.9, borderpad=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
