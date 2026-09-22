"""Shared experiment helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from data.schema import decode_interactions
from env.scene import Scene
from pathfind.diffraction import KIND_DIFFRACT, KIND_REFLECT, reconstruct_interactions


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cpu")


def load_jsonl_index(path: str | Path) -> dict[tuple[int, int], dict]:
    idx: dict[tuple[int, int], dict] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            idx[(int(rec["scene_id"]), int(rec["path_id"]))] = rec
    return idx


def scene_from_json(rec: dict) -> Scene:
    return Scene.from_dict(rec)


def interactions_from_rec(rec: dict) -> list[tuple[str, int]]:
    if rec.get("interactions"):
        return [(str(k), int(i)) for k, i in rec["interactions"]]
    return [(KIND_REFLECT, int(w)) for w in rec.get("wall_ids") or []]


def points_from_t(scene: Scene, wall_ids: list[int], t_on_wall) -> np.ndarray:
    """Legacy: treat every id as a reflecting wall (no diffraction)."""
    return points_from_interactions(
        scene, [(KIND_REFLECT, int(w)) for w in wall_ids], t_on_wall
    )


def points_from_interactions(scene: Scene, interactions, t_on_wall) -> np.ndarray:
    pts = [scene.tx]
    tlist = list(t_on_wall)
    for k, (kind, eid) in enumerate(interactions):
        t = float(tlist[k]) if k < len(tlist) else 0.5
        if kind == KIND_DIFFRACT and 0 <= eid < len(scene.corners):
            pts.append(scene.corners[int(eid)].xy)
        elif kind == KIND_REFLECT and 0 <= eid < len(scene.walls):
            pts.append(scene.walls[int(eid)].point_from_t(t))
        else:
            break
    pts.append(scene.rx)
    return np.stack(pts, axis=0)


def validity_and_points(
    scene: Scene, tokens
) -> tuple[bool, np.ndarray | None, list[tuple[str, int]]]:
    inter = decode_interactions(tokens)
    path = reconstruct_interactions(scene, inter)
    if path is None:
        return False, None, inter
    return True, path.points, inter


def hop_token_accuracy(pred: np.ndarray, gt: np.ndarray, max_hops: int = 4) -> list[float]:
    """Accuracy at sequence hops after TX (interaction1, ..., RX)."""
    acc = []
    for k in range(1, max_hops + 1):
        if k >= len(pred) or k >= len(gt):
            acc.append(float("nan"))
            continue
        acc.append(float(int(pred[k]) == int(gt[k])))
    return acc


def polyline_hop_rmse(pred_pts: np.ndarray | None, gt_pts: np.ndarray) -> list[float]:
    """RMSE of interaction points (excluding Tx) at each hop; nan if missing."""
    out = []
    n = len(gt_pts)
    for i in range(1, n):
        if pred_pts is None or i >= len(pred_pts):
            out.append(float("nan"))
        else:
            out.append(float(np.linalg.norm(pred_pts[i] - gt_pts[i])))
    return out


def batch_to_device(batch: dict, dev: torch.device) -> dict:
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
