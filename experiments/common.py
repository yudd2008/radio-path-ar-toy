"""Shared experiment helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from data.schema import decode_wall_ids
from env.geometry import polyline_length
from env.scene import Scene
from pathfind.image_method import reconstruct_path


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


def points_from_t(scene: Scene, wall_ids: list[int], t_on_wall) -> np.ndarray:
    pts = [scene.tx]
    for wid, t in zip(wall_ids, list(t_on_wall)):
        if wid < 0 or wid >= len(scene.walls):
            break
        pts.append(scene.walls[int(wid)].point_from_t(float(t)))
    pts.append(scene.rx)
    return np.stack(pts, axis=0)


def validity_and_points(scene: Scene, tokens) -> tuple[bool, np.ndarray | None, list[int]]:
    wids = decode_wall_ids(tokens)
    path = reconstruct_path(scene, wids)
    if path is None:
        return False, None, wids
    return True, path.points, wids


def hop_token_accuracy(pred: np.ndarray, gt: np.ndarray, max_hops: int = 4) -> list[float]:
    """Accuracy at sequence hops after TX (wall1, wall2, ..., RX)."""
    # pred/gt include TX at 0
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
