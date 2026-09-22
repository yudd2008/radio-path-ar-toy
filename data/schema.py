"""Discrete tokens + continuous interaction-point schema.

Token layout (per scene; wall/corner IDs are stable named elements):
    PAD=0, TX=1, RX=2,
    R_wall_k  = VOCAB_REFLECT_OFFSET + k,
    D_corner_c = VOCAB_DIFFRACT_OFFSET + c
A path is TX → (R_wall_i | D_corner_j) → … → RX.
Reflection points use t ∈ (0,1) along the named edge; diffraction points
are the corner coordinates (t stored as 0).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from env.scene import Scene
from pathfind.diffraction import KIND_DIFFRACT, KIND_REFLECT
from pathfind.image_method import Path

PAD_ID = 0
TX_ID = 1
RX_ID = 2
VOCAB_REFLECT_OFFSET = 3
VOCAB_WALL_OFFSET = VOCAB_REFLECT_OFFSET  # alias used by the AR models
MAX_WALLS = 20
MAX_CORNERS = 20
VOCAB_DIFFRACT_OFFSET = VOCAB_REFLECT_OFFSET + MAX_WALLS  # 23
VOCAB_SIZE = VOCAB_DIFFRACT_OFFSET + MAX_CORNERS  # 43
MAX_BOUNCES = 3
MAX_SEQ_LEN = MAX_BOUNCES + 2  # TX + interactions + RX
WALL_FEAT_DIM = 6  # x0,y0,x1,y1,nx,ny
CORNER_FEAT_DIM = 6  # x,y,n1x,n1y,n2x,n2y
GRID = 32
KIND_R_INT = 0
KIND_D_INT = 1


def reflect_token_id(wall_id: int) -> int:
    return VOCAB_REFLECT_OFFSET + int(wall_id)


def diffract_token_id(corner_id: int) -> int:
    return VOCAB_DIFFRACT_OFFSET + int(corner_id)


def wall_token_id(wall_id: int) -> int:
    """Backward-compatible alias: reflection on wall_id."""
    return reflect_token_id(wall_id)


def tokens_from_interactions(
    interactions: list[tuple[str, int]], max_seq: int = MAX_SEQ_LEN
) -> np.ndarray:
    seq = [TX_ID]
    for kind, eid in interactions:
        if kind == KIND_DIFFRACT:
            seq.append(diffract_token_id(eid))
        else:
            seq.append(reflect_token_id(eid))
    seq.append(RX_ID)
    out = np.full((max_seq,), PAD_ID, dtype=np.int64)
    n = min(len(seq), max_seq)
    out[:n] = np.array(seq[:n], dtype=np.int64)
    return out


def tokens_from_wall_ids(wall_ids: list[int], max_seq: int = MAX_SEQ_LEN) -> np.ndarray:
    return tokens_from_interactions([(KIND_REFLECT, int(w)) for w in wall_ids], max_seq)


def decode_interactions(tokens: np.ndarray | list[int]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for t in list(tokens):
        t = int(t)
        if t == PAD_ID or t == TX_ID:
            continue
        if t == RX_ID:
            break
        if VOCAB_DIFFRACT_OFFSET <= t < VOCAB_DIFFRACT_OFFSET + MAX_CORNERS:
            out.append((KIND_DIFFRACT, t - VOCAB_DIFFRACT_OFFSET))
        elif VOCAB_REFLECT_OFFSET <= t < VOCAB_DIFFRACT_OFFSET:
            out.append((KIND_REFLECT, t - VOCAB_REFLECT_OFFSET))
    return out


def decode_wall_ids(tokens: np.ndarray | list[int]) -> list[int]:
    """Element ids only (legacy name). Prefer decode_interactions."""
    return [eid for _, eid in decode_interactions(tokens)]


def wall_feature_table(scene: Scene, max_walls: int = MAX_WALLS) -> tuple[np.ndarray, np.ndarray]:
    feats = np.zeros((max_walls, WALL_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((max_walls,), dtype=np.float32)
    for w in scene.walls[:max_walls]:
        feats[w.wall_id] = w.as_array().astype(np.float32)
        mask[w.wall_id] = 1.0
    return feats, mask


def corner_feature_table(
    scene: Scene, max_corners: int = MAX_CORNERS
) -> tuple[np.ndarray, np.ndarray]:
    feats = np.zeros((max_corners, CORNER_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((max_corners,), dtype=np.float32)
    for c in scene.corners[:max_corners]:
        feats[c.corner_id] = c.as_array().astype(np.float32)
        mask[c.corner_id] = 1.0
    return feats, mask


def pack_sample(
    scene: Scene,
    path: Path,
    channels: np.ndarray,
    split: str,
    path_id: int,
    is_shortest: bool,
) -> dict[str, Any]:
    tokens = tokens_from_interactions(path.interactions)
    t_pad = np.zeros((MAX_BOUNCES,), dtype=np.float32)
    if path.n_interactions:
        t_pad[: path.n_interactions] = path.t_on_wall.astype(np.float32)
    pts_pad = np.zeros((MAX_BOUNCES + 2, 2), dtype=np.float32)
    npts = min(len(path.points), MAX_BOUNCES + 2)
    pts_pad[:npts] = path.points[:npts].astype(np.float32)
    wall_ids_pad = np.full((MAX_BOUNCES,), -1, dtype=np.int64)
    kinds_pad = np.full((MAX_BOUNCES,), -1, dtype=np.int64)
    if path.n_interactions:
        wall_ids_pad[: path.n_interactions] = np.array(path.wall_ids, dtype=np.int64)
        kinds_pad[: path.n_interactions] = np.array(
            [KIND_D_INT if k == KIND_DIFFRACT else KIND_R_INT for k in path.kinds],
            dtype=np.int64,
        )
    feats, wmask = wall_feature_table(scene)
    cfeats, cmask = corner_feature_table(scene)
    json_rec = {
        "scene_id": scene.scene_id,
        "path_id": path_id,
        "split": split,
        "world": [float(scene.width), float(scene.height)],
        "rects": [[r.xmin, r.ymin, r.xmax, r.ymax] for r in scene.rects],
        "tx": [float(scene.tx[0]), float(scene.tx[1])],
        "rx": [float(scene.rx[0]), float(scene.rx[1])],
        "tokens": path.tokens(),
        "interactions": [[k, int(i)] for k, i in path.interactions],
        "wall_ids": path.wall_ids,
        "kinds": path.kinds,
        "mechanism": path.mechanism(),
        "points": path.points.tolist(),
        "t_on_wall": path.t_on_wall.tolist(),
        "n_bounces": path.n_bounces,
        "length": path.length,
        "valid": True,
        "is_shortest": is_shortest,
        "n_walls": len(scene.walls),
        "n_corners": len(scene.corners),
    }
    arrays = {
        "tokens": tokens,
        "wall_ids": wall_ids_pad,
        "kinds": kinds_pad,
        "t_on_wall": t_pad,
        "points": pts_pad,
        "n_bounces": np.int64(path.n_bounces),
        "tx": scene.tx.astype(np.float32),
        "rx": scene.rx.astype(np.float32),
        "channels": channels.astype(np.float32),
        "wall_feats": feats,
        "wall_mask": wmask,
        "corner_feats": cfeats,
        "corner_mask": cmask,
        "is_shortest": np.int64(1 if is_shortest else 0),
        "length": np.float32(path.length),
        "scene_id": np.int64(scene.scene_id),
        "path_id": np.int64(path_id),
        "n_walls": np.int64(len(scene.walls)),
        "n_corners": np.int64(len(scene.corners)),
    }
    return {"json": json_rec, "arrays": arrays}
