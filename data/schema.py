"""Discrete tokens + continuous interaction-point schema.

Token layout (per scene, wall IDs are stable named edges):
    PAD=0, TX=1, RX=2, wall_0=3, wall_1=4, ...
A path is TX → wall_i → wall_j → … → RX.
Continuous geometry is the bounce point on each wall, stored both as xy
and as a 1-D parameter t ∈ (0,1) along that named edge.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from env.scene import Scene
from pathfind.image_method import Path

PAD_ID = 0
TX_ID = 1
RX_ID = 2
VOCAB_WALL_OFFSET = 3  # wall_k token = VOCAB_WALL_OFFSET + k

MAX_WALLS = 20
MAX_BOUNCES = 3
MAX_SEQ_LEN = MAX_BOUNCES + 2  # TX + bounces + RX
WALL_FEAT_DIM = 6  # x0,y0,x1,y1,nx,ny
GRID = 32


def wall_token_id(wall_id: int) -> int:
    return VOCAB_WALL_OFFSET + int(wall_id)


def tokens_from_wall_ids(wall_ids: list[int], max_seq: int = MAX_SEQ_LEN) -> np.ndarray:
    seq = [TX_ID] + [wall_token_id(w) for w in wall_ids] + [RX_ID]
    out = np.full((max_seq,), PAD_ID, dtype=np.int64)
    n = min(len(seq), max_seq)
    out[:n] = np.array(seq[:n], dtype=np.int64)
    return out


def decode_wall_ids(tokens: np.ndarray | list[int]) -> list[int]:
    ids: list[int] = []
    for t in list(tokens):
        t = int(t)
        if t == PAD_ID or t == TX_ID:
            continue
        if t == RX_ID:
            break
        if t >= VOCAB_WALL_OFFSET:
            ids.append(t - VOCAB_WALL_OFFSET)
    return ids


def wall_feature_table(scene: Scene, max_walls: int = MAX_WALLS) -> tuple[np.ndarray, np.ndarray]:
    feats = np.zeros((max_walls, WALL_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((max_walls,), dtype=np.float32)
    for w in scene.walls[:max_walls]:
        feats[w.wall_id] = w.as_array().astype(np.float32)
        mask[w.wall_id] = 1.0
    return feats, mask


def pack_sample(
    scene: Scene,
    path: Path,
    channels: np.ndarray,
    split: str,
    path_id: int,
    is_shortest: bool,
) -> dict[str, Any]:
    tokens = tokens_from_wall_ids(path.wall_ids)
    t_pad = np.zeros((MAX_BOUNCES,), dtype=np.float32)
    if path.n_bounces:
        t_pad[: path.n_bounces] = path.t_on_wall.astype(np.float32)
    pts_pad = np.zeros((MAX_BOUNCES + 2, 2), dtype=np.float32)
    npts = min(len(path.points), MAX_BOUNCES + 2)
    pts_pad[:npts] = path.points[:npts].astype(np.float32)
    wall_ids_pad = np.full((MAX_BOUNCES,), -1, dtype=np.int64)
    if path.n_bounces:
        wall_ids_pad[: path.n_bounces] = np.array(path.wall_ids, dtype=np.int64)
    feats, wmask = wall_feature_table(scene)
    json_rec = {
        "scene_id": scene.scene_id,
        "path_id": path_id,
        "split": split,
        "world": [float(scene.width), float(scene.height)],
        "rects": [[r.xmin, r.ymin, r.xmax, r.ymax] for r in scene.rects],
        "tx": [float(scene.tx[0]), float(scene.tx[1])],
        "rx": [float(scene.rx[0]), float(scene.rx[1])],
        "tokens": path.tokens(),
        "wall_ids": path.wall_ids,
        "points": path.points.tolist(),
        "t_on_wall": path.t_on_wall.tolist(),
        "n_bounces": path.n_bounces,
        "length": path.length,
        "valid": True,
        "is_shortest": is_shortest,
        "n_walls": len(scene.walls),
    }
    arrays = {
        "tokens": tokens,
        "wall_ids": wall_ids_pad,
        "t_on_wall": t_pad,
        "points": pts_pad,
        "n_bounces": np.int64(path.n_bounces),
        "tx": scene.tx.astype(np.float32),
        "rx": scene.rx.astype(np.float32),
        "channels": channels.astype(np.float32),
        "wall_feats": feats,
        "wall_mask": wmask,
        "is_shortest": np.int64(1 if is_shortest else 0),
        "length": np.float32(path.length),
        "scene_id": np.int64(scene.scene_id),
        "path_id": np.int64(path_id),
        "n_walls": np.int64(len(scene.walls)),
    }
    return {"json": json_rec, "arrays": arrays}
