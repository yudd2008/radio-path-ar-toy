"""Scene container: obstacles, stable wall IDs, Tx/Rx in free space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geometry import EPS, Rect, Wall, point_in_rect, segment_hits_rect_interior


@dataclass
class Scene:
    width: float
    height: float
    rects: list[Rect]
    walls: list[Wall]
    tx: np.ndarray
    rx: np.ndarray
    scene_id: int = 0
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_rects(
        cls,
        width: float,
        height: float,
        rects: list[Rect],
        tx: np.ndarray,
        rx: np.ndarray,
        scene_id: int = 0,
        meta: Optional[dict] = None,
    ) -> "Scene":
        walls: list[Wall] = []
        for i, rect in enumerate(rects):
            walls.extend(rect.walls(start_wall_id=4 * i))
        return cls(
            width=width,
            height=height,
            rects=list(rects),
            walls=walls,
            tx=np.asarray(tx, dtype=np.float64).reshape(2),
            rx=np.asarray(rx, dtype=np.float64).reshape(2),
            scene_id=scene_id,
            meta=meta or {},
        )

    def wall_by_id(self, wall_id: int) -> Wall:
        return self.walls[wall_id]

    def point_free(self, p: np.ndarray, clearance: float = 0.0) -> bool:
        if not (0.0 <= p[0] <= self.width and 0.0 <= p[1] <= self.height):
            return False
        for rect in self.rects:
            if point_in_rect(p, rect.inflate(clearance), eps=0.0):
                return False
        return True

    def segment_occluded(self, a: np.ndarray, b: np.ndarray) -> bool:
        for rect in self.rects:
            if segment_hits_rect_interior(a, b, rect):
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "world": [float(self.width), float(self.height)],
            "rects": [
                [r.xmin, r.ymin, r.xmax, r.ymax] for r in self.rects
            ],
            "walls": [
                {
                    "id": w.wall_id,
                    "rect_id": w.rect_id,
                    "name": w.name,
                    "x0": w.x0,
                    "y0": w.y0,
                    "x1": w.x1,
                    "y1": w.y1,
                    "nx": w.nx,
                    "ny": w.ny,
                }
                for w in self.walls
            ],
            "tx": [float(self.tx[0]), float(self.tx[1])],
            "rx": [float(self.rx[0]), float(self.rx[1])],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        rects = [
            Rect(i, float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            for i, r in enumerate(d["rects"])
        ]
        world = d["world"]
        return cls.from_rects(
            width=float(world[0]),
            height=float(world[1]),
            rects=rects,
            tx=np.array(d["tx"], dtype=np.float64),
            rx=np.array(d["rx"], dtype=np.float64),
            scene_id=int(d.get("scene_id", 0)),
            meta=dict(d.get("meta") or {}),
        )


def _sample_rect(rng: np.random.Generator, width: float, height: float,
                 min_size: float, max_size: float, margin: float) -> Rect:
    w = float(rng.uniform(min_size, max_size))
    h = float(rng.uniform(min_size, max_size))
    xmin = float(rng.uniform(margin, width - margin - w))
    ymin = float(rng.uniform(margin, height - margin - h))
    return Rect(-1, xmin, ymin, xmin + w, ymin + h)


def _place_rects(
    rng: np.random.Generator,
    n_rects: int,
    width: float,
    height: float,
    min_size: float,
    max_size: float,
    gap: float,
    margin: float,
    max_tries: int = 400,
) -> Optional[list[Rect]]:
    rects: list[Rect] = []
    for _ in range(n_rects):
        placed = False
        for _try in range(max_tries):
            cand = _sample_rect(rng, width, height, min_size, max_size, margin)
            grown = cand.inflate(gap)
            if any(grown.overlaps(r.inflate(gap)) for r in rects):
                continue
            rects.append(
                Rect(len(rects), cand.xmin, cand.ymin, cand.xmax, cand.ymax)
            )
            placed = True
            break
        if not placed:
            return None
    return rects


def _sample_free_point(
    rng: np.random.Generator,
    scene_probe: Scene,
    clearance: float,
    max_tries: int = 200,
) -> Optional[np.ndarray]:
    for _ in range(max_tries):
        p = np.array(
            [
                rng.uniform(clearance, scene_probe.width - clearance),
                rng.uniform(clearance, scene_probe.height - clearance),
            ],
            dtype=np.float64,
        )
        if scene_probe.point_free(p, clearance=clearance):
            return p
    return None


def random_canyon_scene(
    rng: np.random.Generator,
    width: float = 1.0,
    height: float = 1.0,
    scene_id: int = 0,
) -> Optional[Scene]:
    """Two parallel buildings + optional third block: a 2D street-canyon.

    This layout reliably yields 2-bounce ping-pong specular paths, which the
    fully random AABB soup often lacks.
    """
    gap = float(rng.uniform(0.18, 0.32))
    left_w = float(rng.uniform(0.16, 0.28))
    right_w = float(rng.uniform(0.16, 0.28))
    mid = 0.5 * width
    left_xmax = mid - 0.5 * gap
    right_xmin = mid + 0.5 * gap
    left_xmin = max(0.04, left_xmax - left_w)
    right_xmax = min(width - 0.04, right_xmin + right_w)
    y0 = float(rng.uniform(0.08, 0.18))
    y1 = float(rng.uniform(0.82, 0.92))
    rects = [
        Rect(0, left_xmin, y0, left_xmax, y1),
        Rect(1, right_xmin, y0, right_xmax, y1),
    ]
    if rng.random() < 0.55:
        # A third block at one end of the canyon (forces some NLOS).
        bw = float(rng.uniform(0.10, 0.18))
        bh = float(rng.uniform(0.10, 0.16))
        lo = left_xmax + 0.03
        hi = right_xmin - bw - 0.03
        if hi > lo + 0.01:
            bx = float(rng.uniform(lo, hi))
            by = 0.04 if rng.random() < 0.5 else height - 0.04 - bh
            rects.append(Rect(2, bx, by, bx + bw, by + bh))
    dummy = Scene.from_rects(width, height, rects, tx=[0, 0], rx=[0, 0])
    # Tx / Rx inside the canyon free space.
    x_lo, x_hi = left_xmax + 0.03, right_xmin - 0.03
    if x_hi <= x_lo + 0.04:
        return None
    tx = np.array([rng.uniform(x_lo, x_hi), rng.uniform(0.12, 0.40)], dtype=np.float64)
    rx = np.array([rng.uniform(x_lo, x_hi), rng.uniform(0.60, 0.88)], dtype=np.float64)
    if not dummy.point_free(tx, clearance=0.02) or not dummy.point_free(rx, clearance=0.02):
        return None
    return Scene.from_rects(width, height, rects, tx=tx, rx=rx, scene_id=scene_id)


def random_scene(
    rng: np.random.Generator,
    n_rects: int = 3,
    width: float = 1.0,
    height: float = 1.0,
    min_size: float = 0.10,
    max_size: float = 0.28,
    gap: float = 0.04,
    margin: float = 0.04,
    clearance: float = 0.03,
    min_tx_rx: float = 0.18,
    scene_id: int = 0,
) -> Optional[Scene]:
    """Sample non-overlapping AABBs and Tx/Rx in free space.

    Returns None if placement fails (caller should retry).
    """
    n_rects = int(np.clip(n_rects, 1, 6))
    rects = _place_rects(
        rng, n_rects, width, height, min_size, max_size, gap, margin
    )
    if rects is None:
        return None
    dummy = Scene.from_rects(width, height, rects, tx=[0, 0], rx=[0, 0])
    tx = _sample_free_point(rng, dummy, clearance)
    rx = _sample_free_point(rng, dummy, clearance)
    if tx is None or rx is None:
        return None
    if float(np.linalg.norm(tx - rx)) < min_tx_rx:
        return None
    # Reject degenerate almost-touching terminals.
    if abs(tx[0] - rx[0]) < EPS and abs(tx[1] - rx[1]) < EPS:
        return None
    return Scene.from_rects(
        width, height, rects, tx=tx, rx=rx, scene_id=scene_id
    )
