"""Image-method specular reflections on named axis-aligned walls.

This is a 2D toy analogue of WinProp IRT path search:
  discrete interaction sequence (wall IDs) + continuous points on those walls.
We enumerate wall sequences up to a small bounce cap and accept a candidate
only if the unfolded image ray hits every finite wall segment, stays in free
space, and satisfies the reflection law (equal angles, outward-facing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from env.geometry import EPS, Wall, polyline_length
from env.scene import Scene

# Reject bounce points too close to corners (degenerate).
CORNER_MARGIN = 0.04
# Incoming/outgoing must leave a little margin vs the wall tangent.
NORMAL_DOT_MIN = 1e-6


@dataclass
class Path:
    wall_ids: list[int]
    points: np.ndarray  # (n_bounces+2, 2): Tx, interactions..., Rx
    t_on_wall: np.ndarray  # (n_bounces,) in (0,1)
    length: float
    valid: bool = True

    @property
    def n_bounces(self) -> int:
        return len(self.wall_ids)

    def tokens(self) -> list[str]:
        seq = ["TX"]
        seq.extend(f"wall_{i}" for i in self.wall_ids)
        seq.append("RX")
        return seq


def _outward_ok(wall: Wall, incoming_from: np.ndarray, bounce: np.ndarray) -> bool:
    incoming = np.asarray(incoming_from, dtype=np.float64) - np.asarray(bounce, dtype=np.float64)
    n = np.array([wall.nx, wall.ny], dtype=np.float64)
    return float(np.dot(incoming, n)) > NORMAL_DOT_MIN


def specular_angles_ok(
    p_prev: np.ndarray,
    p_hit: np.ndarray,
    p_next: np.ndarray,
    wall: Wall,
    atol: float = 2e-3,
) -> bool:
    """Angle of incidence equals angle of reflection in the plane."""
    u = np.asarray(p_prev, dtype=np.float64) - np.asarray(p_hit, dtype=np.float64)
    v = np.asarray(p_next, dtype=np.float64) - np.asarray(p_hit, dtype=np.float64)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < EPS or nv < EPS:
        return False
    u = u / nu
    v = v / nv
    n = np.array([wall.nx, wall.ny], dtype=np.float64)
    # u, v both point away from the bounce. Specular ⇒ equal normal
    # components on the free-space side and opposite tangents (mirror).
    u_n, v_n = float(np.dot(u, n)), float(np.dot(v, n))
    u_t = u - u_n * n
    v_t = v - v_n * n
    if u_n <= 0 or v_n <= 0:
        return False
    return float(np.linalg.norm(u_t + v_t)) < atol and abs(u_n - v_n) < atol


def _unfold_images(rx: np.ndarray, walls: list[Wall]) -> list[np.ndarray]:
    n = len(walls)
    images = [None] * (n + 1)
    images[n] = np.asarray(rx, dtype=np.float64).copy()
    for i in range(n - 1, -1, -1):
        images[i] = walls[i].reflect_point(images[i + 1])
    return images  # type: ignore[return-value]


def reconstruct_path(
    scene: Scene,
    wall_ids: Iterable[int],
    check_specular: bool = True,
) -> Optional[Path]:
    """Try to realize a discrete wall sequence as a valid specular polyline.

    Returns None if the sequence is geometrically invalid.
    """
    ids = [int(w) for w in wall_ids]
    if any(i < 0 or i >= len(scene.walls) for i in ids):
        return None
    if any(ids[k] == ids[k + 1] for k in range(len(ids) - 1)):
        return None

    walls = [scene.walls[i] for i in ids]
    tx, rx = scene.tx, scene.rx
    if len(walls) == 0:
        if scene.segment_occluded(tx, rx):
            return None
        pts = np.stack([tx, rx], axis=0)
        return Path([], pts, np.zeros((0,), dtype=np.float64), polyline_length(pts), True)

    images = _unfold_images(rx, walls)
    points = [tx]
    ts: list[float] = []
    current = tx
    for i, wall in enumerate(walls):
        hit = wall.intersect_line(current, images[i])
        if hit is None:
            return None
        t = wall.t_from_point(hit)
        if t < CORNER_MARGIN or t > 1.0 - CORNER_MARGIN:
            return None
        # The unfolded target must lie across the mirror: hit is strictly
        # between current and the image.
        vec = images[i] - current
        denom = float(np.dot(vec, vec))
        if denom < EPS:
            return None
        u = float(np.dot(hit - current, vec) / denom)
        if u <= EPS or u >= 1.0 - EPS:
            return None
        if not _outward_ok(wall, current, hit):
            return None
        if scene.segment_occluded(current, hit):
            return None
        points.append(hit)
        ts.append(t)
        current = hit

    if not _outward_ok(walls[-1], rx, points[-1]):
        return None
    if scene.segment_occluded(points[-1], rx):
        return None
    points.append(rx)
    pts = np.stack(points, axis=0)

    if check_specular:
        for i, wall in enumerate(walls):
            if not specular_angles_ok(pts[i], pts[i + 1], pts[i + 2], wall):
                return None

    return Path(ids, pts, np.array(ts, dtype=np.float64), polyline_length(pts), True)


def find_paths(
    scene: Scene,
    max_bounces: int = 3,
    max_paths: int = 12,
) -> list[Path]:
    """Enumerate LoS + specular 1..max_bounces paths, shortest first."""
    found: list[Path] = []
    n_walls = len(scene.walls)
    max_bounces = int(max(0, max_bounces))

    los = reconstruct_path(scene, [])
    if los is not None:
        found.append(los)

    # Breadth-by-bounce-count so 1-bounce and 2-bounce are complete even if
    # 3-bounce enumeration is large. Consecutive identical walls are skipped.
    prev_level: list[list[int]] = [[]]
    for _bounce in range(1, max_bounces + 1):
        nxt: list[list[int]] = []
        for prefix in prev_level:
            last = prefix[-1] if prefix else None
            for wid in range(n_walls):
                if wid == last:
                    continue
                seq = prefix + [wid]
                path = reconstruct_path(scene, seq)
                if path is not None:
                    found.append(path)
                nxt.append(seq)
        prev_level = nxt
    # Unique by wall sequence (reconstruction is deterministic).
    uniq: dict[tuple[int, ...], Path] = {}
    for p in found:
        key = tuple(p.wall_ids)
        prev = uniq.get(key)
        if prev is None or p.length < prev.length:
            uniq[key] = p
    ordered = sorted(uniq.values(), key=lambda p: (p.n_bounces, p.length))
    return ordered[:max_paths]
