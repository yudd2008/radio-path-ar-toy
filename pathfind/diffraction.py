"""Toy 2D corner diffraction (Keller/UTD-style path existence, not coefficients).

WinProp IRT also traces vertical wedges / edge diffraction. Here a rectangle
corner is a convex 90° wedge. We only decide whether a polyline through that
vertex is a geometrically allowed diffracted ray:

  * both legs stay in free space (no obstacle interior);
  * neither endpoint sits in the 90° occupied cone;
  * the vertex is a *silhouette* for at least one endpoint (exactly one of the
    two faces is front-facing) so the path actually wraps the wedge;
  * a small minimum bend, so grazing-along-a-face is rejected.

This is **not** full UTD: no diffraction coefficient, no Keller cone in 3D,
no creeping waves, no slope diffraction. Mixed reflection+diffraction chains
use the image method between diffraction vertices (or Tx/Rx).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from env.geometry import EPS, Corner
from env.scene import Scene

KIND_REFLECT = "R"
KIND_DIFFRACT = "D"

# Reject nearly-straight "diffraction" that is not wrapping.
MIN_BEND_COS = float(np.cos(np.deg2rad(8.0)))


def diffraction_ok(
    corner: Corner,
    incoming_from: np.ndarray,
    outgoing_to: np.ndarray,
    scene: Scene,
) -> bool:
    a = np.asarray(incoming_from, dtype=np.float64)
    b = np.asarray(outgoing_to, dtype=np.float64)
    c = corner.xy
    if corner.n_faces_front(a) == 0 or corner.n_faces_front(b) == 0:
        return False
    # At least one endpoint sees a silhouette (exactly one face).
    if corner.n_faces_front(a) != 1 and corner.n_faces_front(b) != 1:
        return False
    if scene.segment_occluded(a, c) or scene.segment_occluded(c, b):
        return False
    u = c - a
    v = b - c
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return False
    cosang = float(np.dot(u, v) / (nu * nv))
    if cosang > MIN_BEND_COS:
        return False
    return True


def reconstruct_specular_chain(
    scene: Scene,
    start: np.ndarray,
    end: np.ndarray,
    wall_ids: Sequence[int],
    check_specular: bool = True,
):
    """Image-method chain from `start` to `end` via reflecting walls.

    Returns (mid_points, t_list) not including start/end, or None.
    """
    from pathfind.image_method import (
        CORNER_MARGIN,
        _outward_ok,
        _unfold_images,
        specular_angles_ok,
    )

    ids = [int(w) for w in wall_ids]
    if any(i < 0 or i >= len(scene.walls) for i in ids):
        return None
    if any(ids[k] == ids[k + 1] for k in range(len(ids) - 1)):
        return None
    start = np.asarray(start, dtype=np.float64).reshape(2)
    end = np.asarray(end, dtype=np.float64).reshape(2)
    if not ids:
        if scene.segment_occluded(start, end):
            return None
        return [], []

    walls = [scene.walls[i] for i in ids]
    images = _unfold_images(end, walls)
    points: list[np.ndarray] = []
    ts: list[float] = []
    current = start
    for i, wall in enumerate(walls):
        hit = wall.intersect_line(current, images[i])
        if hit is None:
            return None
        t = wall.t_from_point(hit)
        if t < CORNER_MARGIN or t > 1.0 - CORNER_MARGIN:
            return None
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
    if not _outward_ok(walls[-1], end, points[-1]):
        return None
    if scene.segment_occluded(points[-1], end):
        return None
    if check_specular:
        seq_pts = [start, *points, end]
        for i, wall in enumerate(walls):
            if not specular_angles_ok(seq_pts[i], seq_pts[i + 1], seq_pts[i + 2], wall):
                return None
    return points, ts


def reconstruct_interactions(
    scene: Scene,
    interactions: Iterable[tuple[str, int]],
    check_specular: bool = True,
):
    """Realize a mixed reflection/diffraction sequence as a polyline."""
    from env.geometry import polyline_length
    from pathfind.image_method import Path

    inter = [(str(k), int(i)) for k, i in interactions]
    for kind, eid in inter:
        if kind == KIND_REFLECT and not (0 <= eid < len(scene.walls)):
            return None
        if kind == KIND_DIFFRACT and not (0 <= eid < len(scene.corners)):
            return None
        if kind not in (KIND_REFLECT, KIND_DIFFRACT):
            return None
    if any(inter[k] == inter[k + 1] for k in range(len(inter) - 1)):
        return None

    tx, rx = scene.tx, scene.rx
    if not inter:
        if scene.segment_occluded(tx, rx):
            return None
        pts = np.stack([tx, rx], axis=0)
        return Path([], pts, np.zeros((0,), dtype=np.float64), polyline_length(pts), True)

    points_acc = [tx.copy()]
    ts_acc: list[float] = []
    start = tx
    refl_buf: list[int] = []

    for kind, eid in inter:
        if kind == KIND_REFLECT:
            refl_buf.append(eid)
            continue
        corner = scene.corners[eid]
        chain = reconstruct_specular_chain(
            scene, start, corner.xy, refl_buf, check_specular=check_specular
        )
        if chain is None:
            return None
        mids, tlist = chain
        points_acc.extend(mids)
        ts_acc.extend(tlist)
        points_acc.append(corner.xy.copy())
        ts_acc.append(0.0)
        start = corner.xy
        refl_buf = []

    chain = reconstruct_specular_chain(
        scene, start, rx, refl_buf, check_specular=check_specular
    )
    if chain is None:
        return None
    mids, tlist = chain
    points_acc.extend(mids)
    ts_acc.extend(tlist)
    points_acc.append(rx.copy())
    pts = np.stack(points_acc, axis=0)

    if len(pts) != len(inter) + 2:
        return None
    for k, (kind, eid) in enumerate(inter):
        if kind != KIND_DIFFRACT:
            continue
        if not diffraction_ok(scene.corners[eid], pts[k], pts[k + 2], scene):
            return None

    return Path(
        inter,
        pts,
        np.array(ts_acc, dtype=np.float64),
        polyline_length(pts),
        True,
    )
