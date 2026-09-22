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

KIND_REFLECT = "R"
KIND_DIFFRACT = "D"


@dataclass
class Path:
    """A geometrically valid polyline: Tx, interactions, Rx.

    `interactions` is a list of (kind, element_id) with kind in {R, D}:
      R = specular reflection on wall_id, D = diffraction at corner_id.
    """

    interactions: list[tuple[str, int]]
    points: np.ndarray  # (n_inter+2, 2): Tx, interactions..., Rx
    t_on_wall: np.ndarray  # (n_inter,) in (0,1) for R, 0 for D
    length: float
    valid: bool = True

    @property
    def n_bounces(self) -> int:
        return len(self.interactions)

    @property
    def n_interactions(self) -> int:
        return len(self.interactions)

    @property
    def wall_ids(self) -> list[int]:
        """Element ids in order (wall id if R, corner id if D)."""
        return [eid for _, eid in self.interactions]

    @property
    def kinds(self) -> list[str]:
        return [k for k, _ in self.interactions]

    def mechanism(self) -> str:
        ks = self.kinds
        if not ks:
            return "los"
        if all(k == KIND_REFLECT for k in ks):
            return "reflection"
        if all(k == KIND_DIFFRACT for k in ks):
            return "diffraction"
        return "mixed"

    def tokens(self) -> list[str]:
        seq = ["TX"]
        for kind, eid in self.interactions:
            if kind == KIND_DIFFRACT:
                seq.append(f"D_corner_{eid}")
            else:
                seq.append(f"R_wall_{eid}")
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
    """Reflection-only helper (all interactions are specular walls)."""
    from pathfind.diffraction import KIND_REFLECT, reconstruct_interactions

    inter = [(KIND_REFLECT, int(w)) for w in wall_ids]
    return reconstruct_interactions(scene, inter, check_specular=check_specular)


def _labeled_elements(scene: Scene) -> list[tuple[str, int]]:
    els = [(KIND_REFLECT, w.wall_id) for w in scene.walls]
    els.extend((KIND_DIFFRACT, c.corner_id) for c in scene.corners)
    return els


def _pick_diverse(found: list[Path], max_paths: int) -> list[Path]:
    groups = {"los": [], "reflection": [], "diffraction": [], "mixed": []}
    for p in sorted(found, key=lambda q: (q.n_interactions, q.length)):
        groups[p.mechanism()].append(p)
    out: list[Path] = []
    order = ["los", "reflection", "diffraction", "mixed"]
    while len(out) < max_paths and any(groups[k] for k in order):
        for k in order:
            if groups[k] and len(out) < max_paths:
                out.append(groups[k].pop(0))
    return out


def find_paths(
    scene: Scene,
    max_bounces: int = 3,
    max_paths: int = 12,
) -> list[Path]:
    """Enumerate LoS + specular + corner-diffracted paths, diverse then shortest."""
    from pathfind.diffraction import reconstruct_interactions

    found: list[Path] = []
    max_bounces = int(max(0, max_bounces))
    elements = _labeled_elements(scene)

    los = reconstruct_interactions(scene, [])
    if los is not None:
        found.append(los)

    prev_level: list[list[tuple[str, int]]] = [[]]
    for _hop in range(1, max_bounces + 1):
        nxt: list[list[tuple[str, int]]] = []
        for prefix in prev_level:
            last = prefix[-1] if prefix else None
            for lab in elements:
                if lab == last:
                    continue
                seq = prefix + [lab]
                path = reconstruct_interactions(scene, seq)
                if path is not None:
                    found.append(path)
                nxt.append(seq)
        prev_level = nxt

    uniq: dict[tuple[tuple[str, int], ...], Path] = {}
    for p in found:
        key = tuple(p.interactions)
        prev = uniq.get(key)
        if prev is None or p.length < prev.length:
            uniq[key] = p
    return _pick_diverse(list(uniq.values()), max_paths)
