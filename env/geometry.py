"""Axis-aligned rectangles, named walls, and robust 2D intersection tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class Wall:
    """One named edge of an obstacle. IDs are stable within a scene."""

    wall_id: int
    rect_id: int
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    nx: float
    ny: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x0, self.y0, self.x1, self.y1, self.nx, self.ny], dtype=np.float64
        )

    @property
    def is_vertical(self) -> bool:
        return abs(self.x1 - self.x0) < EPS

    @property
    def is_horizontal(self) -> bool:
        return abs(self.y1 - self.y0) < EPS

    @property
    def length(self) -> float:
        return float(np.hypot(self.x1 - self.x0, self.y1 - self.y0))

    @property
    def midpoint(self) -> np.ndarray:
        return np.array(
            [0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1)], dtype=np.float64
        )

    def point_from_t(self, t: float) -> np.ndarray:
        """Map t in [0, 1] along the segment from (x0,y0) to (x1,y1)."""
        return np.array(
            [
                self.x0 + t * (self.x1 - self.x0),
                self.y0 + t * (self.y1 - self.y0),
            ],
            dtype=np.float64,
        )

    def t_from_point(self, p: np.ndarray) -> float:
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        denom = dx * dx + dy * dy
        if denom < EPS:
            return 0.0
        return float(((p[0] - self.x0) * dx + (p[1] - self.y0) * dy) / denom)

    def reflect_point(self, p: np.ndarray) -> np.ndarray:
        """Mirror a point across the infinite line containing this wall."""
        if self.is_vertical:
            return np.array([2.0 * self.x0 - p[0], p[1]], dtype=np.float64)
        if self.is_horizontal:
            return np.array([p[0], 2.0 * self.y0 - p[1]], dtype=np.float64)
        raise ValueError(f"Wall {self.name} is not axis-aligned")

    def intersect_line(self, a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        """Intersect the infinite line AB with this finite wall segment.

        Returns the point if the intersection lies on the open segment,
        otherwise None.
        """
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dx, dy = bx - ax, by - ay

        if self.is_vertical:
            if abs(dx) < EPS:
                return None
            u = (self.x0 - ax) / dx
            y = ay + u * dy
            y_lo, y_hi = (self.y0, self.y1) if self.y0 <= self.y1 else (self.y1, self.y0)
            if y_lo + EPS < y < y_hi - EPS:
                return np.array([self.x0, y], dtype=np.float64)
            return None

        if self.is_horizontal:
            if abs(dy) < EPS:
                return None
            u = (self.y0 - ay) / dy
            x = ax + u * dx
            x_lo, x_hi = (self.x0, self.x1) if self.x0 <= self.x1 else (self.x1, self.x0)
            if x_lo + EPS < x < x_hi - EPS:
                return np.array([x, self.y0], dtype=np.float64)
            return None

        raise ValueError(f"Wall {self.name} is not axis-aligned")

    def contains_point(self, p: np.ndarray, tol: float = 1e-6) -> bool:
        t = self.t_from_point(p)
        if t < -tol or t > 1.0 + tol:
            return False
        proj = self.point_from_t(float(np.clip(t, 0.0, 1.0)))
        return float(np.linalg.norm(proj - np.asarray(p, dtype=np.float64))) <= tol


@dataclass(frozen=True)
class Rect:
    rect_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def inflate(self, m: float) -> "Rect":
        return Rect(
            self.rect_id,
            self.xmin - m,
            self.ymin - m,
            self.xmax + m,
            self.ymax + m,
        )

    def overlaps(self, other: "Rect") -> bool:
        return not (
            self.xmax <= other.xmin
            or other.xmax <= self.xmin
            or self.ymax <= other.ymin
            or other.ymax <= self.ymin
        )

    def walls(self, start_wall_id: int) -> list[Wall]:
        rid = self.rect_id
        return [
            Wall(
                start_wall_id,
                rid,
                f"r{rid}_left",
                self.xmin,
                self.ymin,
                self.xmin,
                self.ymax,
                -1.0,
                0.0,
            ),
            Wall(
                start_wall_id + 1,
                rid,
                f"r{rid}_right",
                self.xmax,
                self.ymin,
                self.xmax,
                self.ymax,
                1.0,
                0.0,
            ),
            Wall(
                start_wall_id + 2,
                rid,
                f"r{rid}_bottom",
                self.xmin,
                self.ymin,
                self.xmax,
                self.ymin,
                0.0,
                -1.0,
            ),
            Wall(
                start_wall_id + 3,
                rid,
                f"r{rid}_top",
                self.xmin,
                self.ymax,
                self.xmax,
                self.ymax,
                0.0,
                1.0,
            ),
        ]

    def corners(self, start_corner_id: int, start_wall_id: int) -> list["Corner"]:
        rid = self.rect_id
        left, right, bottom, top = (
            start_wall_id,
            start_wall_id + 1,
            start_wall_id + 2,
            start_wall_id + 3,
        )
        return [
            Corner(
                start_corner_id,
                rid,
                f"r{rid}_bl",
                self.xmin,
                self.ymin,
                -1.0,
                0.0,
                0.0,
                -1.0,
                left,
                bottom,
            ),
            Corner(
                start_corner_id + 1,
                rid,
                f"r{rid}_br",
                self.xmax,
                self.ymin,
                1.0,
                0.0,
                0.0,
                -1.0,
                right,
                bottom,
            ),
            Corner(
                start_corner_id + 2,
                rid,
                f"r{rid}_tl",
                self.xmin,
                self.ymax,
                -1.0,
                0.0,
                0.0,
                1.0,
                left,
                top,
            ),
            Corner(
                start_corner_id + 3,
                rid,
                f"r{rid}_tr",
                self.xmax,
                self.ymax,
                1.0,
                0.0,
                0.0,
                1.0,
                right,
                top,
            ),
        ]


@dataclass(frozen=True)
class Corner:
    """Convex 90° rectangle vertex (2D vertical wedge). Stable ID per scene."""

    corner_id: int
    rect_id: int
    name: str
    x: float
    y: float
    n1x: float
    n1y: float
    n2x: float
    n2y: float
    wall_id_a: int
    wall_id_b: int

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.n1x, self.n1y, self.n2x, self.n2y], dtype=np.float64
        )

    def n_faces_front(self, p: np.ndarray, eps: float = 1e-6) -> int:
        """How many of the two walls have p on the outward / free-space side."""
        v = np.asarray(p, dtype=np.float64) - self.xy
        n = 0
        if float(v[0] * self.n1x + v[1] * self.n1y) > eps:
            n += 1
        if float(v[0] * self.n2x + v[1] * self.n2y) > eps:
            n += 1
        return n


def point_in_rect(p: np.ndarray, rect: Rect, eps: float = 0.0) -> bool:
    return (
        rect.xmin - eps <= p[0] <= rect.xmax + eps
        and rect.ymin - eps <= p[1] <= rect.ymax + eps
    )


def segment_hits_rect_interior(
    a: np.ndarray, b: np.ndarray, rect: Rect, eps: float = 1e-8
) -> bool:
    """True iff the open segment AB clips the interior of an AABB.

    Touching a boundary at an endpoint (a bounce) is allowed. Grazing along
    a wall is treated as non-interior and therefore not a blocking hit.
    """
    xmin, ymin, xmax, ymax = rect.xmin, rect.ymin, rect.xmax, rect.ymax
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay

    pclip = [-dx, dx, -dy, dy]
    qclip = [ax - xmin, xmax - ax, ay - ymin, ymax - ay]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(pclip, qclip):
        if abs(pi) < EPS:
            if qi < -eps:
                return False
            continue
        r = qi / pi
        if pi < 0:
            if r > u0:
                u0 = r
        else:
            if r < u1:
                u1 = r
        if u0 > u1 + eps:
            return False

    if (u1 - u0) < 10 * eps:
        return False
    um = 0.5 * (u0 + u1)
    um = min(max(um, 0.0), 1.0)
    mx = ax + um * dx
    my = ay + um * dy
    return (xmin + eps < mx < xmax - eps) and (ymin + eps < my < ymax - eps)


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
