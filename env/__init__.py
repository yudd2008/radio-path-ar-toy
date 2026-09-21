"""2D axis-aligned urban toy environment."""

from .geometry import Rect, Wall, point_in_rect, segment_hits_rect_interior
from .scene import Scene, random_canyon_scene, random_scene

__all__ = [
    "Rect",
    "Wall",
    "Scene",
    "random_scene",
    "random_canyon_scene",
    "point_in_rect",
    "segment_hits_rect_interior",
]
