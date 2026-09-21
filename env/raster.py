"""RadioUNet-style occupancy rasterization (context encoder input only)."""

from __future__ import annotations

import numpy as np

from .scene import Scene


def occupancy_grid(scene: Scene, grid: int = 32) -> np.ndarray:
    """Return float32 occupancy in [0, 1] of shape (grid, grid), y-down image order."""
    occ = np.zeros((grid, grid), dtype=np.float32)
    sx = grid / scene.width
    sy = grid / scene.height
    for rect in scene.rects:
        i0 = int(np.floor(rect.xmin * sx))
        i1 = int(np.ceil(rect.xmax * sx))
        j0 = int(np.floor((scene.height - rect.ymax) * sy))
        j1 = int(np.ceil((scene.height - rect.ymin) * sy))
        i0, i1 = max(0, i0), min(grid, i1)
        j0, j1 = max(0, j0), min(grid, j1)
        occ[j0:j1, i0:i1] = 1.0
    return occ


def _splat_gaussian(grid: int, p: np.ndarray, width: float, height: float, sigma: float) -> np.ndarray:
    ys = (np.arange(grid) + 0.5) / grid
    xs = (np.arange(grid) + 0.5) / grid
    gx, gy = np.meshgrid(xs, ys)
    px = float(p[0] / width)
    py = float(1.0 - p[1] / height)
    ch = np.exp(-((gx - px) ** 2 + (gy - py) ** 2) / (2.0 * sigma * sigma))
    return ch.astype(np.float32)


def scene_channels(scene: Scene, grid: int = 32, sigma: float = 0.05) -> np.ndarray:
    """3 x H x W: occupancy, Tx gaussian, Rx gaussian. RadioUNet-like context."""
    occ = occupancy_grid(scene, grid=grid)
    tx = _splat_gaussian(grid, scene.tx, scene.width, scene.height, sigma)
    rx = _splat_gaussian(grid, scene.rx, scene.width, scene.height, sigma)
    return np.stack([occ, tx, rx], axis=0)
