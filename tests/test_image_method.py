"""Geometry and image-method unit tests (no learned models)."""

from __future__ import annotations

import numpy as np

from env.geometry import Rect
from env.scene import Scene
from pathfind.image_method import find_paths, reconstruct_path, specular_angles_ok


def _scene(rects, tx, rx) -> Scene:
    rs = [Rect(i, *r) for i, r in enumerate(rects)]
    return Scene.from_rects(1.0, 1.0, rs, tx=tx, rx=rx)


def test_los_empty():
    scene = _scene([], [0.2, 0.2], [0.8, 0.8])
    paths = find_paths(scene, max_bounces=2)
    assert any(p.n_bounces == 0 for p in paths)
    assert reconstruct_path(scene, []) is not None


def test_blocked_los():
    scene = _scene([[0.4, 0.2, 0.6, 0.8]], [0.15, 0.5], [0.85, 0.5])
    assert reconstruct_path(scene, []) is None


def test_same_side_single_bounce():
    # Thin vertical obstacle; Tx and Rx on its left face.
    scene = _scene([[0.68, 0.2, 0.78, 0.8]], [0.2, 0.3], [0.2, 0.7])
    left = next(w for w in scene.walls if w.name.endswith("_left"))
    path = reconstruct_path(scene, [left.wall_id])
    assert path is not None
    assert path.n_bounces == 1
    p = path.points[1]
    assert abs(p[0] - left.x0) < 1e-8
    assert 0.2 < p[1] < 0.8
    assert specular_angles_ok(path.points[0], path.points[1], path.points[2], left)


def test_canyon_two_bounce():
    rects = [[0.15, 0.12, 0.35, 0.88], [0.65, 0.12, 0.85, 0.88]]
    scene = _scene(rects, [0.5, 0.22], [0.5, 0.78])
    right = next(w for w in scene.walls if w.name == "r0_right")
    left = next(w for w in scene.walls if w.name == "r1_left")
    path = reconstruct_path(scene, [right.wall_id, left.wall_id])
    assert path is not None, "expected ping-pong 2-bounce in the canyon"
    assert path.n_bounces == 2
    assert specular_angles_ok(path.points[0], path.points[1], path.points[2], right)
    assert specular_angles_ok(path.points[1], path.points[2], path.points[3], left)
    # Reconstruction is deterministic.
    again = reconstruct_path(scene, path.wall_ids)
    assert again is not None
    np.testing.assert_allclose(again.points, path.points, atol=1e-8)


def test_invalid_repeats_rejected():
    scene = _scene([[0.68, 0.2, 0.78, 0.8]], [0.2, 0.3], [0.2, 0.7])
    assert reconstruct_path(scene, [0, 0]) is None


if __name__ == "__main__":
    test_los_empty()
    test_blocked_los()
    test_same_side_single_bounce()
    test_canyon_two_bounce()
    test_invalid_repeats_rejected()
    print("ok")
