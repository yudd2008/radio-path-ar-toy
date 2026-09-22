"""Corner diffraction geometry tests."""

from __future__ import annotations

import numpy as np

from env.geometry import Rect
from env.scene import Scene, showcase_reflect_diffract_scene
from pathfind.diffraction import KIND_DIFFRACT, KIND_REFLECT, reconstruct_interactions
from pathfind.image_method import find_paths, reconstruct_path


def _scene(rects, tx, rx) -> Scene:
    rs = [Rect(i, *r) for i, r in enumerate(rects)]
    return Scene.from_rects(1.0, 1.0, rs, tx=tx, rx=rx)


def test_nlos_corner_diffraction():
    # Building; Tx left, Rx below in the shadow. Wrap bottom-left corner.
    scene = _scene([[0.40, 0.30, 0.85, 0.85]], [0.18, 0.62], [0.60, 0.12])
    assert reconstruct_path(scene, []) is None
    bl = next(c for c in scene.corners if c.name.endswith("_bl"))
    path = reconstruct_interactions(scene, [(KIND_DIFFRACT, bl.corner_id)])
    assert path is not None, "expected 1-corner diffraction in the shadow"
    assert path.mechanism() == "diffraction"
    assert path.tokens()[1].startswith("D_corner_")
    np.testing.assert_allclose(path.points[1], bl.xy, atol=1e-8)


def test_interior_cone_rejected():
    scene = _scene([[0.40, 0.30, 0.85, 0.85]], [0.18, 0.62], [0.60, 0.50])
    # Rx is to the right, through the building — even the far corner is invalid.
    tr = next(c for c in scene.corners if c.name.endswith("_tr"))
    assert reconstruct_interactions(scene, [(KIND_DIFFRACT, tr.corner_id)]) is None


def test_showcase_has_reflection_and_diffraction():
    scene = showcase_reflect_diffract_scene(nlos=True)
    paths = find_paths(scene, max_bounces=2, max_paths=12)
    mechs = {p.mechanism() for p in paths}
    assert "diffraction" in mechs, mechs
    assert "reflection" in mechs, mechs
    assert reconstruct_path(scene, []) is None  # LoS blocked in NLOS showcase
    assert any(p.mechanism() == "reflection" and p.n_bounces == 1 for p in paths)
    assert any(p.mechanism() == "diffraction" and p.n_bounces == 1 for p in paths)


def test_tokens_distinguish_kinds():
    scene = showcase_reflect_diffract_scene(nlos=True)
    paths = find_paths(scene, max_bounces=2, max_paths=12)
    dpaths = [p for p in paths if p.mechanism() == "diffraction"]
    rpaths = [p for p in paths if p.mechanism() == "reflection"]
    assert dpaths and rpaths
    assert any(t.startswith("D_corner_") for t in dpaths[0].tokens())
    assert any(t.startswith("R_wall_") for t in rpaths[0].tokens())
    assert "TX" in dpaths[0].tokens() and "RX" in dpaths[0].tokens()


def test_schema_roundtrip_kinds():
    from data.schema import decode_interactions, tokens_from_interactions

    scene = showcase_reflect_diffract_scene(nlos=True)
    paths = find_paths(scene, max_bounces=2, max_paths=12)
    for p in paths:
        tok = tokens_from_interactions(p.interactions)
        assert decode_interactions(tok) == p.interactions


def test_mixed_reconstruct_when_present():
    scene = showcase_reflect_diffract_scene(nlos=True)
    paths = find_paths(scene, max_bounces=2, max_paths=12)
    mixed = [p for p in paths if p.mechanism() == "mixed"]
    if not mixed:
        return
    toks = mixed[0].tokens()
    assert any(t.startswith("R_wall_") for t in toks)
    assert any(t.startswith("D_corner_") for t in toks)
    again = reconstruct_interactions(scene, mixed[0].interactions)
    assert again is not None
    np.testing.assert_allclose(again.points, mixed[0].points, atol=1e-8)


def test_validity_and_points_accepts_diffract_tokens():
    from data.schema import tokens_from_interactions
    from experiments.common import validity_and_points

    scene = showcase_reflect_diffract_scene(nlos=True)
    paths = find_paths(scene, max_bounces=2, max_paths=12)
    dpaths = [p for p in paths if p.mechanism() == "diffraction"]
    assert dpaths
    tok = tokens_from_interactions(dpaths[0].interactions)
    ok, pts, inter = validity_and_points(scene, tok)
    assert ok
    assert inter == dpaths[0].interactions
    np.testing.assert_allclose(pts, dpaths[0].points, atol=1e-8)


if __name__ == "__main__":
    test_nlos_corner_diffraction()
    test_interior_cone_rejected()
    test_showcase_has_reflection_and_diffraction()
    test_tokens_distinguish_kinds()
    test_schema_roundtrip_kinds()
    test_mixed_reconstruct_when_present()
    test_validity_and_points_accepts_diffract_tokens()
    print("ok")
