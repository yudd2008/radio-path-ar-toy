"""Random 2D urban scenes + exact few-bounce specular paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from env.raster import scene_channels
from env.scene import Scene, random_canyon_scene, random_scene
from pathfind.image_method import find_paths

from .schema import GRID, MAX_BOUNCES, pack_sample


def _one_scene(
    rng: np.random.Generator,
    scene_id: int,
    prefer_nlos: bool,
    max_bounces: int,
    max_paths: int,
    n_rects: Optional[int] = None,
) -> Optional[tuple[Scene, list]]:
    if rng.random() < 0.55:
        scene = random_canyon_scene(rng, scene_id=scene_id)
    else:
        if n_rects is None:
            n_rects = int(rng.integers(2, 5))
        scene = random_scene(rng, n_rects=n_rects, scene_id=scene_id)
    if scene is None:
        return None
    paths = find_paths(scene, max_bounces=max_bounces, max_paths=max_paths)
    if not paths:
        return None
    has_bounce = any(p.n_bounces >= 1 for p in paths)
    if prefer_nlos and not has_bounce:
        return None
    # Drop LoS-only scenes occasionally even if not prefer_nlos, to keep
    # bounce paths common enough for the AR study.
    if (not prefer_nlos) and (not has_bounce) and rng.random() < 0.7:
        return None
    return scene, paths


def generate_split(
    rng: np.random.Generator,
    n_scenes: int,
    split: str,
    start_id: int = 0,
    max_bounces: int = MAX_BOUNCES,
    max_paths: int = 8,
    nlos_frac: float = 0.75,
    max_attempts_factor: int = 30,
) -> list[dict]:
    samples: list[dict] = []
    attempts = 0
    n_done = 0
    cap = max(n_scenes * max_attempts_factor, n_scenes + 50)
    pbar = tqdm(total=n_scenes, desc=f"scenes[{split}]", leave=False)
    while n_done < n_scenes and attempts < cap:
        attempts += 1
        sid = start_id + n_done
        prefer_nlos = rng.random() < nlos_frac
        got = _one_scene(rng, sid, prefer_nlos, max_bounces, max_paths)
        if got is None:
            continue
        scene, paths = got
        channels = scene_channels(scene, grid=GRID)
        for pid, path in enumerate(paths):
            samples.append(
                pack_sample(
                    scene,
                    path,
                    channels,
                    split,
                    pid,
                    is_shortest=(pid == 0),
                )
            )
        n_done += 1
        pbar.update(1)
    pbar.close()
    return samples


def _stack(samples: list[dict], key: str) -> np.ndarray:
    return np.stack([s["arrays"][key] for s in samples], axis=0)


def save_npz(samples: list[dict], path: Path) -> None:
    keys = samples[0]["arrays"].keys()
    payload = {k: _stack(samples, k) for k in keys}
    # split name as bytes
    splits = np.array([s["json"]["split"] for s in samples])
    payload["split"] = splits
    np.savez_compressed(path, **payload)


def save_jsonl(samples: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s["json"], ensure_ascii=False) + "\n")


def generate_dataset(
    out_dir: str | Path,
    seed: int = 0,
    n_train: int = 280,
    n_val: int = 60,
    n_test: int = 80,
    max_bounces: int = MAX_BOUNCES,
    max_paths: int = 8,
) -> dict[str, int]:
    """Write JSONL (human schema) + NPZ (training tensors)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train = generate_split(rng, n_train, "train", start_id=0, max_bounces=max_bounces, max_paths=max_paths)
    val = generate_split(rng, n_val, "val", start_id=10_000, max_bounces=max_bounces, max_paths=max_paths)
    test = generate_split(rng, n_test, "test", start_id=20_000, max_bounces=max_bounces, max_paths=max_paths)
    all_samples = train + val + test
    save_npz(all_samples, out / "dataset.npz")
    save_jsonl(all_samples, out / "dataset.jsonl")
    # Tiny schema example
    example = {
        "description": "Each JSONL row is one specular path in one scene.",
        "tokens": "TX → wall_k → ... → RX; wall IDs are stable per scene.",
        "t_on_wall": "1-D parameter in (0,1) of each bounce along that named edge.",
        "points": "Polyline including Tx and Rx.",
        "n_samples": len(all_samples),
        "n_train_paths": len(train),
        "n_val_paths": len(val),
        "n_test_paths": len(test),
        "example": train[0]["json"] if train else {},
    }
    (out / "schema_example.json").write_text(json.dumps(example, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {
        "train_paths": len(train),
        "val_paths": len(val),
        "test_paths": len(test),
        "train_scenes": len({s["json"]["scene_id"] for s in train}),
        "val_scenes": len({s["json"]["scene_id"] for s in val}),
        "test_scenes": len({s["json"]["scene_id"] for s in test}),
    }
    (out / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts
