"""PyTorch dataset over packed NPZ samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PathNPZDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        split: str | None = None,
        shortest_only: bool = False,
        min_bounces: int = 0,
        max_bounces: int | None = None,
    ):
        blob = np.load(npz_path, allow_pickle=True)
        splits = blob["split"]
        n = len(splits)
        mask = np.ones(n, dtype=bool)
        if split is not None:
            mask &= splits == split
        if shortest_only:
            mask &= blob["is_shortest"] == 1
        if min_bounces:
            mask &= blob["n_bounces"] >= min_bounces
        if max_bounces is not None:
            mask &= blob["n_bounces"] <= max_bounces
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError(f"No samples for split={split} shortest_only={shortest_only}")
        self.idx = idx
        self.data = {k: blob[k][idx] for k in blob.files if k != "split"}
        self.split_names = splits[idx]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        def t(key: str) -> torch.Tensor:
            arr = self.data[key][i]
            if np.issubdtype(arr.dtype, np.floating):
                return torch.from_numpy(np.ascontiguousarray(arr)).float()
            return torch.from_numpy(np.ascontiguousarray(arr)).long()

        bounce_mask = torch.zeros(self.data["t_on_wall"].shape[1], dtype=torch.float32)
        nb = int(self.data["n_bounces"][i])
        if nb > 0:
            bounce_mask[:nb] = 1.0
        return {
            "tokens": t("tokens"),
            "wall_ids": t("wall_ids"),
            "t_on_wall": t("t_on_wall"),
            "points": t("points"),
            "n_bounces": t("n_bounces"),
            "tx": t("tx"),
            "rx": t("rx"),
            "channels": t("channels"),
            "wall_feats": t("wall_feats"),
            "wall_mask": t("wall_mask"),
            "bounce_mask": bounce_mask,
            "is_shortest": t("is_shortest"),
            "scene_id": t("scene_id"),
            "path_id": t("path_id"),
            "n_walls": t("n_walls"),
        }
