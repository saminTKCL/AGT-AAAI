"""Dataset loading with mandatory real geom-gcn 10-split protocol."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Actor, Planetoid, WebKB, WikipediaNetwork
from torch_geometric.utils import add_self_loops

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"

DATASETS = [
    "Texas",
    "Cornell",
    "Wisconsin",
    "Actor",
    "Chameleon",
    "Squirrel",
    "Cora",
    "Citeseer",
    "Pubmed",
]

GRAIN_PAPER_RESULTS = {
    "Cora": 88.52,
    "Citeseer": 81.25,
    "Pubmed": 87.04,
    "Cornell": 90.12,
    "Wisconsin": 89.01,
    "Texas": 87.69,
    "Actor": 38.89,
    "Chameleon": 56.43,
    "Squirrel": 53.43,
}


def _dataset_key(name: str) -> str:
    return "film" if name.lower() == "actor" else name.lower()


def _dataset_path(name: str) -> Path:
    return DATA_ROOT / name


def _split_candidates(name: str, seed: int) -> list[Path]:
    key = _dataset_key(name)
    sid = seed % 10
    fname = f"{key}_split_0.6_0.2_{sid}.npz"
    candidates = [
        DATA_ROOT / "geom_splits" / fname,
        _dataset_path(name) / "raw" / fname,
        _dataset_path(name) / fname,
        _dataset_path(name) / key / "raw" / fname,
        _dataset_path(name) / "geom_gcn" / "raw" / fname,
        _dataset_path(name) / key / "geom_gcn" / "raw" / fname,
        ROOT / "experiments" / "data" / name / "raw" / fname,
        ROOT / "experiments" / "data" / name / key / "geom_gcn" / "raw" / fname,
    ]
    if name in ("Chameleon", "Squirrel"):
        candidates.insert(1, _dataset_path(name) / name.lower() / "geom_gcn" / "raw" / fname)
    return candidates


def load_geom_split(name: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Path]:
    """Load official geom-gcn split masks. Raises if not found."""
    for split_file in _split_candidates(name, seed):
        if split_file.exists():
            masks = np.load(split_file)
            return (
                torch.from_numpy(masks["train_mask"].astype(bool)),
                torch.from_numpy(masks["val_mask"].astype(bool)),
                torch.from_numpy(masks["test_mask"].astype(bool)),
                split_file,
            )
    tried = "\n  ".join(str(p) for p in _split_candidates(name, seed))
    raise FileNotFoundError(
        f"No real geom-gcn split for {name} seed={seed}. Tried:\n  {tried}\n"
        f"Run: python3 scripts/download_geom_splits.py && python3 scripts/sync_geom_splits.py"
    )


def load_dataset(name: str, seed: int = 0, device: torch.device | None = None) -> Data:
    """Load real graph with official geom-gcn split (never synthetic)."""
    name = name.strip()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}. Choose from {DATASETS}")

    path = str(_dataset_path(name))
    os.makedirs(path, exist_ok=True)

    if name in ("Cora", "Citeseer", "Pubmed"):
        dataset = Planetoid(path, name)
        data = dataset[0]
    elif name in ("Texas", "Cornell", "Wisconsin"):
        dataset = WebKB(path, name)
        data = dataset[0]
    elif name == "Actor":
        dataset = Actor(path)
        data = dataset[0]
    elif name in ("Chameleon", "Squirrel"):
        dataset = WikipediaNetwork(path, name.lower())
        data = dataset[0]
    else:
        raise ValueError(name)

    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    train_m, val_m, test_m, split_path = load_geom_split(name, seed)

    data.train_mask = train_m
    data.val_mask = val_m
    data.test_mask = test_m
    data.split_source = str(split_path)

    if device is not None:
        data = data.to(device)
    return data


def edge_homophily(data: Data) -> float:
    src, dst = data.edge_index
    same = (data.y[src] == data.y[dst]).float()
    return float(same.mean().item())


def verify_all_splits() -> list[str]:
    """Return list of missing splits."""
    missing = []
    for name in DATASETS:
        for seed in range(10):
            try:
                load_geom_split(name, seed)
            except FileNotFoundError:
                missing.append(f"{name}/seed{seed}")
    return missing
