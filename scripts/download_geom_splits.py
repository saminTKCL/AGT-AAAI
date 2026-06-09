#!/usr/bin/env python3
"""Download official geom-gcn 10-split masks from the Pei et al. repository."""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "geom_splits"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/splits"
DATASETS = [
    "cora", "citeseer", "pubmed",
    "texas", "cornell", "wisconsin",
    "chameleon", "squirrel", "film",
]

for ds in DATASETS:
    for seed in range(10):
        fname = f"{ds}_split_0.6_0.2_{seed}.npz"
        dest = OUT / fname
        if dest.exists() and dest.stat().st_size > 100:
            continue
        url = f"{BASE}/{fname}"
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"OK  {fname}")
        except Exception as exc:
            print(f"SKIP {fname}: {exc}")

print(f"Done. {len(list(OUT.glob('*.npz')))} files in {OUT}")
