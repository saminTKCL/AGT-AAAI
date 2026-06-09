#!/usr/bin/env python3
"""Copy all geom-gcn split npz files into data/geom_splits/ for unified loading."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "geom_splits"
OUT.mkdir(parents=True, exist_ok=True)

copied = 0
for npz in DATA.rglob("*split_0.6_0.2_*.npz"):
    dest = OUT / npz.name
    if not dest.exists() or dest.stat().st_size != npz.stat().st_size:
        shutil.copy2(npz, dest)
        copied += 1

print(f"Synced {copied} new files to {OUT} ({len(list(OUT.glob('*.npz')))} total)")
