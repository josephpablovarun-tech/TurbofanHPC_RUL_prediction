"""Engine-separated train/validation split with a saved manifest (FRD FR03)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def split_engines(engine_ids, val_fraction: float = 0.2, seed: int = 42) -> dict:
    ids = np.array(sorted({int(e) for e in engine_ids}))
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    val = np.sort(rng.choice(ids, size=n_val, replace=False))
    train = np.setdiff1d(ids, val)
    return {
        "seed": seed,
        "val_fraction": val_fraction,
        "n_train_engines": int(len(train)),
        "n_val_engines": int(len(val)),
        "train_engine_ids": [int(x) for x in train],
        "val_engine_ids": [int(x) for x in val],
    }


def save_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
