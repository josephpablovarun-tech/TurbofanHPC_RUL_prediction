"""Raw-data validation (FRD FR01). Rejects bad data with actionable errors; never drops rows silently."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import NUMERIC_COLS, RAW_COLUMNS


class DataValidationError(ValueError):
    """Raised when raw data fails validation. `.errors` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Data validation failed:\n- " + "\n- ".join(errors))


def find_cycle_gaps(df: pd.DataFrame) -> dict[int, list[int]]:
    """Return {engine_id: [cycles after which a gap occurs]}. Gaps are warnings, not errors."""
    gaps: dict[int, list[int]] = {}
    for eid, g in df.groupby("engine_id", sort=True):
        c = g["cycle"].to_numpy()
        idx = np.where(np.diff(c) > 1)[0]
        if len(idx):
            gaps[int(eid)] = [int(c[i]) for i in idx]
    return gaps


def validate_raw(df: pd.DataFrame) -> dict:
    """Validate a raw history frame. Returns a report dict; raises DataValidationError on failure."""
    errors: list[str] = []

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        # Without the columns the remaining checks are not meaningful.
        raise DataValidationError([f"Missing required columns: {missing}"])

    # Non-finite values (NaN / inf) in any required column
    for col in RAW_COLUMNS:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(vals)
        if bad.any():
            rows = df.index[bad].tolist()[:5]
            errors.append(f"Column '{col}' has {int(bad.sum())} non-finite value(s); first rows: {rows}")

    if errors:
        raise DataValidationError(errors)

    # Positive integers for ids
    for col in ("engine_id", "cycle"):
        v = df[col].to_numpy(dtype=float)
        if (v <= 0).any() or (v != np.floor(v)).any():
            errors.append(f"Column '{col}' must contain positive integers")

    # Duplicate keys
    dup = df.duplicated(subset=["engine_id", "cycle"], keep=False)
    if dup.any():
        keys = df.loc[dup, ["engine_id", "cycle"]].drop_duplicates().head(5).values.tolist()
        errors.append(f"{int(dup.sum())} rows share a duplicate (engine_id, cycle) key; examples: {keys}")

    # Cycle order must be strictly increasing within each engine, in file order
    for eid, g in df.groupby("engine_id", sort=True):
        if (np.diff(g["cycle"].to_numpy()) <= 0).any():
            errors.append(f"Engine {int(eid)}: cycle is not strictly increasing in file order")
            break

    if errors:
        raise DataValidationError(errors)

    gaps = find_cycle_gaps(df)
    return {
        "n_rows": int(len(df)),
        "n_engines": int(df["engine_id"].nunique()),
        "constant_columns": [c for c in NUMERIC_COLS if df[c].nunique() <= 1],
        "cycle_gaps": gaps,
        "warnings": [f"Engine {e} has cycle gaps after cycles {c}" for e, c in gaps.items()],
    }
