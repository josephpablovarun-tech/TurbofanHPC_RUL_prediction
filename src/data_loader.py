"""Load whitespace-separated C-MAPSS files or CSV equivalents (FRD FR01)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.schema import RAW_COLUMNS
from src.validation import DataValidationError


def load_raw_history(path: str | Path) -> pd.DataFrame:
    """Read a history file. Trailing empty columns are removed only if confirmed entirely empty."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if list(df.columns[: len(RAW_COLUMNS)]) != RAW_COLUMNS and df.shape[1] == len(RAW_COLUMNS):
            # Headerless CSV: assign the documented names
            df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    else:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        extra = df.columns[len(RAW_COLUMNS):]
        if len(extra):
            if df[extra].isna().all().all():
                df = df.drop(columns=extra)
            else:
                raise DataValidationError(
                    [f"{path.name}: {len(extra)} unexpected non-empty column(s) beyond the {len(RAW_COLUMNS)} documented"]
                )
        if df.shape[1] != len(RAW_COLUMNS):
            raise DataValidationError(
                [f"{path.name}: expected {len(RAW_COLUMNS)} columns, found {df.shape[1]}"]
            )
        df.columns = RAW_COLUMNS
    for c in ("engine_id", "cycle"):
        if c in df.columns and df[c].notna().all():
            df[c] = df[c].astype(int)
    return df


def load_test_rul(path: str | Path) -> pd.Series:
    """Official RUL labels: one value per test engine, in engine_id order (1..N)."""
    s = pd.read_csv(path, sep=r"\s+", header=None, engine="python")[0]
    s.index = range(1, len(s) + 1)
    s.name = "rul_cycles"
    s.index.name = "engine_id"
    return s
