"""Descriptive and exploratory analysis helpers for the C-MAPSS FD001 files (pure pandas / numpy, no Streamlit).

Used by the Data explorer view. The numbers are computed from the files themselves. "Early life" and "late life" mean the first and
last 20% of an engine's run to failure, so those comparisons need run-to-failure histories (the train file). Correlations and shifts
describe observed associations, not causes.
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.schema import NUMERIC_COLS

EDGE = 0.2  # first / last 20% of an engine's life


def lifetimes(df: pd.DataFrame) -> pd.Series:
    """Last recorded cycle of each engine (the total life for a train engine; the observed length for a test engine)."""
    return df.groupby("engine_id")["cycle"].max()


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Train files only: RUL = the engine's last cycle minus the current cycle (uncapped)."""
    out = df.copy()
    out["rul"] = out.groupby("engine_id")["cycle"].transform("max") - out["cycle"]
    return out


def overview(df: pd.DataFrame) -> Dict[str, Any]:
    life = lifetimes(df)
    const = [c for c in NUMERIC_COLS if c in df.columns and df[c].nunique(dropna=True) <= 1]
    return {"rows": int(len(df)), "engines": int(df["engine_id"].nunique()), "columns": int(df.shape[1]),
            "cycles_min": int(life.min()), "cycles_median": float(life.median()), "cycles_mean": float(life.mean()), "cycles_max": int(life.max()),
            "missing_cells": int(df.isna().sum().sum()), "duplicate_keys": int(df.duplicated(["engine_id", "cycle"]).sum()),
            "constant_columns": const}


def describe_table(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    d = df[list(cols)].describe().T
    d["skew"] = df[list(cols)].skew()
    d["unique values"] = df[list(cols)].nunique()
    d["constant"] = d["unique values"] <= 1
    return d.reset_index().rename(columns={"index": "column"})


def histogram(values, bins: int = 30, value_range: Optional[Sequence[float]] = None) -> pd.DataFrame:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return pd.DataFrame({"start": [], "end": [], "count": []})
    counts, edges = np.histogram(v, bins=bins, range=value_range)
    return pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts})


def _phase_masks(df: pd.DataFrame, edge: float = EDGE):
    frac = df["cycle"] / df.groupby("engine_id")["cycle"].transform("max")
    return frac <= edge, frac > 1 - edge


def phase_histograms(df: pd.DataFrame, col: str, bins: int = 30, edge: float = EDGE) -> pd.DataFrame:
    """Histogram of ``col`` for the first vs the last 20% of each engine's life, as a percentage of each phase's rows (run-to-failure data)."""
    early, late = _phase_masks(df, edge)
    lo, hi = float(df[col].min()), float(df[col].max())
    parts = []
    for name, mask in (("early life (first 20%)", early), ("late life (last 20%)", late)):
        h = histogram(df.loc[mask, col], bins, (lo, hi))
        h["percent"] = 100.0 * h["count"] / max(1, int(mask.sum()))
        h["phase"] = name
        parts.append(h)
    return pd.concat(parts, ignore_index=True)


def early_late_shift(df: pd.DataFrame, cols: Sequence[str], edge: float = EDGE) -> pd.DataFrame:
    """Mean in the last 20% of life minus the mean in the first 20%, in units of the column's overall standard deviation, largest first."""
    early, late = _phase_masks(df, edge)
    rows = []
    for c in cols:
        sd = float(df[c].std(ddof=0))
        e, l = float(df.loc[early, c].mean()), float(df.loc[late, c].mean())
        rows.append({"column": c, "early mean": e, "late mean": l, "shift (std units)": 0.0 if sd == 0 else (l - e) / sd})
    out = pd.DataFrame(rows, columns=["column", "early mean", "late mean", "shift (std units)"])
    order = np.argsort(-out["shift (std units)"].abs().to_numpy(), kind="stable")
    return out.iloc[order].reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame, cols: Sequence[str], method: str = "pearson") -> pd.DataFrame:
    keep = [c for c in cols if df[c].nunique() > 1]  # a constant column has no defined correlation
    return df[keep].corr(method=method)


def correlation_pairs(corr: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    cols, rows = list(corr.columns), []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                rows.append({"a": cols[i], "b": cols[j], "correlation": round(float(r), 3)})
    out = pd.DataFrame(rows, columns=["a", "b", "correlation"])
    if out.empty:
        return out
    return out.iloc[np.argsort(-out["correlation"].abs().to_numpy(), kind="stable")].reset_index(drop=True)


def rul_correlations(df_with_rul: pd.DataFrame, cols: Sequence[str], method: str = "spearman") -> pd.DataFrame:
    rows = []
    for c in cols:
        if df_with_rul[c].nunique() > 1:
            rows.append({"column": c, "correlation with RUL": round(float(df_with_rul[c].corr(df_with_rul["rul"], method=method)), 3)})
    out = pd.DataFrame(rows, columns=["column", "correlation with RUL"])
    if out.empty:
        return out
    return out.iloc[np.argsort(-out["correlation with RUL"].abs().to_numpy(), kind="stable")].reset_index(drop=True)


def quality_table(df: pd.DataFrame, cols: Sequence[str] = tuple(NUMERIC_COLS)) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        rows.append({"column": c, "missing": int(s.isna().sum()), "unique values": int(s.nunique()), "constant": bool(s.nunique() <= 1),
                     "min": float(s.min()), "max": float(s.max()), "outliers (IQR rule)": n_out, "outliers %": round(100.0 * n_out / max(1, len(s)), 2)})
    return pd.DataFrame(rows)


def cycle_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Engines with missing cycle numbers between their first and last cycle (empty when every history is contiguous)."""
    g = df.groupby("engine_id")["cycle"]
    out = pd.DataFrame({"engine_id": g.size().index, "rows": g.size().to_numpy(), "first_cycle": g.min().to_numpy(), "last_cycle": g.max().to_numpy()})
    out["missing_cycles"] = out["last_cycle"] - out["first_cycle"] + 1 - out["rows"]
    return out[out["missing_cycles"] > 0].reset_index(drop=True)


def trend_frame(df: pd.DataFrame, engines: Sequence[int], sensors: Sequence[str], x: str = "cycle", standardise: bool = False) -> pd.DataFrame:
    """Long-format sensor histories for a few engines. ``x`` is "cycle", or "rul" (run-to-failure data only). Standardising uses the whole file's mean and std."""
    sub = df[df["engine_id"].isin(list(engines))]
    xs = (sub.groupby("engine_id")["cycle"].transform("max") - sub["cycle"]) if x == "rul" else sub["cycle"]
    parts = []
    for s in sensors:
        v = sub[s].astype(float)
        if standardise:
            sd = float(df[s].std(ddof=0))
            v = (v - float(df[s].mean())) / sd if sd > 0 else v * 0.0
        parts.append(pd.DataFrame({"x": xs.to_numpy(), "value": v.to_numpy(), "engine": sub["engine_id"].astype(str).to_numpy(), "sensor": s}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"x": [], "value": [], "engine": [], "sensor": []})


# ---------------------------------------------------------------- RUL label file
def label_join(test_df: pd.DataFrame, rul: pd.Series) -> pd.DataFrame:
    """One row per test engine: observed cycles, the official RUL label and the total life they imply (observed + label)."""
    observed = lifetimes(test_df)
    out = pd.DataFrame({"engine_id": observed.index, "observed_cycles": observed.to_numpy()})
    out["rul_label"] = out["engine_id"].map(rul).astype(int)
    out["implied_total_life"] = out["observed_cycles"] + out["rul_label"]
    out["observed_fraction"] = (out["observed_cycles"] / out["implied_total_life"]).round(3)
    return out


def band_counts(values: Sequence[float], normal_above: float, watch_above: float) -> Dict[str, int]:
    v = np.asarray(values, dtype=float)
    return {"Normal": int((v > normal_above).sum()), "Watch": int(((v > watch_above) & (v <= normal_above)).sum()), "Review": int((v <= watch_above).sum())}
