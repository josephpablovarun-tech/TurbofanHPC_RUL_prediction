"""Causal feature builder shared by the notebooks, the agent tools and the future API.

Ported from ``build_features`` in ``model/03_model_tuning.ipynb`` (the "extended" feature set,
97 columns). Every feature at a row uses only that engine's current and earlier rows
(FRD FR04, FR05, AT03, AT04). The column order must equal ``bundle["features"]`` of the saved
models; ``src.predictor`` checks this when it loads them.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Sensors the saved models read (generic C-MAPSS identifiers; physical names are not verified).
MODEL_SENSORS: List[str] = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21)]
WINDOWS = (5, 10)
FEATURE_SCHEMA_VERSION = "extended-97-v1"


def extended_feature_columns(sensors: List[str] = MODEL_SENSORS) -> List[str]:
    """Names of the model inputs, in the order the saved models expect."""
    cols = list(sensors)
    for w in WINDOWS:
        for s in sensors:
            cols += [f"{s}_mean_{w}", f"{s}_std_{w}"]
    cols.append("cycle")
    for s in sensors:
        cols += [f"{s}_vs_start", f"{s}_ewm20", f"{s}_mean_20"]
    return cols


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling and smoothed features per engine (identical logic for training and serving)."""
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    g = df.groupby("engine_id")[MODEL_SENSORS]
    cols = {}
    for w in WINDOWS:
        roll = g.rolling(window=w, min_periods=1)
        mean = roll.mean().reset_index(level=0, drop=True).sort_index()
        std = roll.std(ddof=0).reset_index(level=0, drop=True).sort_index()
        for s in MODEL_SENSORS:
            cols[f"{s}_mean_{w}"] = mean[s]
            cols[f"{s}_std_{w}"] = std[s]
    # engine's own early level: running mean for the first 5 cycles, then the mean of cycles 1-5 (all in the past)
    early = g.transform(lambda x: x.expanding().mean().where(np.arange(len(x)) < 5).ffill())
    ewm = g.transform(lambda x: x.ewm(span=20, adjust=False).mean())
    roll20 = g.rolling(window=20, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    for s in MODEL_SENSORS:
        cols[f"{s}_vs_start"] = df[s] - early[s]
        cols[f"{s}_ewm20"] = ewm[s]
        cols[f"{s}_mean_20"] = roll20[s]
    return pd.concat([df, pd.DataFrame(cols)], axis=1)


def feature_row_at_cutoff(history: pd.DataFrame, cutoff_cycle: int) -> pd.DataFrame:
    """One-row frame of model inputs at ``cutoff_cycle`` for a single engine.

    Rows after the cutoff are removed BEFORE any feature is computed, so a later cycle can never
    reach the model even if the caller passes a longer history (AT03).
    """
    needed = ["engine_id", "cycle"] + MODEL_SENSORS
    missing = [c for c in needed if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing required column(s): {missing}")
    if history["engine_id"].nunique() != 1:
        raise ValueError("history must contain exactly one engine")
    hist = history.loc[history["cycle"] <= cutoff_cycle]
    if hist.empty or int(hist["cycle"].max()) != int(cutoff_cycle):
        raise ValueError(f"cutoff cycle {cutoff_cycle} is not an observed cycle of this engine")
    if not np.isfinite(hist[MODEL_SENSORS].to_numpy(float)).all():
        raise ValueError("sensor values must be finite")
    if hist["cycle"].duplicated().any():
        raise ValueError("duplicate cycle values in history")
    return build_features(hist).iloc[[-1]].reset_index(drop=True)
