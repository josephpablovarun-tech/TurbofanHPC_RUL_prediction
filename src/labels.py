"""RUL labels (FRD FR02). Uncapped by default (FR06)."""
from __future__ import annotations

import pandas as pd


def add_training_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL = final recorded cycle of that engine minus the current cycle. Training data only."""
    out = df.copy()
    out["rul_cycles"] = out.groupby("engine_id")["cycle"].transform("max") - out["cycle"]
    return out


def build_test_endpoint_labels(test_df: pd.DataFrame, rul: pd.Series) -> pd.DataFrame:
    """One row per test engine: last observed cycle plus the supplied RUL at that cycle.

    The last observed test cycle is NOT failure; the supplied label says how many cycles remain.
    """
    last = test_df.groupby("engine_id")["cycle"].max().rename("last_cycle").to_frame()
    missing = set(last.index) - set(rul.index)
    if missing:
        raise ValueError(f"No supplied RUL label for test engine(s): {sorted(missing)[:5]}")
    last["rul_cycles"] = rul.reindex(last.index).to_numpy()
    return last.reset_index()


def add_test_trajectory_rul(test_df: pd.DataFrame, rul: pd.Series) -> pd.DataFrame:
    """Analysis-only truth: endpoint RUL + final observed cycle - current cycle (FR10)."""
    ends = build_test_endpoint_labels(test_df, rul).set_index("engine_id")
    out = test_df.copy()
    out["rul_cycles"] = (
        out["engine_id"].map(ends["rul_cycles"]) + out["engine_id"].map(ends["last_cycle"]) - out["cycle"]
    )
    return out
