"""Import a fleet of engines from an uploaded file and score the last cycle of each engine (FD001-like data only).

Checks follow the requirements doc (UI06/UI07): problems with the file as a whole reject it with an actionable message; a problem with one
engine (non-finite values, duplicate or out-of-order cycles, a failed prediction) rejects only that engine and the rest are still scored.
Scoring uses the same predictor as the rest of the app and reads only the uploaded rows. Optional true-RUL labels are used for the
evaluation table only; they never reach the model or a Slack message.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Agent.tools.alerts import life_used
from src.schema import NUMERIC_COLS, RAW_COLUMNS

MAX_BYTES = 10 * 1024 * 1024  # FRD UI06 defaults
MAX_ROWS = 50_000


class UploadError(ValueError):
    """The file as a whole cannot be used; the message says what to fix."""


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_upload(data: bytes, name: str = "the upload") -> Tuple[pd.DataFrame, List[str]]:
    """Parse CSV (with or without a header) or whitespace-separated 26-column text into the raw FD001 schema."""
    if len(data) > MAX_BYTES:
        raise UploadError(f"{name} is {len(data) / 1e6:.1f} MB; the limit is {MAX_BYTES // (1024 * 1024)} MB.")
    if not data.strip():
        raise UploadError(f"{name} is empty.")
    text = data.decode("utf-8", errors="replace")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    tokens = [t for t in re.split(r"[,\s]+", first.strip()) if t]
    headerless = all(_is_number(t) for t in tokens)
    warnings: List[str] = []
    try:
        if "," in first:
            df = pd.read_csv(io.StringIO(text), header=None if headerless else 0)
        else:
            df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None if headerless else 0, engine="python")
    except Exception as e:
        raise UploadError(f"{name} could not be read as a table ({type(e).__name__}).") from None
    if headerless:
        extra = df.columns[len(RAW_COLUMNS):]
        if len(extra) and df[extra].isna().all().all():
            df = df.drop(columns=extra)
        if df.shape[1] != len(RAW_COLUMNS):
            raise UploadError(f"{name}: expected {len(RAW_COLUMNS)} columns (engine_id, cycle, 3 settings, 21 sensors), found {df.shape[1]}.")
        df.columns = RAW_COLUMNS
    else:
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            raise UploadError(f"{name} is missing required column(s): {', '.join(missing)}.")
        ignored = [c for c in df.columns if c not in RAW_COLUMNS]
        if ignored:
            warnings.append(f"Ignored extra column(s): {', '.join(ignored)}.")
        df = df[RAW_COLUMNS]
    if len(df) > MAX_ROWS:
        raise UploadError(f"{name} has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    df = df.apply(pd.to_numeric, errors="coerce")
    bad = df["engine_id"].isna() | df["cycle"].isna() | (df["engine_id"] % 1 != 0) | (df["cycle"] % 1 != 0)
    if bad.any():
        raise UploadError(f"engine_id and cycle must be whole numbers; {int(bad.sum())} row(s) are not (first at data row {int(bad.idxmax()) + 1}).")
    df["engine_id"], df["cycle"] = df["engine_id"].astype(int), df["cycle"].astype(int)
    return df.reset_index(drop=True), warnings


def parse_labels(data: bytes, engine_ids: Sequence[int], name: str = "the labels file") -> pd.Series:
    """True RUL per engine, from either one number per line (engine order, like RUL_FD001) or a CSV with engine_id and true_rul columns."""
    text = data.decode("utf-8", errors="replace")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    ids = sorted(int(e) for e in engine_ids)
    if any(c.isalpha() for c in first):
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise UploadError(f"{name} could not be read ({type(e).__name__}).") from None
        col = next((c for c in ("true_rul", "rul", "rul_cycles") if c in df.columns), None)
        if "engine_id" not in df.columns or col is None:
            raise UploadError(f"{name} needs the columns engine_id and true_rul, or one number per line.")
        s = pd.Series(df[col].to_numpy(float), index=df["engine_id"].astype(int).to_numpy())
        missing = [e for e in ids if e not in s.index]
        if missing:
            raise UploadError(f"{name} has no label for engine(s) {missing[:5]}{'...' if len(missing) > 5 else ''}.")
        return s.loc[ids].rename("true_rul")
    vals = [float(t) for t in re.split(r"[,\s]+", text.strip()) if t]
    if len(vals) != len(ids):
        raise UploadError(f"{name} has {len(vals)} value(s) but the readings have {len(ids)} engine(s); give one value per engine, in engine_id order.")
    return pd.Series(vals, index=ids, name="true_rul")


def demo_band(rul: float, th: Dict[str, Any]) -> str:
    return "Normal" if rul > th["band_normal_above"] else ("Watch" if rul > th["band_watch_above"] else "Review")


@dataclass
class FleetResult:
    table: pd.DataFrame
    rejected: List[Tuple[int, str]]
    n_engines: int
    flagged: int
    model_version: str
    evaluation: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)


def _engine_problem(g: pd.DataFrame) -> Optional[str]:
    bad_cols = [c for c in NUMERIC_COLS if not np.isfinite(g[c].to_numpy(float)).all()]
    if bad_cols:
        return f"missing, non-numeric or non-finite values in {', '.join(bad_cols[:4])}{'...' if len(bad_cols) > 4 else ''}"
    cyc = g["cycle"].to_numpy()
    if (cyc < 1).any():
        return "cycle numbers must be 1 or more"
    if len(set(cyc.tolist())) != len(cyc):
        return "duplicate cycle numbers"
    if (np.diff(cyc) <= 0).any():
        return "cycles are not in increasing order (sort each engine by cycle)"
    return None


COLUMNS = ["engine_id", "last_cycle", "predicted_rul", "support_rul", "gap", "life_used_pct", "band", "flagged", "warnings"]


def score_fleet(df: pd.DataFrame, predictor, th: Dict[str, Any], alert_threshold: float, labels: Optional[pd.Series] = None) -> FleetResult:
    rows, rejected = [], []
    ids = sorted(int(e) for e in df["engine_id"].unique())
    for eid in ids:
        g = df[df["engine_id"] == eid].reset_index(drop=True)
        problem = _engine_problem(g)
        if problem:
            rejected.append((eid, problem))
            continue
        last = int(g["cycle"].iloc[-1])
        try:
            res = predictor.predict(g, last)
        except Exception as e:
            rejected.append((eid, f"prediction failed: {e}"))
            continue
        used = life_used(last, res.primary_rul)
        warns = []
        if len(g) < th["short_history_cycles"]:
            warns.append(f"short history ({len(g)} cycles)")
        if (np.diff(g["cycle"].to_numpy()) > 1).any():
            warns.append("cycle gaps")
        rows.append(dict(engine_id=eid, last_cycle=last, predicted_rul=round(res.primary_rul, 1), support_rul=round(res.support_rul, 1),
                         gap=round(res.gap, 1), life_used_pct=round(100 * used, 1), band=demo_band(res.primary_rul, th),
                         flagged=bool(used >= alert_threshold), warnings="; ".join(warns)))
    table = pd.DataFrame(rows, columns=COLUMNS)
    table = table.sort_values(["life_used_pct", "engine_id"], ascending=[False, True]).reset_index(drop=True)
    result = FleetResult(table=table, rejected=rejected, n_engines=len(ids), flagged=int(table["flagged"].sum()) if len(table) else 0,
                         model_version=predictor.model_version)
    if labels is not None and len(table):
        table["true_rul"] = table["engine_id"].map(labels).astype(float)
        table["error"] = (table["predicted_rul"] - table["true_rul"]).round(1)
        table["true_life_used_pct"] = (100 * table["last_cycle"] / (table["last_cycle"] + table["true_rul"])).round(1)
        table["truly_flagged"] = table["true_life_used_pct"] >= 100 * alert_threshold
        tp = int((table["flagged"] & table["truly_flagged"]).sum())
        fp = int((table["flagged"] & ~table["truly_flagged"]).sum())
        fn = int((~table["flagged"] & table["truly_flagged"]).sum())
        result.evaluation = dict(n=len(table), mae=float(table["error"].abs().mean()), mean_signed_error=float(table["error"].mean()),
                                 over20=float((table["error"] > 20).mean()), true_positive=tp, false_positive=fp, false_negative=fn,
                                 true_negative=len(table) - tp - fp - fn, precision=(tp / (tp + fp)) if tp + fp else None,
                                 recall=(tp / (tp + fn)) if tp + fn else None)
    return result
