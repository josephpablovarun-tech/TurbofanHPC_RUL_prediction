"""Simulate batches of "new engines" from the FD001 train and test files, for the fleet-scoring demo.

Each simulated engine starts from a real engine (a TRAIN engine with its full run to failure, or a TEST engine with its supplied RUL label),
is cut off at a chosen number of cycles before the end of its life, and gets small Gaussian sensor noise so it is not an exact copy.
The result is a headered CSV in the raw FD001 schema, which is what the app's Fleet scoring view imports. The true remaining life of every
simulated engine is known, and is written to a separate truth file that the app never reads unless you upload it as labels.

Honesty notes: engines derived from TRAIN engines may be ones the saved models were fitted on, so predictions for them are optimistic; the
batches are simulations of the same simulated fleet, not independent real data.

    python -m pipelines.simulate_fleet --out data/simulated
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_loader import load_raw_history, load_test_rul
from src.schema import NUMERIC_COLS, RAW_COLUMNS

MIN_CYCLES = 31  # the shortest observed history in the official test file

PROFILES: Dict[str, Dict] = {
    "batch1": dict(seed=101, rul_range=(60, 150), test_share=0.4, about="healthy fleet: most engines still have plenty of life left"),
    "batch2": dict(seed=202, rul_range=(5, 150), test_share=0.4, about="mixed fleet: engines spread across the whole life"),
    "batch3": dict(seed=303, rul_range=(5, 55), test_share=0.4, about="ageing fleet: many engines close to the end of their life"),
}


def noise_scales(*frames: pd.DataFrame) -> pd.Series:
    """Typical measurement noise per column: the median over engines of std(first difference) / sqrt(2). Zero for constant columns."""
    out = {}
    for c in NUMERIC_COLS:
        per_engine = [g[c].diff().dropna().std(ddof=0) / np.sqrt(2) for f in frames for _, g in f.groupby("engine_id") if len(g) > 2]
        out[c] = float(np.nanmedian(per_engine)) if per_engine else 0.0
    return pd.Series(out)


def _sources(train: pd.DataFrame, test: pd.DataFrame, labels: pd.Series) -> Tuple[List[Dict], List[Dict]]:
    tr = [dict(src="train", engine=int(e), df=g.sort_values("cycle").reset_index(drop=True), total=int(g["cycle"].max()), last=int(g["cycle"].max()))
          for e, g in train.groupby("engine_id")]
    te = [dict(src="test", engine=int(e), df=g.sort_values("cycle").reset_index(drop=True), total=int(g["cycle"].max()) + int(labels.loc[e]),
               last=int(g["cycle"].max())) for e, g in test.groupby("engine_id") if g["cycle"].max() >= MIN_CYCLES]
    return tr, te


def simulate_batch(train: pd.DataFrame, test: pd.DataFrame, labels: pd.Series, n: int, seed: int, rul_range: Tuple[int, int],
                   test_share: float, noise_scale: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (readings, truth). Engine ids are 1..n. ``truth`` holds each engine's true remaining life and where it came from."""
    rng = np.random.default_rng(seed)
    scales = noise_scales(train, test)
    train_src, test_src = _sources(train, test, labels)
    parts, truth = [], []
    for k in range(1, n + 1):
        pool = test_src if rng.random() < test_share else train_src
        s = pool[int(rng.integers(len(pool)))]
        r = int(rng.integers(rul_range[0], rul_range[1] + 1))
        if s["src"] == "test":  # a test engine cannot be extended past its last observed cycle
            r = max(r, s["total"] - s["last"])
        cutoff = int(np.clip(s["total"] - r, MIN_CYCLES, s["last"]))
        h = s["df"].loc[s["df"]["cycle"] <= cutoff].copy()
        for c in NUMERIC_COLS:
            if scales[c] > 0:
                h[c] = h[c] + rng.normal(0.0, noise_scale * scales[c], size=len(h))
        h["engine_id"] = k
        parts.append(h[RAW_COLUMNS])
        truth.append(dict(engine_id=k, source=s["src"], source_engine=s["engine"], last_cycle=cutoff, true_rul=s["total"] - cutoff,
                          total_life=s["total"], true_life_used=round(cutoff / s["total"], 4)))
    df = pd.concat(parts, ignore_index=True)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].round(4)
    return df, pd.DataFrame(truth)


def write_batches(raw_dir: Path, out_dir: Path, n: int = 100) -> Dict[str, Dict]:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    train = load_raw_history(raw_dir / "train_FD001.txt")
    test = load_raw_history(raw_dir / "test_FD001.txt")
    labels = load_test_rul(raw_dir / "RUL_FD001.txt")
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, p in PROFILES.items():
        df, truth = simulate_batch(train, test, labels, n, p["seed"], p["rul_range"], p["test_share"])
        df.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.4f")
        truth.to_csv(out_dir / "truth" / f"{name}_truth.csv", index=False)
        summary[name] = dict(rows=len(df), engines=n, about=p["about"], truly_over_80=int((truth["true_life_used"] >= 0.8).sum()),
                             from_test=int((truth["source"] == "test").sum()))
    (out_dir / "README.md").write_text(
        "# Simulated fleet batches (for the Fleet scoring demo)\n\n"
        "Generated by `python -m pipelines.simulate_fleet` from `train_FD001` and `test_FD001` (+ `RUL_FD001`). Each file has 100 engines in the raw "
        "FD001 schema with a header row. Every engine is a real FD001 engine cut off at a chosen point, with small Gaussian sensor noise added.\n\n"
        + "\n".join(f"* `{k}.csv`: {v['about']} ({v['rows']} rows; {v['from_test']} engines from test, {v['engines'] - v['from_test']} from train; "
                    f"{v['truly_over_80']} engines truly at or past 80% of their life)" for k, v in summary.items())
        + "\n\n`truth/batchN_truth.csv` holds each engine's true remaining life and source engine. The app never reads it unless you upload it as labels.\n\n"
        "Engines derived from train engines may be ones the saved models were fitted on, so their predictions are optimistic. These are simulations "
        "of the same simulated fleet, not independent real data.\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", default="data/raw/CMAPSS")
    ap.add_argument("--out", default="data/simulated")
    ap.add_argument("--n", type=int, default=100, help="engines per batch")
    a = ap.parse_args()
    for name, v in write_batches(Path(a.raw_dir), Path(a.out), a.n).items():
        print(f"{name}: {v['rows']} rows, {v['engines']} engines ({v['from_test']} from test), {v['truly_over_80']} truly at or past 80% of life | {v['about']}")


if __name__ == "__main__":
    main()
