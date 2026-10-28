"""Shared fixtures: a synthetic FD001-style dataset and small real XGBoost / Random Forest bundles.

The bundles use the same dictionary layout as model_save/*.joblib, so the whole stack (feature
builder, predictor, tools, agents) is exercised without the real NASA files or the real models.
Run Agent/tests/test_real_artifacts.py for checks against your actual data and saved models.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Agent.config import load_config  # noqa: E402
from src.features import MODEL_SENSORS, build_features, extended_feature_columns  # noqa: E402
from src.schema import RAW_COLUMNS  # noqa: E402

# test-engine last cycles known from the real FD001 test file (used by the eval cases); the rest are synthetic
KNOWN_TEST_LENGTHS = {1: 31, 2: 49, 3: 126, 4: 106, 24: 186, 45: 152, 67: 71, 78: 72}


def make_engine(engine_id, n, rng):
    t = np.arange(1, n + 1)
    row = {"engine_id": engine_id, "cycle": t, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
    for i in range(1, 22):
        drift = (i % 5 - 2) * 0.5
        row[f"sensor_{i}"] = 500 + i + drift * (t / n) ** 2 * 10 + rng.normal(0, 0.3, n)
    return pd.DataFrame(row)


def write_fd001(raw_dir: Path, seed=0):
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    train = pd.concat([make_engine(i, int(rng.integers(150, 300)), rng) for i in range(1, 101)], ignore_index=True)
    test_lengths = {i: KNOWN_TEST_LENGTHS.get(i, int(rng.integers(40, 200))) for i in range(1, 101)}
    test = pd.concat([make_engine(i, test_lengths[i], rng) for i in range(1, 101)], ignore_index=True)
    for name, df in (("train_FD001.txt", train), ("test_FD001.txt", test)):
        df[RAW_COLUMNS].to_csv(raw_dir / name, sep=" ", header=False, index=False, float_format="%.4f")
    labels = np.random.default_rng(seed + 1).integers(5, 120, size=100)  # synthetic official RUL label per test engine (engine ids 1..100)
    (raw_dir / "RUL_FD001.txt").write_text("".join(f"{int(v)} \n" for v in labels))
    return train, test


def write_bundles(model_dir: Path, train: pd.DataFrame, seed=0):
    model_dir.mkdir(parents=True, exist_ok=True)
    train = train.copy()
    train["rul_cycles"] = train.groupby("engine_id")["cycle"].transform("max") - train["cycle"]
    feats = build_features(train)
    cols = extended_feature_columns()
    X, y = feats[cols].to_numpy(float), np.sqrt(feats["rul_cycles"].to_numpy(float))
    ids = list(range(1, 71))
    base = {"features": cols, "feature_set": "extended", "target": "sqrt", "train_engine_ids": ids, "seed": seed,
            "versions": {"note": "synthetic test bundle"}}
    models = {"xgboost": XGBRegressor(n_estimators=30, max_depth=4, random_state=seed, n_jobs=1),
              "random_forest": RandomForestRegressor(n_estimators=20, min_samples_leaf=5, random_state=seed, n_jobs=1)}
    for name, m in models.items():
        m.fit(X, y)
        joblib.dump({**base, "name": name, "model": m, "settings": {}}, model_dir / f"{name}.joblib", compress=3)


@pytest.fixture(scope="session")
def fixture_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("agentfix")
    train, _ = write_fd001(root / "raw")
    write_bundles(root / "models", train[train["engine_id"] <= 30])
    return {"raw": root / "raw", "models": root / "models", "train": train}


@pytest.fixture(scope="session")
def cfg(fixture_dirs):
    c = load_config()
    c["data"]["raw_dir"] = str(fixture_dirs["raw"])
    c["model"]["model_dir"] = str(fixture_dirs["models"])
    c["alerts"] = {"enabled": False}  # tests must never post to Slack, even if a webhook URL is set in the environment
    return c


@pytest.fixture()
def system(cfg, tmp_path):
    from Agent.orchestration.orchestrator import AgentSystem
    c = dict(cfg)
    c["logging"] = {"trace_file": str(tmp_path / "traces.jsonl")}
    return AgentSystem.from_config(c)
