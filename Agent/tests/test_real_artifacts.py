"""Checks against YOUR real data and saved models. Skipped automatically when they are not present.

Compares the agent's predictor with the predictions saved in artifacts/model (notebook outputs),
which catches any drift between the notebook feature code and src/features.py.
"""
import numpy as np
import pandas as pd
import pytest

from Agent.config import REPO_ROOT, load_config, resolve

CSV = REPO_ROOT / "artifacts/model/check_vs_RUL_FD001_per_engine_errors.csv"
cfg = load_config()
HAVE = (resolve(cfg["model"]["model_dir"]) / "xgboost.joblib").exists() and (resolve(cfg["data"]["raw_dir"]) / cfg["data"]["test"]).exists() and CSV.exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="real data/models/artifacts not available")


def test_endpoint_predictions_match_saved_csv():
    from Agent.orchestration.orchestrator import AgentSystem
    from Agent.schemas.contracts import PredictRULInput, RequestContext
    sysm = AgentSystem.from_config(cfg)
    ref = pd.read_csv(CSV)
    diffs_x, diffs_r = [], []
    for _, row in ref.head(25).iterrows():
        e, last = int(row["engine_id"]), int(row["last_cycle"])
        ctx = RequestContext(request_id="r", engine_id=e, dataset_split="test", cutoff_cycle=last, question="x",
                             model_version=sysm.suite.model_version, corpus_version=sysm.suite.corpus_version)
        out = sysm.suite.call("predict_rul", ctx, PredictRULInput())
        diffs_x.append(abs(out.predicted_rul_cycles - row["XGBoost predicted"]))
        diffs_r.append(abs(out.support_rul_cycles - row["Random Forest predicted"]))
    assert max(diffs_x) < 0.5, f"XGBoost max abs diff {max(diffs_x):.3f}"
    assert max(diffs_r) < 0.5, f"RF max abs diff {max(diffs_r):.3f}"
