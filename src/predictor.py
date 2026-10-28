"""Load the saved XGBoost (primary) and Random Forest (support) bundles and predict RUL
from a raw engine history.

Bundles are pickles: only load files you trust. ``from_dir`` loads fixed file names from one
directory and records each file's SHA-256 so a prediction can be traced to the exact file
(FRD ML03 style immutable version, until an MLflow registry exists).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet

import joblib

from src.features import FEATURE_SCHEMA_VERSION, extended_feature_columns, feature_row_at_cutoff
from src.rul_models import predict_rul

BUNDLE_FILES = {"xgboost": "xgboost.joblib", "random_forest": "random_forest.joblib"}


class PredictorError(RuntimeError):
    """Raised when a bundle cannot be used (missing file, feature schema mismatch)."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class LoadedModel:
    name: str
    bundle: Dict[str, Any]
    sha256: str

    @property
    def version(self) -> str:
        return f"{self.name}-{self.sha256[:12]}"


@dataclass
class PredictionResult:
    primary_rul: float
    support_rul: float
    gap: float
    n_observed: int
    cutoff_cycle: int
    feature_row: Dict[str, float]
    primary_version: str
    support_version: str
    feature_schema_version: str


def _load(model_dir: Path, name: str) -> LoadedModel:
    path = model_dir / BUNDLE_FILES[name]
    if not path.exists():
        raise PredictorError(f"model file not found: {path}")
    bundle = joblib.load(path)
    expected = extended_feature_columns()
    if list(bundle.get("features", [])) != expected:
        raise PredictorError(f"{path.name}: feature columns differ from src.features.extended_feature_columns()")
    return LoadedModel(name=name, bundle=bundle, sha256=file_sha256(path))


class RULPredictor:
    def __init__(self, primary: LoadedModel, support: LoadedModel):
        self.primary, self.support = primary, support

    @classmethod
    def from_dir(cls, model_dir, primary: str = "xgboost", support: str = "random_forest") -> "RULPredictor":
        model_dir = Path(model_dir)
        return cls(_load(model_dir, primary), _load(model_dir, support))

    @property
    def model_version(self) -> str:
        return self.primary.version

    @property
    def support_version(self) -> str:
        return self.support.version

    @property
    def train_engine_ids(self) -> FrozenSet[int]:
        return frozenset(int(i) for i in self.primary.bundle.get("train_engine_ids", []))

    def predict(self, history, cutoff_cycle: int) -> PredictionResult:
        """RUL (cycles) at ``cutoff_cycle`` using only rows up to that cycle."""
        row = feature_row_at_cutoff(history, int(cutoff_cycle))
        p = float(predict_rul(self.primary.bundle, row)[0])
        s = float(predict_rul(self.support.bundle, row)[0])
        feats = {c: float(row[c].iloc[0]) for c in self.primary.bundle["features"]}
        n_obs = int((history["cycle"] <= cutoff_cycle).sum())
        return PredictionResult(
            primary_rul=p, support_rul=s, gap=abs(p - s), n_observed=n_obs, cutoff_cycle=int(cutoff_cycle),
            feature_row=feats, primary_version=self.primary.version, support_version=self.support.version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )
