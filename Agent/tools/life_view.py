"""Retrospective life view for TRAIN engines: total life, the sensors that track progression to failure, and full-life charts.

This is deliberately NOT one of the cutoff-bound agent tools. It reads an engine's complete run to failure, which only train
engines have, so it is refused for the test split and is labelled retrospective wherever it is shown. Normal assessments, the
agent tools and ordinary chat answers keep the strict cutoff.

"Associated with progression to failure" is an observed rank correlation between a sensor and the cycle number. It is not a
statement that the sensor caused the degradation (sensor names and units are not verified in this project).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Agent.schemas.contracts import LifeChart, LifeSensorRow, LifeView
from Agent.tools.datastore import DataStore
from src.features import MODEL_SENSORS, build_features
from src.predictor import RULPredictor
from src.rul_models import predict_rul

EDGE_FRACTION = 0.2  # first / last 20% of the life are compared for the "late minus early" column
MIN_CYCLES = 10


@dataclass
class LifeData:
    summary: LifeView
    cycles: List[int]
    sensors: Dict[str, List[float]]
    actual_rul: List[int]
    history: pd.DataFrame


class LifeViewer:
    def __init__(self, store: DataStore, predictor: RULPredictor, faults: Optional[Dict[str, str]] = None):
        self.store, self.predictor = store, predictor
        self.faults = faults if faults is not None else {}  # shared with the tool suite so the outage demo also covers this view
        self._data: Dict[Tuple, LifeData] = {}
        self._pred: Dict[Tuple, List[float]] = {}

    def role(self, engine_id: int) -> str:
        return "train_in_sample" if engine_id in self.predictor.train_engine_ids else "validation_holdout"

    # ------------------------------------------------------------------ summary and sensor ranking
    def get(self, split: str, engine_id: int) -> LifeData:
        hist = self.store.full_history(split, engine_id)  # raises SelectionError unless the split is train
        key = (engine_id, self.predictor.model_version, self.store.data_checksum)
        if key in self._data:
            return self._data[key]
        cyc = hist["cycle"].to_numpy()
        n, total = len(cyc), int(cyc.max())
        notes: List[str] = []
        rows: List[LifeSensorRow] = []
        if n < MIN_CYCLES:
            notes.append(f"Only {n} cycles are recorded, too few to rank sensors.")
        else:
            k = max(3, int(round(n * EDGE_FRACTION)))
            cyc_s = pd.Series(cyc, index=hist.index, dtype=float)
            for s in MODEL_SENSORS:
                v = hist[s].astype(float)
                spread = float(v.std(ddof=0))
                if not np.isfinite(spread) or spread == 0:
                    continue
                rho = float(v.corr(cyc_s, method="spearman"))
                if not np.isfinite(rho):
                    continue
                late = float(v.iloc[-k:].mean() - v.iloc[:k].mean()) / spread
                rows.append(LifeSensorRow(sensor=s, direction="rises" if rho > 0 else "falls", spearman_rho=round(rho, 3),
                                          late_minus_early=round(late, 2)))
            rows.sort(key=lambda r: (-abs(r.spearman_rho), r.sensor))
            skipped = [s for s in MODEL_SENSORS if s not in {r.sensor for r in rows}]
            if skipped:
                notes.append(f"Constant over the full life, so not ranked: {', '.join(skipped)}.")
        summary = LifeView(dataset_split="train", engine_id=engine_id, total_cycles=total, data_role=self.role(engine_id),
                           model_version=self.predictor.model_version, ranking=rows, notes=notes)
        data = LifeData(summary=summary, cycles=[int(c) for c in cyc], sensors={s: [float(x) for x in hist[s]] for s in MODEL_SENSORS},
                        actual_rul=[int(x) for x in (total - cyc)], history=hist)
        self._data[key] = data
        return data

    # ------------------------------------------------------------------ model-predicted RUL at every cycle
    def predicted_rul(self, data: LifeData) -> Optional[List[float]]:
        """Primary-model RUL at each cycle. Features are causal, so row i equals a prediction made with cycles up to i only."""
        if self.faults.get("predict_rul") == "unavailable":
            return None
        key = (data.summary.engine_id, self.predictor.model_version, self.store.data_checksum)
        if key not in self._pred:
            try:
                out = predict_rul(self.predictor.primary.bundle, build_features(data.history))
            except Exception:
                return None
            self._pred[key] = [round(float(x), 2) for x in out]
        return self._pred[key]

    # ------------------------------------------------------------------ charts
    def charts(self, data: LifeData, sensors: List[str], with_rul: bool = True, normalise: bool = False) -> Tuple[List[LifeChart], List[str]]:
        v, notes, charts = data.summary, [], []
        if sensors:
            if len(sensors) == 1 and not normalise:
                s = sensors[0]
                charts.append(LifeChart(title=f"{s} over the full life of train engine {v.engine_id} (raw values, axis fitted to the data)", cycles=data.cycles,
                                        series={s: data.sensors[s]}, zero_baseline=False,
                                        caption="Raw sensor values on an axis fitted to this sensor's own range (it does not start at zero); units are not verified."))
            else:
                series: Dict[str, List[float]] = {}
                for s in sensors:
                    x = np.asarray(data.sensors[s], dtype=float)
                    sp = float(x.std())
                    series[s] = [float((a - x[0]) / sp) if sp > 0 else 0.0 for a in x]
                charts.append(LifeChart(title=f"Sensors over the full life of train engine {v.engine_id} (change since cycle 1)", cycles=data.cycles,
                                        series=series, zero_baseline=False, caption="Change since cycle 1 in units of each sensor's own spread (unitless), so sensors with different scales share one axis."))
        if with_rul:
            series2: Dict[str, Any] = {"Actual RUL": [float(x) for x in data.actual_rul]}
            caption = "Actual RUL is the engine's last cycle minus the current cycle (uncapped)."
            pred = self.predicted_rul(data)
            if pred is None:
                notes.append("The model-predicted RUL is unavailable right now, so only the actual RUL is drawn. No predicted value has been made up.")
            else:
                series2["Predicted RUL"] = pred
                caption += " Predicted RUL is the primary model's estimate at each cycle using only that cycle and earlier ones."
                if v.data_role == "train_in_sample":
                    caption += " This engine is part of the model's training data (in-sample), so the match is optimistic and not a fair test."
            charts.append(LifeChart(title=f"RUL over the full life of train engine {v.engine_id}: actual vs model prediction (cycles)",
                                    cycles=data.cycles, series=series2, caption=caption))
        return charts, notes
