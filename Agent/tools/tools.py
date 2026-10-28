"""The four agent tools. Cutoff, engine and version checks live HERE, not in prompts.

``ToolSuite.call`` receives the immutable ``RequestContext`` from the harness. The tool inputs
(``schemas.contracts``) have no engine, split or cutoff fields and forbid extra keys, so neither
the user text nor the model can override the context.
"""
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from Agent.retrieval.index import SearchIndex
from Agent.schemas.contracts import (
    ChartData, FleetRow, ModelMetadataInput, ModelMetadataOutput, PredictRULInput, PredictRULOutput, RequestContext,
    ScreenFleetInput, ScreenFleetOutput, SearchDocumentsInput, SearchDocumentsOutput, Passage, SensorSummary,
    SensorTrendsInput, SensorTrendsOutput,
)
from Agent.tools.datastore import DataStore, SelectionError
from src.features import MODEL_SENSORS
from src.predictor import RULPredictor


class ToolError(RuntimeError):
    """A tool could not produce a valid result (the harness records it and the agents degrade)."""


class ToolSuite:
    def __init__(self, predictor: RULPredictor, store: DataStore, index: SearchIndex, cfg: Dict[str, Any], metrics: Dict[str, Any]):
        self.predictor, self.store, self.index, self.cfg, self.metrics = predictor, store, index, cfg, metrics
        self.th = cfg["thresholds"]
        self._cache: Dict[Tuple, PredictRULOutput] = {}
        self._fleet_cache: Dict[Tuple, List[Dict[str, Any]]] = {}
        self._train_ref: Optional[Dict[str, Any]] = None
        # fault injection for tests, evals and the dashboard demo: {"predict_rul": "unavailable"|"slow"|"mismatch", "search_documents": "unavailable"}
        self.faults: Dict[str, str] = {}
        self.fault_sleep_s = 30.0
        # optional hook called with (ctx, PredictRULOutput) after every successful prediction (used for Slack alerts); it can never break a prediction
        self.on_prediction: Optional[Callable[[RequestContext, PredictRULOutput], Any]] = None

    @property
    def model_version(self) -> str:
        return self.predictor.model_version

    @property
    def corpus_version(self) -> str:
        return self.index.corpus_version

    # ------------------------------------------------------------------ dispatch
    def call(self, tool: str, ctx: RequestContext, args: Any):
        if ctx.model_version != self.model_version:
            raise ToolError(f"Model version changed during the request ({ctx.model_version} vs {self.model_version}).")
        if ctx.corpus_version != self.corpus_version:
            raise ToolError("Document corpus changed during the request.")
        try:
            self.store.validate_selection(ctx.dataset_split, ctx.engine_id, ctx.cutoff_cycle)
        except SelectionError as e:
            raise ToolError(str(e))
        fn = {"predict_rul": self._predict_rul, "get_sensor_trends": self._trends,
              "get_model_metadata": self._metadata, "search_documents": self._search,
              "screen_fleet": self._screen_fleet}.get(tool)
        if fn is None:
            raise ToolError(f"Unknown tool '{tool}'.")
        return fn(ctx, args)

    @staticmethod
    def _ids(ctx: RequestContext) -> Dict[str, Any]:
        return dict(request_id=ctx.request_id, engine_id=ctx.engine_id, dataset_split=ctx.dataset_split,
                    cutoff_cycle=ctx.cutoff_cycle, model_version=ctx.model_version, corpus_version=ctx.corpus_version)

    def data_role(self, split: str, engine_id: int) -> str:
        if split == "test":
            return "official_test"
        return "train_in_sample" if engine_id in self.predictor.train_engine_ids else "validation_holdout"

    # ------------------------------------------------------------------ predict_rul
    def _predict_rul(self, ctx: RequestContext, args: PredictRULInput) -> PredictRULOutput:
        fault = self.faults.get("predict_rul")
        if fault == "unavailable":
            raise ToolError("Prediction service unavailable (injected fault).")
        if fault == "slow":
            time.sleep(self.fault_sleep_s)
        key = (ctx.dataset_split, ctx.engine_id, ctx.cutoff_cycle, self.store.data_checksum, self.predictor.primary.sha256[:12],
               self.predictor.support.sha256[:12], ctx.model_version)
        if key in self._cache:
            out = self._cache[key].model_copy(update={**self._ids(ctx), "cache_status": "hit"})
        else:
            hist = self.store.history(ctx.dataset_split, ctx.engine_id, ctx.cutoff_cycle)
            try:
                res = self.predictor.predict(hist, ctx.cutoff_cycle)
            except ValueError as e:
                raise ToolError(f"Prediction failed: {e}")
            gap = res.gap
            agreement = "close" if gap <= self.th["agreement_close_cycles"] else ("wide" if gap > self.th["agreement_wide_cycles"] else "moderate")
            band = "Normal" if res.primary_rul > self.th["band_normal_above"] else ("Watch" if res.primary_rul > self.th["band_watch_above"] else "Review")
            role = self.data_role(ctx.dataset_split, ctx.engine_id)
            warns: List[str] = []
            if res.n_observed < self.th["short_history_cycles"]:
                warns.append(f"Short history: only {res.n_observed} cycles observed. Several of the largest test endpoint errors were overestimates for engines with short histories, so treat this estimate with extra caution.")
            if role == "train_in_sample":
                warns.append("This engine is part of the training data of the saved models (in-sample), so the estimate is optimistic and not a fair test.")
            if agreement == "wide":
                warns.append(f"The two models disagree by {gap:.1f} cycles (uncalibrated agreement heuristic).")
            cyc = hist["cycle"].to_numpy()
            if len(cyc) > 1 and (np.diff(cyc) > 1).any():
                warns.append("Cycle gaps in the observed history; rolling windows count observations, not cycles.")
            out = PredictRULOutput(
                **self._ids(ctx), observed_cycles=res.n_observed, predicted_rul_cycles=res.primary_rul,
                support_rul_cycles=res.support_rul, support_model_version=res.support_version,
                model_agreement_gap_cycles=gap, model_agreement=agreement, demo_band=band,
                feature_schema_version=res.feature_schema_version, data_role=role, warnings=warns,
                cache_status="miss", feature_vector=res.feature_row)
            self._cache[key] = out
        if self.on_prediction is not None:
            try:
                self.on_prediction(ctx, out)
            except Exception:
                pass  # an alerting problem must never break or change a prediction
        if fault == "mismatch":  # simulate a result for the wrong engine; the harness must reject it
            out = out.model_copy(update={"engine_id": ctx.engine_id + 1})
        return out

    # ------------------------------------------------------------------ get_sensor_trends
    def _trends(self, ctx: RequestContext, args: SensorTrendsInput) -> SensorTrendsOutput:
        hist = self.store.history(ctx.dataset_split, ctx.engine_id, ctx.cutoff_cycle)
        n = len(hist)
        warns: List[str] = []
        if args.sensors:
            bad = [s for s in args.sensors if s not in MODEL_SENSORS]
            if bad:
                raise ToolError(f"Unsupported sensor(s) {bad}. Allowed: {MODEL_SENSORS}")
        window = min(args.window, n)
        if window < args.window:
            warns.append(f"Only {n} cycles observed; the window was reduced to {n}.")
        cyc = hist["cycle"].to_numpy()
        if n > 1 and (np.diff(cyc) > 1).any():
            warns.append("Cycle gaps detected in the observed history.")
        stats: Dict[str, SensorSummary] = {}
        score: Dict[str, float] = {}
        spreads: Dict[str, float] = {}
        for s in MODEL_SENSORS:
            v = hist[s].to_numpy(float)
            w = v[-window:]
            slope: Optional[float] = float(np.polyfit(cyc[-window:], w, 1)[0]) if window >= 3 and np.ptp(cyc[-window:]) > 0 else None
            spread = float(np.std(v))
            change = float(w[-1] - w[0])
            rw = max(5, window // 3) if window >= 10 else None
            recent_slope: Optional[float] = (float(np.polyfit(cyc[-rw:], v[-rw:], 1)[0]) if rw and np.ptp(cyc[-rw:]) > 0 else None)
            trend = abs(slope) * (window - 1) if slope is not None else abs(change)
            score[s] = 0.0 if spread == 0 else trend / spread
            spreads[s] = spread
            stats[s] = SensorSummary(sensor=s, first=float(w[0]), last=float(w[-1]), min=float(w.min()), max=float(w.max()),
                                     window_mean=float(w.mean()), change_over_window=change, slope_per_cycle=slope,
                                     recent_slope_per_cycle=recent_slope, recent_window=rw if recent_slope is not None else None,
                                     normalised_trend=round(score[s], 4), evidence_id=f"trend:{s}")
            if spread == 0:
                warns.append(f"{s} is constant over the observed history.")
        if args.sensors:
            chosen, rule = list(dict.fromkeys(args.sensors)), "sensors requested by the caller"
        else:
            chosen = sorted(MODEL_SENSORS, key=lambda s: (-score[s], s))[:args.top_n]
            count = {4: "four", 6: "six"}.get(args.top_n, str(args.top_n))
            rule = f"{count} sensors with the largest least-squares trend over the window relative to the engine's own spread (ranks movement, not model importance)"
        first_row = max(0, n - args.chart_last_n) if args.chart_last_n else 0
        step = max(1, int(np.ceil((n - first_row) / 300)))
        idx = list(range(first_row, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        if args.chart_mode == "change":  # unitless: change since the first plotted cycle, in units of the sensor's own spread
            series = {s: [((float(hist[s].iloc[i]) - float(hist[s].iloc[first_row])) / spreads[s]) if spreads[s] > 0 else 0.0 for i in idx] for s in chosen}
        else:
            series = {s: [float(hist[s].iloc[i]) for i in idx] for s in chosen}
        chart = ChartData(cycles=[int(cyc[i]) for i in idx], series=series, mode=args.chart_mode)
        return SensorTrendsOutput(**self._ids(ctx), observed_cycles=n, window_used=window, selection_rule=rule,
                                  sensors=[stats[s] for s in chosen], chart=chart, quality_warnings=warns,
                                  data_quality=self._data_quality(hist), range_check=self._range_check(hist))

    # ------------------------------------------------------------------ data quality and training-range checks
    @staticmethod
    def _data_quality(hist) -> Dict[str, Any]:
        """Measured checks on the observed rows only. A successful prediction is not evidence of data quality."""
        cyc = hist["cycle"].to_numpy()
        gaps = np.diff(cyc) - 1 if len(cyc) > 1 else np.array([], dtype=int)
        value_cols = [c for c in hist.columns if c.startswith("sensor_") or c.startswith("setting_")]
        return {"rows_checked": int(len(hist)), "first_cycle": int(cyc[0]), "last_cycle": int(cyc[-1]),
                "missing_values": int(hist[value_cols].isna().sum().sum()),
                "duplicated_cycles": int(hist["cycle"].duplicated().sum()),
                "cycle_order_violations": int((np.diff(cyc) <= 0).sum()) if len(cyc) > 1 else 0,
                "cycle_gaps": int((gaps > 0).sum()), "missing_cycles_in_gaps": int(gaps[gaps > 0].sum()),
                "constant_model_sensors": [s for s in MODEL_SENSORS if float(np.ptp(hist[s].to_numpy(float))) == 0.0] if len(hist) > 1 else []}

    def _training_reference(self) -> Optional[Dict[str, Any]]:
        """Per-channel min/max over the rows of the engines the saved models were trained on."""
        if self._train_ref is not None:
            return self._train_ref or None
        try:
            tr = self.store.frame("train")
            ids = self.predictor.train_engine_ids
            tr = tr[tr["engine_id"].isin(ids)] if ids else tr
            cols = MODEL_SENSORS + ["setting_1", "setting_2", "setting_3"]
            self._train_ref = {"n_engines": int(tr["engine_id"].nunique()), "n_rows": int(len(tr)),
                               "min": {c: float(tr[c].min()) for c in cols}, "max": {c: float(tr[c].max()) for c in cols}}
        except Exception:  # reference unavailable: reported as such, never guessed
            self._train_ref = {}
        return self._train_ref or None

    def _range_check(self, hist) -> Dict[str, Any]:
        ref = self._training_reference()
        if ref is None:
            return {"available": False, "reason": "training reference statistics could not be computed"}
        last = hist.iloc[-1]
        out_sensors, settings = [], {}
        for s in MODEL_SENSORS:
            lo, hi, v = ref["min"][s], ref["max"][s], float(last[s])
            n_out = int(((hist[s] < lo) | (hist[s] > hi)).sum())
            if v < lo or v > hi or n_out:
                out_sensors.append({"sensor": s, "latest": v, "training_min": lo, "training_max": hi,
                                    "latest_outside": bool(v < lo or v > hi), "observed_rows_outside": n_out})
        for c in ("setting_1", "setting_2", "setting_3"):
            lo, hi, v = ref["min"][c], ref["max"][c], float(last[c])
            settings[c] = {"latest": v, "training_min": lo, "training_max": hi, "within_training_range": bool(lo <= v <= hi)}
        return {"available": True, "reference": f"min and max of observed rows of the {ref['n_engines']} training engines the saved models were fitted on",
                "n_reference_rows": ref["n_rows"], "sensors_checked": len(MODEL_SENSORS), "sensors_with_values_outside": out_sensors,
                "latest_cycle_sensors_outside": sum(1 for o in out_sensors if o["latest_outside"]), "settings": settings}

    # ------------------------------------------------------------------ screen_fleet
    def _screen_fleet(self, ctx: RequestContext, args: ScreenFleetInput) -> ScreenFleetOutput:
        if ctx.dataset_split != "test":
            raise ToolError("Fleet screening uses each engine's latest observed cycle and is offered for the test split only; "
                            "training engines run to failure, so their last cycle is not an observation made before failure.")
        key = (ctx.dataset_split, self.store.data_checksum, self.predictor.primary.sha256[:12], self.predictor.support.sha256[:12])
        if key not in self._fleet_cache:
            rows: List[Dict[str, Any]] = []
            for eid in self.store.engine_ids(ctx.dataset_split):
                last = self.store.max_cycle(ctx.dataset_split, eid)
                hist = self.store.history(ctx.dataset_split, eid, last)
                try:
                    res = self.predictor.predict(hist, last)
                except ValueError as e:
                    rows.append({"engine_id": eid, "observed_cycle": last, "error": str(e)})
                    continue
                flags: List[str] = []
                if res.n_observed < self.th["short_history_cycles"]:
                    flags.append(f"short history ({res.n_observed} cycles)")
                if res.gap > self.th["agreement_wide_cycles"]:
                    flags.append("models disagree widely")
                cyc = hist["cycle"].to_numpy()
                if len(cyc) > 1 and (np.diff(cyc) > 1).any():
                    flags.append("cycle gaps")
                if self.data_role(ctx.dataset_split, eid) == "train_in_sample":
                    flags.append("in-sample")
                band = "Normal" if res.primary_rul > self.th["band_normal_above"] else ("Watch" if res.primary_rul > self.th["band_watch_above"] else "Review")
                rows.append({"engine_id": eid, "observed_cycle": last, "rul": res.primary_rul, "band": band, "flags": flags})
            self._fleet_cache[key] = rows
        ok = sorted((r for r in self._fleet_cache[key] if "rul" in r), key=lambda r: (r["rul"], r["engine_id"]))
        below = [r for r in ok if args.threshold_cycles is not None and r["rul"] < args.threshold_cycles]
        shown = below[:args.top_n] if args.threshold_cycles is not None else ok[:args.top_n]
        if args.threshold_cycles is not None and not shown:
            shown = []
        out_rows = [FleetRow(rank=i + 1, engine_id=r["engine_id"], dataset_split=ctx.dataset_split, observed_cycle=r["observed_cycle"],
                             predicted_rul_cycles=r["rul"], demo_band=r["band"], flags=r["flags"]) for i, r in enumerate(shown)]
        return ScreenFleetOutput(**self._ids(ctx), n_engines=len(ok), threshold_cycles=args.threshold_cycles,
                                 n_below_threshold=len(below) if args.threshold_cycles is not None else None, rows=out_rows,
                                 ranking_rule="ascending predicted RUL (XGBoost) at each engine's own latest observed cycle; ties broken by engine id")

    # ------------------------------------------------------------------ get_model_metadata
    def _metadata(self, ctx: RequestContext, args: ModelMetadataInput) -> ModelMetadataOutput:
        meta = dict(self.metrics)
        meta["serving"] = {"primary_model_version": self.predictor.model_version, "support_model_version": self.predictor.support_version,
                           "feature_schema_version": self.predictor.primary.bundle.get("feature_set", "extended"),
                           "n_features": len(self.predictor.primary.bundle["features"]),
                           "n_training_engines_in_bundle": len(self.predictor.train_engine_ids)}
        meta.pop("note", None)
        meta.pop("source_files", None)
        return ModelMetadataOutput(**self._ids(ctx), metadata=meta)

    # ------------------------------------------------------------------ search_documents
    def _search(self, ctx: RequestContext, args: SearchDocumentsInput) -> SearchDocumentsOutput:
        if self.faults.get("search_documents") == "unavailable":
            raise ToolError("Document index unavailable (injected fault).")
        hits = self.index.search(args.query, args.top_k)
        return SearchDocumentsOutput(
            **self._ids(ctx), query=args.query,
            results=[Passage(chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title, section=c.section, revision=c.revision, text=c.text, score=round(s, 4))
                     for c, s in hits])
