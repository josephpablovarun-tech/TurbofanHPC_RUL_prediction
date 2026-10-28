"""Features, predictor, tools and retrieval (FRD: causal cutoff, versioned models, evidence tools)."""
import numpy as np
import pytest

from Agent.schemas.contracts import ModelMetadataInput, PredictRULInput, RequestContext, SearchDocumentsInput, SensorTrendsInput
from Agent.tools.tools import ToolError
from src.features import build_features, extended_feature_columns, feature_row_at_cutoff
from src.predictor import RULPredictor


def ctx_for(system, engine=24, split="test", cutoff=140, q="x"):
    return RequestContext(request_id="t1", engine_id=engine, dataset_split=split, cutoff_cycle=cutoff, question=q,
                          model_version=system.suite.model_version, corpus_version=system.suite.corpus_version)


def test_feature_count_and_no_leakage(fixture_dirs):
    tr = fixture_dirs["train"]
    e = tr[tr.engine_id == 5].reset_index(drop=True)
    assert len(extended_feature_columns()) == 97
    full = build_features(e)
    part = build_features(e[e.cycle <= 100])
    cols = extended_feature_columns()
    np.testing.assert_allclose(full.loc[full.cycle <= 100, cols].to_numpy(float), part[cols].to_numpy(float), rtol=1e-9, atol=1e-9)


def test_cutoff_truncation_ignores_future(fixture_dirs):
    tr = fixture_dirs["train"]
    e = tr[tr.engine_id == 5].copy()
    a = feature_row_at_cutoff(e, 100)
    e.loc[e.cycle > 100, "sensor_2"] += 1000
    b = feature_row_at_cutoff(e, 100)
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float))


def test_feature_row_validation(fixture_dirs):
    tr = fixture_dirs["train"]
    with pytest.raises(ValueError):
        feature_row_at_cutoff(tr[tr.engine_id <= 2], 10)  # two engines
    with pytest.raises(ValueError):
        feature_row_at_cutoff(tr[tr.engine_id == 1], 10_000)  # unobserved cutoff


def test_predictor_loads_and_predicts(fixture_dirs):
    p = RULPredictor.from_dir(fixture_dirs["models"])
    assert p.model_version.startswith("xgboost-") and p.support_version.startswith("random_forest-")
    tr = fixture_dirs["train"]
    r = p.predict(tr[tr.engine_id == 3], 80)
    assert r.primary_rul >= 0 and r.support_rul >= 0 and r.n_observed == 80


def test_tool_rejects_bad_context(system):
    c = ctx_for(system, cutoff=10_000)
    with pytest.raises(ToolError):
        system.suite.call("predict_rul", c, PredictRULInput())
    with pytest.raises(ToolError):
        system.suite.call("predict_rul", ctx_for(system, engine=999), PredictRULInput())


def test_predict_cache_and_fields(system):
    c = ctx_for(system)
    a = system.suite.call("predict_rul", c, PredictRULInput())
    b = system.suite.call("predict_rul", c, PredictRULInput())
    assert a.cache_status == "miss" and b.cache_status == "hit"
    assert a.predicted_rul_cycles == b.predicted_rul_cycles
    assert a.data_role == "official_test" and a.engine_id == 24


def test_short_history_warning(system):
    out = system.suite.call("predict_rul", ctx_for(system, engine=1, cutoff=31), PredictRULInput())
    assert any("Short history" in w for w in out.warnings)


def test_trends_allowlist_and_window(system):
    c = ctx_for(system)
    with pytest.raises(ToolError):
        system.suite.call("get_sensor_trends", c, SensorTrendsInput(sensors=["sensor_1"]))
    out = system.suite.call("get_sensor_trends", c, SensorTrendsInput(window=30))
    assert len(out.sensors) == 4 and max(out.chart.cycles) <= 140
    only = system.suite.call("get_sensor_trends", c, SensorTrendsInput(sensors=["sensor_2"], window=10))
    assert [s.sensor for s in only.sensors] == ["sensor_2"]


def test_trend_chart_modes(system):
    c = ctx_for(system, cutoff=100)
    raw = system.suite.call("get_sensor_trends", c, SensorTrendsInput())
    chg = system.suite.call("get_sensor_trends", c, SensorTrendsInput(chart_mode="change", top_n=6))
    assert raw.chart.mode == "raw" and len(raw.sensors) == 4
    assert chg.chart.mode == "change" and len(chg.sensors) == 6 and len(chg.chart.series) == 6
    assert all(v[0] == 0.0 for v in chg.chart.series.values())  # change is measured from the first plotted cycle


def test_trends_never_read_beyond_cutoff(system):
    out = system.suite.call("get_sensor_trends", ctx_for(system, cutoff=100), SensorTrendsInput())
    assert max(out.chart.cycles) == 100


def test_metadata_and_version_guard(system):
    out = system.suite.call("get_model_metadata", ctx_for(system), ModelMetadataInput())
    assert "limitations" in out.metadata and out.metadata["serving"]["primary_model_version"] == system.suite.model_version
    bad = ctx_for(system).model_copy(update={"model_version": "other-123"})
    with pytest.raises(ToolError):
        system.suite.call("get_model_metadata", bad, ModelMetadataInput())


@pytest.mark.parametrize("query,gold", [
    ("what is the FD001 target and how is RUL defined", {"DG-05", "DG-06", "DG-01"}),
    ("known limitations of the model", {"MC-06"}),
    ("which sensors does the model use", {"SG-02", "DG-08"}),
    ("sensor units physical measurement", {"SG-01"}),
])
def test_retrieval_gold(system, query, gold):
    out = system.suite.call("search_documents", ctx_for(system), SearchDocumentsInput(query=query, top_k=4))
    assert gold & {p.chunk_id for p in out.results}


def test_search_fault(system):
    system.suite.faults["search_documents"] = "unavailable"
    with pytest.raises(ToolError):
        system.suite.call("search_documents", ctx_for(system), SearchDocumentsInput(query="limits"))
