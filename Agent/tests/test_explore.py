"""Data explorer analysis helpers and the official test RUL labels (evaluation display only)."""
from pathlib import Path

import numpy as np
import pytest

from Agent.tools import explore
from Agent.tools.datastore import DataStore, SelectionError
from src.schema import SENSOR_COLS


@pytest.fixture()
def train(system):
    return system.suite.store.frame("train")


def test_overview_matches_the_file(train):
    o = explore.overview(train)
    assert o["engines"] == 100 and o["rows"] == len(train) and 150 <= o["cycles_min"] and o["cycles_max"] <= 300
    assert o["missing_cells"] == 0 and o["duplicate_keys"] == 0
    assert o["constant_columns"] == ["setting_1", "setting_2", "setting_3"]  # the synthetic fixture holds the settings fixed


def test_rul_and_life_phase_histograms(train):
    t = explore.add_rul(train)
    e = t[t["engine_id"] == 1]
    assert e["rul"].iloc[-1] == 0 and e["rul"].iloc[0] == len(e) - 1
    ph = explore.phase_histograms(train, "sensor_4", 10)
    assert set(ph["phase"]) == {"early life (first 20%)", "late life (last 20%)"}
    assert np.allclose(ph.groupby("phase")["percent"].sum(), 100.0)


def test_early_late_shift_finds_the_drifting_sensors(train):
    shift = explore.early_late_shift(train, SENSOR_COLS)
    # fixture: sensor i drifts over the life unless i % 5 == 2 (noise only)
    assert all(int(c.split("_")[1]) % 5 != 2 for c in shift["column"].head(6))
    assert shift["shift (std units)"].abs().is_monotonic_decreasing


def test_correlations(train):
    c = explore.correlation_matrix(train, SENSOR_COLS)
    assert np.allclose(np.diag(c), 1.0)
    pairs = explore.correlation_pairs(c, 0.9)
    assert (pairs["correlation"].abs() >= 0.9).all()
    r = explore.rul_correlations(explore.add_rul(train), SENSOR_COLS)
    assert r["correlation with RUL"].abs().is_monotonic_decreasing and len(r) == 21


def test_quality_and_cycle_gaps(train):
    q = explore.quality_table(train)
    assert len(q) == 24 and q["missing"].sum() == 0 and bool(q.loc[q["column"] == "setting_1", "constant"].iloc[0])
    assert explore.cycle_gap_table(train).empty
    gappy = train[~((train["engine_id"] == 3) & train["cycle"].isin([10, 11]))]
    g = explore.cycle_gap_table(gappy)
    assert g["engine_id"].tolist() == [3] and g["missing_cycles"].tolist() == [2]


def test_trend_frame(train):
    f = explore.trend_frame(train, [1, 2], ["sensor_4", "sensor_11"], "rul", True)
    n = int(train["engine_id"].isin([1, 2]).sum())
    assert len(f) == 2 * n and set(f["engine"]) == {"1", "2"} and f["x"].min() == 0
    assert (explore.trend_frame(train, [1], ["setting_1"], standardise=True)["value"] == 0).all()  # a constant column standardises to zero, not NaN


def test_labels_join_and_bands(system, fixture_dirs):
    store = system.suite.store
    labels = store.rul_labels()
    raw = [int(x) for x in (fixture_dirs["raw"] / "RUL_FD001.txt").read_text().split()]
    assert labels.tolist() == raw and list(labels.index) == list(range(1, 101))
    j = explore.label_join(store.frame("test"), labels)
    assert (j["implied_total_life"] == j["observed_cycles"] + j["rul_label"]).all() and j["observed_fraction"].between(0, 1).all()
    assert sum(explore.band_counts(labels, 50, 20).values()) == 100


def test_true_rul_at(system):
    store = system.suite.store
    label, last = int(store.rul_labels().loc[3]), store.max_cycle("test", 3)
    assert store.true_rul_at("test", 3, last) == label and store.true_rul_at("test", 3, last - 10) == label + 10
    assert store.true_rul_at("train", 5, 50) == store.max_cycle("train", 5) - 50


def test_labels_absent_or_misaligned(fixture_dirs, tmp_path):
    raw = fixture_dirs["raw"]
    assert DataStore(raw, "train_FD001.txt", "test_FD001.txt").true_rul_at("test", 1, 1) is None  # no label file configured
    assert DataStore(raw, "train_FD001.txt", "test_FD001.txt", "nope.txt").true_rul_at("test", 1, 1) is None  # configured but missing
    short = tmp_path / "short.txt"
    short.write_text("10 \n20 \n")
    with pytest.raises(SelectionError):
        DataStore(raw, "train_FD001.txt", "test_FD001.txt", str(short)).rul_labels()  # 2 labels for 100 engines


def test_labels_are_not_reachable_from_the_agent():
    """Leakage guard: the agent tools, orchestration and chat answers never read the official labels."""
    root = Path(__file__).resolve().parents[1]
    for rel in ("tools/tools.py", "orchestration/agents.py", "orchestration/orchestrator.py", "orchestration/supervisor.py", "orchestration/synthesis.py"):
        text = (root / rel).read_text()
        assert "rul_labels" not in text and "true_rul_at" not in text, rel
