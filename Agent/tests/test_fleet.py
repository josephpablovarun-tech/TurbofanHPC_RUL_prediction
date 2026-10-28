"""Fleet scoring: the batch simulator, import checks, batch scoring, labels and the one-message Slack summary."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing import v1 as st_testing

from Agent.tools import fleet_upload
from Agent.tools.alerts import AlertService
from Agent.tools.fleet_upload import UploadError, parse_labels, parse_upload, score_fleet
from pipelines.simulate_fleet import simulate_batch
from src.schema import RAW_COLUMNS


class FakeSender:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)


@pytest.fixture()
def batch(system):
    store = system.suite.store
    return simulate_batch(store.frame("train"), store.frame("test"), store.rul_labels(), 20, 5, (5, 60), 0.4)


def test_simulated_batch_is_well_formed_and_reproducible(system, batch):
    df, truth = batch
    assert list(df.columns) == RAW_COLUMNS and np.isfinite(df.to_numpy(float)).all()
    g = df.groupby("engine_id")["cycle"]
    assert sorted(df["engine_id"].unique()) == list(range(1, 21)) and (g.min() == 1).all() and (g.max() == g.size()).all()
    assert (truth["last_cycle"].to_numpy() == g.max().to_numpy()).all() and (truth["last_cycle"] >= 31).all() and (truth["true_rul"] >= 5).all()
    assert (truth["last_cycle"] + truth["true_rul"] == truth["total_life"]).all()
    store = system.suite.store
    again, _ = simulate_batch(store.frame("train"), store.frame("test"), store.rul_labels(), 20, 5, (5, 60), 0.4)
    other, _ = simulate_batch(store.frame("train"), store.frame("test"), store.rul_labels(), 20, 6, (5, 60), 0.4)
    assert df.equals(again) and not df.equals(other)
    row = truth[truth["source"] == "train"].iloc[0]  # noise was added: a simulated engine is not an exact copy of its source
    train = store.frame("train")
    src = train[(train["engine_id"] == int(row["source_engine"])) & (train["cycle"] == 1)]["sensor_2"].iloc[0]
    assert df[(df["engine_id"] == int(row["engine_id"])) & (df["cycle"] == 1)]["sensor_2"].iloc[0] != src


def test_parse_upload_accepts_the_three_layouts(batch):
    df, _ = batch
    for blob in (df.to_csv(index=False), df.to_csv(index=False, header=False), df.to_csv(index=False, header=False, sep=" ")):
        out, warnings = parse_upload(blob.encode(), "x")
        assert list(out.columns) == RAW_COLUMNS and len(out) == len(df) and out["engine_id"].nunique() == 20 and warnings == []
    out, warnings = parse_upload((df.assign(note="x").to_csv(index=False)).encode(), "x")
    assert "note" in warnings[0] and list(out.columns) == RAW_COLUMNS


def test_parse_upload_rejects_the_file_with_a_clear_message(batch, monkeypatch):
    df, _ = batch
    with pytest.raises(UploadError, match="sensor_3"):
        parse_upload(df.drop(columns=["sensor_3"]).to_csv(index=False).encode(), "x")
    with pytest.raises(UploadError, match="expected 26 columns"):
        parse_upload(df.iloc[:, :25].to_csv(index=False, header=False, sep=" ").encode(), "x")
    with pytest.raises(UploadError, match="MB"):
        parse_upload(b"1 " * 6_000_000, "x")
    with pytest.raises(UploadError, match="empty"):
        parse_upload(b"  \n", "x")
    with pytest.raises(UploadError, match="whole numbers"):
        parse_upload(df.assign(engine_id=df["engine_id"] + 0.5).to_csv(index=False).encode(), "x")
    monkeypatch.setattr(fleet_upload, "MAX_ROWS", 10)
    with pytest.raises(UploadError, match="limit"):
        parse_upload(df.to_csv(index=False).encode(), "x")


def test_score_fleet_scores_every_engine(system, batch):
    df, _ = batch
    th = system.cfg["thresholds"]
    res = score_fleet(df, system.suite.predictor, th, 0.8)
    t = res.table
    assert len(t) == 20 and res.rejected == [] and res.n_engines == 20 and res.model_version == system.suite.model_version
    assert t["life_used_pct"].is_monotonic_decreasing and (t["predicted_rul"] >= 0).all() and np.isfinite(t["predicted_rul"]).all()
    assert t.loc[t["flagged"], "life_used_pct"].min() >= 79.95 if t["flagged"].any() else True
    assert t.loc[~t["flagged"], "life_used_pct"].max() < 80.05 if (~t["flagged"]).any() else True
    assert res.flagged == int(t["flagged"].sum())
    for _, r in t.iterrows():
        if min(abs(r["predicted_rul"] - th["band_normal_above"]), abs(r["predicted_rul"] - th["band_watch_above"])) < 0.1:
            continue  # the table rounds the RUL to 1 decimal, so skip values sitting on a band boundary
        assert r["band"] == ("Normal" if r["predicted_rul"] > th["band_normal_above"] else ("Watch" if r["predicted_rul"] > th["band_watch_above"] else "Review"))


def test_one_bad_engine_does_not_stop_the_others(system, batch):
    df, _ = batch
    df = df.copy()
    df.loc[(df["engine_id"] == 3).idxmax(), "sensor_4"] = np.nan                       # engine 3: a missing value
    dup = df[df["engine_id"] == 4].iloc[[5]]                                          # engine 4: a repeated cycle
    a, b = df.index[df["engine_id"] == 5][[2, 3]]                                     # engine 5: two rows out of order
    df.loc[a, "cycle"], df.loc[b, "cycle"] = df.loc[b, "cycle"], df.loc[a, "cycle"]
    df = pd.concat([df, dup], ignore_index=True)
    res = score_fleet(df, system.suite.predictor, system.cfg["thresholds"], 0.8)
    reasons = dict(res.rejected)
    assert sorted(reasons) == [3, 4, 5] and "non-finite" in reasons[3] and "duplicate" in reasons[4] and "increasing" in reasons[5]
    assert len(res.table) == 17 and res.n_engines == 20


def test_labels_give_an_evaluation_but_stay_out_of_the_alert(system, batch):
    df, truth = batch
    ids = list(range(1, 21))
    by_csv = parse_labels(truth.to_csv(index=False).encode(), ids)
    by_lines = parse_labels("\n".join(str(v) for v in truth["true_rul"]).encode(), ids)
    assert by_csv.tolist() == by_lines.tolist() == truth["true_rul"].astype(float).tolist()
    res = score_fleet(df, system.suite.predictor, system.cfg["thresholds"], 0.8, by_csv)
    ev = res.evaluation
    assert ev["n"] == 20 and np.isfinite(ev["mae"]) and ev["true_positive"] + ev["false_positive"] + ev["false_negative"] + ev["true_negative"] == 20
    t = res.table.sort_values("engine_id")
    assert np.allclose(t["true_life_used_pct"].to_numpy(), 100 * truth["true_life_used"].to_numpy(), atol=0.06)
    with pytest.raises(UploadError, match="one value per engine"):
        parse_labels(b"1\n2\n3", ids)
    with pytest.raises(UploadError, match="no label"):
        parse_labels(truth.iloc[:15].to_csv(index=False).encode(), ids)


def test_fleet_summary_is_one_message_once_per_file(tmp_path):
    sender = FakeSender()
    s = AlertService(0.8, 0.75, sender, tmp_path / "s.json", tmp_path / "a.jsonl")
    rows = [dict(engine_id=i, last_cycle=150 + i, life_used_pct=99.0 - i, predicted_rul=3.0 + i, band="Review") for i in range(1, 21)]
    assert s.send_fleet_summary("fp1", "batch3.csv", rows, 100, "m1").status == "alerted"
    assert s.send_fleet_summary("fp1", "batch3.csv", rows, 100, "m1").status == "already_alerted" and len(sender.sent) == 1
    assert s.send_fleet_summary("fp1", "batch3.csv", rows, 100, "m1", force=True).status == "alerted" and len(sender.sent) == 2
    assert s.send_fleet_summary("fp2", "batch2.csv", rows, 100, "m1").status == "alerted"  # a different file is a different scan
    text = sender.sent[0]
    assert "20 of 100 engines" in text and "Engine 1:" in text and "and 5 more" in text and "not a maintenance decision" in text and "true" not in text.lower()
    assert s.send_fleet_summary("fp3", "batch1.csv", [], 100, "m1").status == "below_threshold" and len(sender.sent) == 3  # nothing flagged, nothing sent
    dry = AlertService(0.8, 0.75, None, tmp_path / "d.json", tmp_path / "d.jsonl")
    ev = dry.send_fleet_summary("fp1", "batch3.csv", rows, 100, "m1")
    assert ev.status == "dry_run" and ev.message and not (tmp_path / "d.json").exists()


def test_fleet_view_scores_the_official_test_engines(cfg, monkeypatch):
    import Agent.config as c
    monkeypatch.setattr(c, "load_config", lambda path=None: cfg)
    at = st_testing.AppTest.from_file(str(c.AGENT_DIR / "app" / "app.py"), default_timeout=120).run()
    at.radio(key="view").set_value("Fleet scoring").run()
    assert not at.exception
    at.radio(key="fleet_source").set_value("Official test engines (100)").run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert int(m["Engines scored"]) + int(m["Rejected engines"]) == 100
    assert any("MAE" in x.label for x in at.metric)  # the official labels give an evaluation table
