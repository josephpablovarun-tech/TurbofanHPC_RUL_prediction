"""80%-of-life Slack alert: decision logic, once-per-crossing behaviour, safety, and the prediction hook (no test ever posts to Slack)."""
import io
import json
import urllib.request

import pytest

from Agent.config import load_config
from Agent.schemas.contracts import PredictRULInput, RequestContext
from Agent.tools.alerts import AlertError, AlertService, SlackWebhookSender, life_used


class FakeSender:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send(self, text):
        if self.fail:
            raise AlertError("boom")
        self.sent.append(text)


def svc(tmp_path, sender=None, **kw):
    return AlertService(0.80, 0.75, sender, tmp_path / "state.json", tmp_path / "alerts.jsonl", **kw)


def ev(s, cycle, pred, engine=1, split="train", version="m1", role="train_in_sample"):
    return s.evaluate(split, engine, cycle, pred, pred + 2.0, "Watch", version, role)


def test_life_used_math():
    assert life_used(80, 20) == pytest.approx(0.8) and life_used(160, 31.1) == pytest.approx(0.8373, abs=1e-3) and life_used(0, 0) == 0.0


def test_below_threshold_does_not_alert(tmp_path):
    sender = FakeSender()
    assert ev(svc(tmp_path, sender), 100, 100).status == "below_threshold" and sender.sent == []


def test_alerts_once_per_crossing(tmp_path):
    sender = FakeSender()
    s = svc(tmp_path, sender)
    assert ev(s, 160, 31.1).status == "alerted"
    assert ev(s, 165, 28.0).status == "already_alerted" and ev(s, 179, 16.7).status == "already_alerted"
    assert len(sender.sent) == 1


def test_hysteresis_and_rearm(tmp_path):
    sender = FakeSender()
    s = svc(tmp_path, sender)
    ev(s, 160, 31.1)                                   # 84%: alert
    assert ev(s, 154, 46).status == "below_threshold"  # 77%: inside the dead band, stays armed-off (no new alert)
    assert ev(s, 160, 31.1).status == "already_alerted" and len(sender.sent) == 1
    assert ev(s, 150, 55).status == "below_threshold"  # 73%: below rearm_below, re-arms
    assert ev(s, 160, 31.1).status == "alerted" and len(sender.sent) == 2


def test_engines_and_splits_are_independent(tmp_path):
    sender = FakeSender()
    s = svc(tmp_path, sender)
    for engine, split in ((1, "train"), (2, "train"), (1, "test")):
        assert ev(s, 160, 31.1, engine=engine, split=split).status == "alerted"
    assert len(sender.sent) == 3


def test_failed_send_is_retried_not_marked(tmp_path):
    s = svc(tmp_path, FakeSender(fail=True))
    e = ev(s, 160, 31.1)
    assert e.status == "send_failed" and "boom" in e.error
    good = FakeSender()
    s.sender = good
    assert ev(s, 160, 31.1).status == "alerted" and len(good.sent) == 1


def test_dry_run_decides_but_sends_nothing(tmp_path):
    s = svc(tmp_path, None)
    e = ev(s, 160, 31.1)
    assert e.status == "dry_run" and e.message and s.mode == "dry-run" and "DRY-RUN" in s.status_text()
    assert ev(s, 165, 28).status == "already_alerted"
    assert not (tmp_path / "state.json").exists()  # a dry run never marks an engine as really alerted


def test_state_survives_a_restart(tmp_path):
    sender = FakeSender()
    ev(svc(tmp_path, sender), 160, 31.1)
    again = FakeSender()
    assert ev(svc(tmp_path, again), 165, 28).status == "already_alerted" and again.sent == []


def test_message_contents_and_no_ground_truth(tmp_path):
    sender = FakeSender()
    ev(svc(tmp_path, sender), 160, 31.1, version="xgboost-abc")
    text = sender.sent[0]
    for part in ("train engine 1", "84%", "cycle 160", "31.1", "xgboost-abc", "not a maintenance decision", "training data"):
        assert part in text, part
    assert "true" not in text.lower()  # the label / true RUL is never part of an alert


def test_event_log_has_no_secret(tmp_path):
    ev(svc(tmp_path, FakeSender()), 160, 31.1)
    rec = json.loads((tmp_path / "alerts.jsonl").read_text().splitlines()[0])
    assert rec["status"] == "alerted" and "hooks.slack.com" not in json.dumps(rec)


def test_webhook_sender_validation_and_secret_hygiene(monkeypatch):
    with pytest.raises(ValueError):
        SlackWebhookSender("https://example.com/hook")
    secret = "https://hooks.slack.com/services/T000/B000/SECRETSECRET"

    def boom(*a, **k):
        raise OSError("cannot reach " + secret)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(AlertError) as e:
        SlackWebhookSender(secret).send("x")
    assert "SECRETSECRET" not in str(e.value) and "hooks.slack.com" not in str(e.value)


def test_webhook_sender_accepts_ok_and_rejects_other(monkeypatch):
    class Resp(io.BytesIO):
        def __init__(self, body, status=200):
            super().__init__(body)
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    secret = "https://hooks.slack.com/services/T000/B000/XXXX"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Resp(b"ok"))
    SlackWebhookSender(secret).send("hello")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Resp(b"invalid_payload", 400))
    with pytest.raises(AlertError):
        SlackWebhookSender(secret).send("hello")


def test_from_config_modes(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["logging"] = {"trace_file": str(tmp_path / "t.jsonl")}
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert AlertService.from_config(cfg).mode == "dry-run"
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/not-slack")
    bad = AlertService.from_config(cfg)
    assert bad.mode == "dry-run" and "https://hooks.slack.com/" in bad.status_text()
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T000/B000/XXXX")
    ok = AlertService.from_config(cfg)
    assert ok.mode == "slack" and "hooks.slack.com" not in ok.status_text()  # the URL itself is never shown


def test_tests_and_evals_have_alerts_off(cfg):
    assert cfg["alerts"]["enabled"] is False  # the shared test config must never post to Slack
    assert load_config()["alerts"]["life_used_threshold"] == 0.80


def test_prediction_hook_alerts_once_and_never_breaks_a_prediction(system, tmp_path):
    sender = FakeSender()
    service = AlertService(0.01, 0.005, sender, tmp_path / "s.json", tmp_path / "a.jsonl")  # threshold so low that any prediction crosses it
    system.suite.on_prediction = service.on_prediction
    ctx = RequestContext(request_id="r1", engine_id=5, dataset_split="train", cutoff_cycle=100, question="", model_version=system.suite.model_version,
                         corpus_version=system.suite.corpus_version)
    first = system.suite.call("predict_rul", ctx, PredictRULInput())
    system.suite.call("predict_rul", ctx, PredictRULInput())  # a cache hit still notifies, but dedupes
    assert len(sender.sent) == 1 and service.last_event.status in ("already_alerted", "alerted")
    system.suite.on_prediction = lambda c, o: 1 / 0  # a failing hook must not change or block the prediction
    assert system.suite.call("predict_rul", ctx, PredictRULInput()).predicted_rul_cycles == first.predicted_rul_cycles
