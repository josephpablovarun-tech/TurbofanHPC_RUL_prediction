"""Runs the 20-case evaluation set against the synthetic fixtures (wiring check, not model quality)."""
import json


def test_eval_suite_all_pass(cfg, tmp_path, monkeypatch):
    from Agent.evals import run as ev
    monkeypatch.setattr(ev, "load_config", lambda: cfg)
    s = ev.run(llm=False, out_dir=str(tmp_path))
    failed = [(r["id"], r.get("failed_checks") or r.get("error")) for r in s["rows"] if not r["passed"]]
    assert s["cases_run"] == 20 and not failed, failed
    assert (tmp_path / "report_deterministic.md").exists()
