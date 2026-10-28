"""Harness: allowlists, budget, timeout, deadline, context mismatch (FRD: enforced outside prompts)."""
import time

import pytest

from Agent.orchestration import harness as h
from Agent.orchestration.agents import DOC_AGENT, DOC_TOOLS, ENGINE_AGENT, ENGINE_TOOLS
from Agent.schemas.contracts import RequestContext


def runner(system, budget=5, timeout=20, deadline=120, cutoff=140, engine=24):
    ctx = RequestContext(request_id="h1", engine_id=engine, dataset_split="test", cutoff_cycle=cutoff, question="q",
                         model_version=system.suite.model_version, corpus_version=system.suite.corpus_version)
    return h.ToolRunner(system.suite, ctx, {ENGINE_AGENT: set(ENGINE_TOOLS), DOC_AGENT: set(DOC_TOOLS)}, budget, timeout, time.monotonic() + deadline)


def test_allowlist(system):
    r = runner(system)
    with pytest.raises(h.PermissionDenied):
        r.run(DOC_AGENT, "predict_rul", {})
    with pytest.raises(h.PermissionDenied):
        r.run(ENGINE_AGENT, "search_documents", {"query": "x"})
    assert r.executed == 0 and r.events[-1].status == "rejected"


@pytest.mark.parametrize("args", [{"engine_id": 3}, {"cutoff_cycle": 200}, {"dataset_split": "train"}])
def test_context_fields_rejected(system, args):
    r = runner(system)
    with pytest.raises(h.ArgumentRejected):
        r.run(ENGINE_AGENT, "predict_rul", args)


def test_budget(system):
    r = runner(system, budget=2)
    r.run(ENGINE_AGENT, "predict_rul", {})
    r.run(ENGINE_AGENT, "get_model_metadata", {})
    with pytest.raises(h.BudgetExceeded):
        r.run(ENGINE_AGENT, "get_sensor_trends", {})
    assert r.executed == 2 and r.events[-1].status == "budget_exceeded"


def test_timeout_and_late_result_ignored(system):
    system.suite.faults["predict_rul"] = "slow"
    system.suite.fault_sleep_s = 1.0
    r = runner(system, timeout=0.2)
    with pytest.raises(h.ToolTimeout):
        r.run(ENGINE_AGENT, "predict_rul", {})
    assert r.events[-1].status == "timeout"
    time.sleep(1.2)  # the late result must not appear anywhere
    assert [e.status for e in r.events] == ["timeout"]


def test_deadline(system):
    r = runner(system, deadline=-1)
    with pytest.raises(h.DeadlineExceeded):
        r.run(ENGINE_AGENT, "predict_rul", {})


def test_context_mismatch_rejected(system):
    system.suite.faults["predict_rul"] = "mismatch"
    r = runner(system)
    with pytest.raises(h.ContextMismatch):
        r.run(ENGINE_AGENT, "predict_rul", {})
    assert r.events[-1].status == "context_mismatch"


def test_cancel(system):
    r = runner(system)
    r.cancel()
    with pytest.raises(h.HarnessError):
        r.run(ENGINE_AGENT, "predict_rul", {})


def test_trace_fields(system):
    r = runner(system)
    r.run(ENGINE_AGENT, "predict_rul", {})
    e = r.events[0]
    assert e.agent == ENGINE_AGENT and e.tool == "predict_rul" and e.model_version == system.suite.model_version and e.status == "ok"
