"""Evaluation runner. ``python -m Agent.evals.run [--llm] [--cases ID,ID] [--out DIR]``

Deterministic mode needs no LLM. ``--llm`` uses Ollama (falls back and reports if unavailable).
Every metric is printed with its denominator. Reference numbers come from calling the predictor
directly, independent of the agent.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from Agent.config import AGENT_DIR, REPO_ROOT, load_config
from Agent.orchestration.orchestrator import AgentSystem
from Agent.retrieval.index import SearchIndex
from src.predictor import RULPredictor

CASES = AGENT_DIR / "evals" / "cases.json"
POISON = AGENT_DIR / "evals" / "fixtures" / "poisoned_docs"


def reference_rul(system, split, engine, cutoff):
    h = system.suite.store.history(split, engine, cutoff)
    r = system.suite.predictor.predict(h, cutoff)
    return r.primary_rul, r.support_rul


def check(case, resp, system):
    ex, res = case["expect"], {}
    res["status"] = ("status" not in ex) or resp.status == ex["status"]
    if "route" in ex:
        res["route"] = resp.route == ex["route"]
    if "prediction" in ex:
        res["prediction_presence"] = (resp.prediction is not None) == ex["prediction"]
    if "max_calls" in ex:
        res["budget"] = resp.tool_calls <= ex["max_calls"]
    if "warning_contains" in ex:
        res["warning"] = any(ex["warning_contains"] in w for w in (resp.prediction.warnings if resp.prediction else []))
    if "text_contains" in ex:
        res["text_contains"] = ex["text_contains"] in resp.answer_markdown
    for bad in ex.get("text_not_contains", []):
        res[f"absent:{bad}"] = bad not in resp.answer_markdown
    if ex.get("abstain"):
        res["abstained"] = resp.status == "insufficient_evidence" and "can't answer" in resp.answer_markdown
    if ex.get("flag_notice"):
        res["notice"] = any("cannot change the selected context" in w for w in resp.warnings)
    if "chart_max_cycle" in ex:
        res["chart_cutoff"] = resp.chart is not None and max(resp.chart.cycles) <= ex["chart_max_cycle"]
    return res


def run(llm=False, only=None, out_dir=None):
    cfg = load_config()
    cfg = dict(cfg)
    out = Path(out_dir) if out_dir else AGENT_DIR / "evals" / "results"
    out.mkdir(parents=True, exist_ok=True)
    cfg["logging"] = {"trace_file": str(out / "eval_traces.jsonl")}
    cfg["alerts"] = {"enabled": False}  # evals must never post to Slack, even if a webhook URL is set in the environment
    system = AgentSystem.from_config(cfg)
    poisoned = AgentSystem.from_config(cfg, extra_doc_dirs=[POISON])
    mode = "llm" if llm else "deterministic"
    spec = json.loads(CASES.read_text())
    cases = [c for c in spec["cases"] if not only or c["id"] in only]
    rows, lat, calls = [], [], []
    m = {"numeric_fidelity": [0, 0], "context_integrity": [0, 0], "citation_validity": [0, 0], "retrieval_recall_at_4": [0, 0],
         "abstention": [0, 0], "task_completion": [0, 0], "budget_respected": [0, 0]}
    for c in cases:
        s = poisoned if c.get("poisoned_docs") else system
        s.suite.faults.clear()
        s.suite.faults.update(c.get("fault", {}))
        t = time.monotonic()
        print(f"[{c['id']}] running...", flush=True)
        try:
            resp = s.run(c["engine"], c["split"], c["cutoff"], c["q"], mode=mode)
        except Exception as e:  # a crash is a failed case, reported
            rows.append({"id": c["id"], "category": c["category"], "passed": False, "error": f"{type(e).__name__}: {e}"})
            continue
        s.suite.faults.clear()
        lat.append(time.monotonic() - t)
        calls.append(resp.tool_calls)
        res = check(c, resp, s)
        print(f"[{c['id']}] {resp.status} · {resp.agent_mode} · {resp.tool_calls} calls · {resp.latency_s:.1f}s", flush=True)
        if resp.prediction is not None:
            ref, sup = reference_rul(s, c["split"], c["engine"], c["cutoff"])
            ok = abs(resp.prediction.predicted_rul_cycles - ref) < 1e-6 and abs(resp.prediction.support_rul_cycles - sup) < 1e-6
            res["numeric_fidelity"] = ok
            m["numeric_fidelity"][0] += ok; m["numeric_fidelity"][1] += 1
        ids = {(e.args.get("engine_id"), ) for e in resp.trace}
        ctx_ok = all(e.model_version == resp.context.model_version for e in resp.trace) and resp.context.engine_id == c["engine"] \
            and resp.context.cutoff_cycle == c["cutoff"] and (resp.prediction is None or resp.prediction.observed_cycles <= c["cutoff"])
        m["context_integrity"][0] += ctx_ok; m["context_integrity"][1] += 1
        res["context_integrity"] = ctx_ok
        if resp.doc_statements:
            retrieved = {p for st in resp.doc_statements for p in st.source_ids}
            valid = all(any(cit.chunk_id == p for cit in resp.citations) for p in retrieved)
            m["citation_validity"][0] += valid; m["citation_validity"][1] += 1
            res["citation_validity"] = valid
        if "gold" in c["expect"]:
            hit = bool(set(c["expect"]["gold"]) & {cit.chunk_id for cit in resp.citations})
            m["retrieval_recall_at_4"][0] += hit; m["retrieval_recall_at_4"][1] += 1
            res["gold_cited"] = hit
        if c["expect"].get("abstain"):
            m["abstention"][0] += res["abstained"]; m["abstention"][1] += 1
        m["budget_respected"][0] += resp.tool_calls <= cfg["limits"]["tool_budget"]; m["budget_respected"][1] += 1
        passed = all(res.values())
        m["task_completion"][0] += passed; m["task_completion"][1] += 1
        rows.append({"id": c["id"], "category": c["category"], "passed": passed, "status": resp.status, "route": resp.route,
                     "agent_mode": resp.agent_mode, "tool_calls": resp.tool_calls, "latency_s": resp.latency_s,
                     "failed_checks": [k for k, v in res.items() if not v]})
    p = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None
    summary = {"mode": mode, "cases_run": len(cases), "corpus_version": system.suite.corpus_version, "model_version": system.suite.model_version,
               "metrics": {k: {"passed": v[0], "of": v[1]} for k, v in m.items()},
               "latency_s": {"p50": p(lat, .5), "p95": p(lat, .95)}, "tool_calls": {"mean": statistics.mean(calls) if calls else None, "max": max(calls) if calls else None},
               "rows": rows}
    (out / f"results_{mode}.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# Agent evaluation ({mode})", "", f"Model `{summary['model_version']}` · corpus `{summary['corpus_version']}` · {len(cases)} cases", "",
             "| Metric | Result |", "|---|---|"]
    lines += [f"| {k} | {v['passed']}/{v['of']} |" for k, v in summary["metrics"].items()]
    lines += ["", f"Latency p50 {summary['latency_s']['p50']}, p95 {summary['latency_s']['p95']} s. Mean tool calls {summary['tool_calls']['mean']}.", "",
              "| Case | Category | Pass | Status | Calls | Failed checks |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['category']} | {'yes' if r['passed'] else 'NO'} | {r.get('status', r.get('error'))} | {r.get('tool_calls', '')} | {', '.join(r.get('failed_checks', []))} |" for r in rows]
    (out / f"report_{mode}.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--cases", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    s = run(a.llm, set(filter(None, a.cases.split(","))) or None, a.out or None)
    print((Path(a.out) if a.out else AGENT_DIR / "evals" / "results") / f"report_{s['mode']}.md")
    for k, v in s["metrics"].items():
        print(f"{k}: {v['passed']}/{v['of']}")
    print("failed:", [r["id"] for r in s["rows"] if not r["passed"]])
