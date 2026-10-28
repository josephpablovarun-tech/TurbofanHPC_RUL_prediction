"""AgentSystem: dashboard input -> immutable context -> supervisor -> specialists -> validation -> response.

Request flow and contracts follow the roadmap. Model and corpus versions are resolved once at
request start and bound into the context; every tool call, result and trace event carries them.
"""
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from Agent.config import REPO_ROOT, resolve
from Agent.orchestration.agents import (
    DOC_AGENT, DOC_TOOLS, ENGINE_AGENT, ENGINE_TOOLS, FLEET_AGENT, FLEET_TOOLS, run_doc_agent, run_engine_agent, run_fleet_agent)
from Agent.orchestration.boundaries import BOUNDARIES
from Agent.orchestration.harness import ToolRunner
from Agent.orchestration.llm import OllamaClient
from Agent.orchestration.session import Session
from Agent.orchestration.supervisor import route_question
from Agent.orchestration.synthesis import (
    boundary_lead, build_response, data_quality_block, evidence_block, explanation_text, fleet_block, range_check_block, scenario_block,
    life_block, sensor_plot_block, trend_rate_block)
from Agent.retrieval.index import SearchIndex
from Agent.schemas.contracts import AssessmentResponse, EngineFindings, RequestContext
from Agent.tools.alerts import AlertService
from Agent.tools.datastore import DataStore, SelectionError
from Agent.tools.life_view import LifeViewer
from Agent.tools.tools import ToolSuite
from src.features import MODEL_SENSORS
from src.predictor import RULPredictor

FALLBACK_LIMITATIONS = [
    "Simulated FD001 data; estimates are in operating cycles, not calendar time, and are not maintenance decisions.",
    "Point estimate only: no uncertainty interval is provided.",
]
ABSTAIN_QUERY = "sources contain no maintenance procedures repair actions diagnoses safety limits not covered"


class AgentSystem:
    def __init__(self, cfg: Dict[str, Any], suite: ToolSuite, llm=None):
        self.cfg, self.suite, self.llm = cfg, suite, llm
        self._life_viewer: Optional[LifeViewer] = None
        self.alerts: Optional[AlertService] = None  # Slack alert on high estimated life used; wired in from_config when enabled

    @property
    def life(self) -> LifeViewer:
        """Retrospective life view for train engines (outside the cutoff-bound tool harness; see tools/life_view.py)."""
        if self._life_viewer is None:
            self._life_viewer = LifeViewer(self.suite.store, self.suite.predictor, self.suite.faults)
        return self._life_viewer

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], llm=None, extra_doc_dirs: Sequence = ()) -> "AgentSystem":
        predictor = RULPredictor.from_dir(resolve(cfg["model"]["model_dir"]), cfg["model"]["primary"], cfg["model"]["support"])
        store = DataStore(resolve(cfg["data"]["raw_dir"]), cfg["data"]["train"], cfg["data"]["test"], cfg["data"].get("rul"))
        index = SearchIndex.from_dirs([resolve(cfg["retrieval"]["docs_dir"])] + [Path(d) for d in extra_doc_dirs])
        metrics = json.loads(resolve(cfg["metrics_file"]).read_text())
        suite = ToolSuite(predictor, store, index, cfg, metrics)
        if llm is None and "llm" in cfg:
            c = cfg["llm"]
            llm = OllamaClient(c["base_url"], c["model"], c.get("temperature", 0), c.get("seed", 42), c.get("request_timeout_s", 90))
        system = cls(cfg, suite, llm)
        if cfg.get("alerts", {}).get("enabled", False):
            system.alerts = AlertService.from_config(cfg)
            suite.on_prediction = system.alerts.on_prediction
        return system

    def llm_status(self) -> Tuple[bool, str]:
        if self.llm is None:
            return False, "no LLM configured"
        return self.llm.available() if hasattr(self.llm, "available") else (True, "LLM client ready")

    # ------------------------------------------------------------------------------------------------
    def run(self, engine_id: int, dataset_split: str, cutoff_cycle: int, question: str, mode: str = "auto",
            session: Optional[Session] = None) -> AssessmentResponse:
        t0 = time.monotonic()
        lim, th = self.cfg["limits"], self.cfg["thresholds"]
        try:
            ctx = RequestContext(request_id=uuid.uuid4().hex[:12], engine_id=int(engine_id), dataset_split=dataset_split,
                                 cutoff_cycle=int(cutoff_cycle), question=(question or "").strip()[:1000],
                                 model_version=self.suite.model_version, corpus_version=self.suite.corpus_version)
        except (ValidationError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid request context: {e}")
        if session is not None:
            session.note_context(ctx)
        warnings: List[str] = []

        def finish(**kw) -> AssessmentResponse:
            resp = build_response(ctx=ctx, latency_s=time.monotonic() - t0, **kw)
            if session is not None:
                session.add(ctx, ctx.question, resp.answer_markdown)
            self._log(resp)
            return resp

        empty = dict(findings=EngineFindings(), explanation="", statements=[], passages={}, pred=None, chart=None, trace=[], tool_calls=0,
                     limitations=FALLBACK_LIMITATIONS)
        try:
            self.suite.store.validate_selection(ctx.dataset_split, ctx.engine_id, ctx.cutoff_cycle)
        except SelectionError as e:
            return finish(route="rejected", agent_mode="none", status="error", reason=str(e), warnings=[], **empty)

        decision = route_question(ctx.question, ctx, self.suite.store)
        warnings += decision.notices
        intent = decision.intent

        if decision.route == "reject":  # invalid engine or cycle reference: no tool is invoked
            return finish(route="rejected", agent_mode="none", status="error", reason=decision.clarification, warnings=warnings,
                          clarification=decision.clarification, intent=intent, response_mode="reject", **empty)
        if decision.route == "clarify" and not decision.cite:
            return finish(route="clarify", agent_mode="none", status="insufficient_evidence", reason=None, warnings=warnings,
                          clarification=decision.clarification, intent=intent, response_mode="clarify", **empty)

        if decision.route == "life":
            return self._life(ctx, decision, finish, warnings, empty)

        llm = None
        if mode in ("llm", "auto") and self.llm is not None:
            ok, msg = self.llm_status()
            if ok:
                llm = self.llm
            elif mode == "llm":
                warnings.append(f"LLM mode was requested but is unavailable ({msg}); ran deterministically.")
        deadline = t0 + lim["deadline_s"]
        runner = ToolRunner(self.suite, ctx, {ENGINE_AGENT: set(ENGINE_TOOLS), DOC_AGENT: set(DOC_TOOLS), FLEET_AGENT: set(FLEET_TOOLS)},
                            lim["tool_budget"], lim["tool_timeout_s"], deadline)
        history = session.history_text(ctx) if session is not None else ""

        if decision.route == "fleet":
            return self._fleet(ctx, decision, runner, finish, warnings, empty)
        if decision.route == "boundary" or (decision.route == "clarify" and decision.cite):
            return self._boundary(ctx, decision, runner, finish, warnings, empty, deadline, lim)

        need_engine = decision.route in ("assessment", "mixed")
        need_docs = decision.route in ("documentation", "mixed") or bool(decision.abstain_topics)
        modes: List[str] = []
        fixed_docs = bool(decision.abstain_topics) or bool(decision.cite)  # fixed sources: the LLM is not involved in that answer

        # a chart request needs only the sensor tool: no prediction card, no model-written text, no repeat of the RUL answer
        plot_only = intent == "sensor_plot" and decision.route == "assessment"
        eng = doc = None
        if need_engine:
            detail = intent in ("sensor_change", "trend_rate", "sensor_plot")
            if plot_only:
                llm = None
            eng = run_engine_agent(runner, ctx, llm, history, lim["max_agent_steps"], deadline, tool_args=decision.tool_args, detail=detail,
                                   tools=("get_sensor_trends",) if plot_only else ENGINE_TOOLS)
            modes.append(eng.mode)
        if need_docs:
            if decision.abstain_topics:
                query = ABSTAIN_QUERY
            elif decision.doc_query:
                query = decision.doc_query
            else:
                query = ctx.question + (" model limitations" if decision.route == "mixed" else "")
            doc = run_doc_agent(runner, ctx, None if fixed_docs else llm, history, lim["max_agent_steps"], deadline, query,
                                cite=decision.cite or None)
            if not fixed_docs:
                modes.append(doc.mode)

        pred = eng.outputs.get("predict_rul") if eng else None
        trends = eng.outputs.get("get_sensor_trends") if eng else None
        meta = eng.outputs.get("get_model_metadata") if eng else None
        reason = None
        if need_engine and pred is None and not plot_only:
            reason = eng.errors.get("predict_rul", "prediction was not produced")
        if eng:
            for tool, err in eng.errors.items():
                warnings.append(f"{tool} unavailable: {err}")
            warnings += (list(pred.warnings) if pred else []) + (list(trends.quality_warnings) if trends else [])
        if doc:
            for tool, err in doc.errors.items():
                warnings.append(f"{tool} unavailable: {err}")
        statements = doc.statements if doc else []
        if decision.abstain_topics and doc:  # keep only the 'not covered' statement(s); other hits are not relevant to the request
            statements = [s for s in doc.statements if any(c.startswith("MC-09") for c in s.source_ids)]

        extra: List[str] = []
        lead = ""
        if need_docs and doc is not None and not doc.searched:
            lead = ("**The project documents could not be searched**, so I cannot give a sourced answer and have not cited anything. "
                    "No document content is claimed below.")
        if trends is not None:
            if intent == "data_quality":
                extra.append(data_quality_block(trends))
            elif intent == "range_check":
                extra.append(range_check_block(trends))
            elif intent == "domain_check":
                extra.append(range_check_block(trends, settings=True))
            elif intent == "trend_rate":
                extra.append(trend_rate_block(trends))
        if plot_only:
            if trends is not None:
                extra.append(sensor_plot_block(trends))
            else:
                lead = f"**The sensor chart could not be produced**: {eng.errors.get('get_sensor_trends', 'no sensor data was returned')}"
        if intent == "unit_scenario" and pred is not None:
            extra.append(scenario_block(pred, decision.params["cycles_per_day"]))
        if intent == "evidence":
            extra.append(evidence_block(ctx, pred, runner.events, self.suite.store.data_checksum))
        status = self._status(decision, need_engine, need_docs, pred, trends, meta, doc)
        if plot_only:
            status = "ok" if trends is not None else "error"
        if intent in ("data_quality", "range_check", "domain_check", "trend_rate", "sensor_change", "sensor_plot", "unit_scenario", "evidence") and trends is not None \
                and pred is None and status == "partial":
            status = "partial"
        limitations = list(meta.metadata.get("limitations", FALLBACK_LIMITATIONS)) if meta else FALLBACK_LIMITATIONS
        notes = (eng.notes if eng else []) + (doc.notes if doc else [])
        if notes:
            warnings += [f"Validation note: {n}" for n in notes]
        agent_mode = "deterministic" if llm is None else ("deterministic_fallback" if any(m != "llm" for m in modes) else "llm")
        show_expl = need_engine and intent in ("assess", "evidence", "unit_scenario")
        return finish(route=decision.route, agent_mode=agent_mode, status=status, pred=pred, reason=reason,
                      findings=eng.findings if eng else EngineFindings(), explanation=explanation_text(pred, meta) if show_expl else "",
                      statements=statements, passages=doc.passages if doc else {}, limitations=limitations, warnings=warnings,
                      chart=trends.chart if trends else None, trace=runner.events, tool_calls=runner.executed,
                      abstain=bool(decision.abstain_topics), lead=lead, extra_blocks=[b for b in extra if b], intent=intent,
                      show_prediction=not plot_only, show_unavailable=not plot_only,
                      evidence=[f"{e.tool}#{e.seq}:{e.status}" for e in runner.events] if intent == "evidence" else [])

    # ------------------------------------------------------------------------------------------------
    def _boundary(self, ctx, decision, runner, finish, warnings, empty, deadline, lim):
        """Fixed, reviewed wording for declined, rejected, unsupported and input-needed requests (see boundaries.py)."""
        lim_steps = lim["max_agent_steps"]
        if decision.boundary:
            b = BOUNDARIES[decision.boundary]
            lead, cite, query, mode = boundary_lead(b.headline, b.body, b.offer), b.cite, b.doc_query, b.mode
            with_pred, with_checks = b.with_prediction, b.with_checks
        else:  # clarification that also cites a source (e.g. cycles are not days)
            lead, cite, query, mode = decision.clarification, decision.cite, decision.doc_query, "clarify"
            with_pred = with_checks = False
        eng = doc = None
        if with_pred or with_checks:
            tools = ENGINE_TOOLS if with_pred else ("get_sensor_trends",)
            eng = run_engine_agent(runner, ctx, None, "", lim_steps, deadline, tool_args=decision.tool_args, tools=tools)
        if cite:
            doc = run_doc_agent(runner, ctx, None, "", lim_steps, deadline, query, cite=cite)
        pred = eng.outputs.get("predict_rul") if eng else None
        trends = eng.outputs.get("get_sensor_trends") if eng else None
        meta = eng.outputs.get("get_model_metadata") if eng else None
        reason = (eng.errors.get("predict_rul", "prediction was not produced") if (eng and with_pred and pred is None) else None)
        if eng:
            for tool, err in eng.errors.items():
                warnings.append(f"{tool} unavailable: {err}")
            warnings += (list(pred.warnings) if pred else []) + (list(trends.quality_warnings) if trends else [])
        extra: List[str] = []
        if with_checks and trends is not None:
            extra += [data_quality_block(trends), range_check_block(trends)]
        if doc:
            for tool, err in doc.errors.items():
                warnings.append(f"{tool} unavailable: {err}")
            if not doc.searched:
                extra.append("**The project documents could not be searched**, so no source is cited for the statement above; it is my fixed application boundary, not a retrieved document.")
        notes = (eng.notes if eng else []) + (doc.notes if doc else [])
        warnings += [f"Validation note: {n}" for n in notes]
        statements = doc.statements if doc else []
        if mode == "decline" or mode == "unsupported" or mode == "clarify_inputs" or mode == "clarify":
            status = "insufficient_evidence"
        elif decision.boundary in ("fabricate", "cutoff_bypass"):  # a real assessment follows the refusal
            status = "ok" if (pred is not None and trends is not None and meta is not None) else ("partial" if (pred is not None or trends is not None) else "error")
        else:
            status = "error"
        response_mode = {"decline": "decline", "unsupported": "not_implemented", "clarify_inputs": "clarify", "clarify": "clarify", "reject": "reject"}[mode]
        limitations = list(meta.metadata.get("limitations", FALLBACK_LIMITATIONS)) if meta else FALLBACK_LIMITATIONS
        return finish(route="boundary" if decision.boundary else "clarify", agent_mode="deterministic", status=status, pred=pred, reason=reason,
                      findings=eng.findings if (eng and with_pred) else EngineFindings(), explanation="", statements=statements,
                      passages=doc.passages if doc else {}, limitations=limitations if with_pred else FALLBACK_LIMITATIONS, warnings=warnings,
                      chart=trends.chart if (trends and with_pred) else None, trace=runner.events, tool_calls=runner.executed,
                      lead=lead, extra_blocks=[b for b in extra if b], show_prediction=with_pred, show_unavailable=with_pred,
                      intent=decision.intent, response_mode=response_mode)

    def _life(self, ctx, decision, finish, warnings, empty):
        """Retrospective view of a train engine's full run to failure: total life, sensors that track progression, full-life charts."""
        try:
            data = self.life.get(ctx.dataset_split, ctx.engine_id)
        except SelectionError as e:
            return finish(route="life", agent_mode="deterministic", status="error", reason=str(e), warnings=warnings, intent=decision.intent,
                          show_prediction=False, show_unavailable=False, lead=f"**The life view could not be produced**: {e}", **empty)
        view = data.summary
        requested = list(decision.params.get("sensors", []))
        valid = [s for s in requested if s in MODEL_SENSORS]
        bad = [s for s in requested if s not in MODEL_SENSORS]
        if not requested and decision.intent == "life_overview":
            valid = [r.sensor for r in view.ranking[:3]]
        rescale = decision.intent == "life_rescale"
        with_rul = bool(decision.params.get("rul")) if rescale else True
        charts, notes = self.life.charts(data, valid, with_rul=with_rul, normalise=bool(decision.params.get("normalise")))
        if bad:
            notes.append(f"{', '.join(bad)} is not one of the sensors the models read, so it is not plotted. Available: {', '.join(MODEL_SENSORS)}.")
        warnings += [f"Retrospective view: this reads all {view.total_cycles} recorded cycles of train engine {ctx.engine_id}, including cycles after the selected "
                     f"cutoff ({ctx.cutoff_cycle}). It is available for train engines only; the test split and ordinary answers keep the strict cutoff."]
        if view.data_role == "train_in_sample" and with_rul:
            warnings.append("This engine is part of the training data of the saved models (in-sample), so predicted values on it are optimistic and not a fair test.")
        if rescale:
            if not valid:
                note = "**No chart drawn**: name one of the sensors the models read, for example \"plot sensor_4 over the full life\"."
            elif decision.params.get("normalise"):
                note = (f"**Rescaled chart**: {', '.join(valid)} shown as change since cycle 1 in units of the sensor's own spread (unitless), so the shape is visible "
                        "whatever the sensor's units or offset.")
            else:
                spans = "; ".join(f"{s} ranges from {min(data.sensors[s]):.4g} to {max(data.sensors[s]):.4g}" for s in valid)
                note = ("**Rescaled chart**: the y-axis now fits each sensor's own range instead of starting at zero, which had flattened the line. "
                        f"{spans}. Ask for \"normalise {valid[0]}\" to show change since cycle 1 in units of its own spread.")
            return finish(route="life", agent_mode="deterministic", status="ok", reason=None, warnings=warnings + notes, intent=decision.intent,
                          show_prediction=False, show_unavailable=False, extra_blocks=[note], life=view, life_charts=charts, **empty)
        warnings += notes + list(view.notes)
        shown = ", ".join(valid) if valid else "no sensor"
        intro = f"**Charts above**: {shown} over the full life, and the actual vs model-predicted RUL."
        if not valid:
            intro += " Name a sensor to add its chart, for example: \"plot " + (view.ranking[0].sensor if view.ranking else "sensor_11") + " and the RUL\"."
        elif decision.intent == "life_overview":
            intro += " To see one sensor on its own with the RUL, ask for example: \"plot " + valid[0] + " and the RUL\"."
        return finish(route="life", agent_mode="deterministic", status="ok", reason=None, warnings=warnings, intent=decision.intent,
                      show_prediction=False, show_unavailable=False, extra_blocks=[life_block(view), intro], life=view, life_charts=charts, **empty)

    def _fleet(self, ctx, decision, runner, finish, warnings, empty):
        args = {k: v for k, v in decision.params.items() if k == "threshold_cycles"}
        res = run_fleet_agent(runner, args)
        purpose = "review" if re.search(r"review|priorit|first", ctx.question.lower()) else "rank"
        if res.output is None:
            warnings.append(f"screen_fleet unavailable: {res.error}")
            return finish(route="fleet", agent_mode="deterministic", status="error", reason=res.error, warnings=warnings, intent="fleet",
                          trace=runner.events, tool_calls=runner.executed, show_prediction=False, show_unavailable=False,
                          lead=f"**Fleet screening was not produced**: {res.error}", **{k: v for k, v in empty.items() if k not in ("trace", "tool_calls")})
        out = res.output
        table = [r.model_dump() for r in out.rows]
        return finish(route="fleet", agent_mode="deterministic", status="ok", reason=None, warnings=warnings, intent="fleet",
                      trace=runner.events, tool_calls=runner.executed, show_prediction=False, show_unavailable=False, table=table,
                      extra_blocks=[fleet_block(out, purpose)], **{k: v for k, v in empty.items() if k not in ("trace", "tool_calls")})

    @staticmethod
    def _status(decision, need_engine, need_docs, pred, trends, meta, doc) -> str:
        if decision.abstain_topics:
            return "insufficient_evidence"
        engine_ok = pred is not None and trends is not None and meta is not None
        engine_any = pred is not None or trends is not None
        doc_ok = doc is not None and doc.searched and not doc.errors
        if decision.route == "assessment":
            return "ok" if engine_ok else ("partial" if engine_any else "error")
        if decision.route == "documentation":
            if not doc_ok:
                return "error"
            return "ok" if doc.statements else "insufficient_evidence"
        # mixed
        if not engine_any and not doc_ok:
            return "error"
        return "ok" if engine_ok and doc_ok and doc.statements else "partial"

    def _log(self, resp: AssessmentResponse) -> None:
        path = self.cfg.get("logging", {}).get("trace_file")
        if not path:
            return
        try:
            p = resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            rec = {"trace_id": resp.trace_id, "context": resp.context.model_dump(), "route": resp.route, "status": resp.status,
                   "agent_mode": resp.agent_mode, "latency_s": resp.latency_s, "tool_calls": resp.tool_calls,
                   "events": [e.model_dump() for e in resp.trace]}
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass  # logging must never break an answer
