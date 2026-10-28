"""Compose the final response. Every number shown comes from a tool field (formatted here), not
from model-written prose; model-written text appears only after validation and is labelled."""
from typing import Any, Dict, List, Optional

from Agent.schemas.contracts import (
    AssessmentResponse, ChartData, Citation, DocStatement, EngineFindings, LifeChart, LifeView, ModelMetadataOutput, Passage,
    PredictionBlock, PredictRULOutput, RequestContext, ScreenFleetOutput, SensorTrendsOutput, TraceEvent)

MODEL_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest"}


def prediction_block(p: PredictRULOutput) -> PredictionBlock:
    return PredictionBlock(
        predicted_rul_cycles=p.predicted_rul_cycles, support_rul_cycles=p.support_rul_cycles, model_agreement=p.model_agreement,
        model_agreement_gap_cycles=p.model_agreement_gap_cycles, demo_band=p.demo_band, model_version=p.model_version,
        support_model_version=p.support_model_version, feature_schema_version=p.feature_schema_version, data_role=p.data_role,
        observed_cycles=p.observed_cycles, warnings=list(p.warnings))


def explanation_text(p: Optional[PredictRULOutput], meta: Optional[ModelMetadataOutput]) -> str:
    """Templated 'how to read this estimate' paragraph; numbers are formatted from tool fields."""
    if p is None:
        return ""
    s = (f"The primary model ({p.model_version}) reads {p.observed_cycles} observed cycles of this engine's sensor history through cycle "
         f"{p.cutoff_cycle}, builds rolling and smoothed features from the past observations only, and returns a point estimate of "
         f"{p.predicted_rul_cycles:.1f} operating cycles. There is no uncertainty interval. The Random Forest support model gives "
         f"{p.support_rul_cycles:.1f} cycles, a gap of {p.model_agreement_gap_cycles:.1f} cycles ({AGREEMENT_TEXT[p.model_agreement]}); a small gap "
         "suggests the models read this engine similarly, but this is an uncalibrated heuristic, not a confidence level.")
    try:
        ev = meta.metadata["evaluation"]
        s += (f" On the test engines, the primary model's typical error across arbitrary cutoffs was about "
              f"{ev['test_all_cycles']['xgboost']['MAE']:.1f} cycles (MAE over all rows); the smaller endpoint MAE of "
              f"{ev['test_endpoint']['xgboost']['MAE']:.1f} cycles applies only to the final observed cycle of truncated test engines.")
    except (AttributeError, KeyError, TypeError):
        pass
    return s


def _citations(statements: List[DocStatement], passages: Dict[str, Passage]) -> List[Citation]:
    seen, out = set(), []
    for st in statements:
        for cid in st.source_ids:
            if cid in passages and cid not in seen:
                p = passages[cid]
                seen.add(cid)
                out.append(Citation(chunk_id=p.chunk_id, doc_id=p.doc_id, title=p.title, section=p.section, revision=p.revision))
    return out


def build_markdown(ctx: RequestContext, status: str, pred: Optional[PredictRULOutput], reason: Optional[str], findings: EngineFindings,
                   explanation: str, analyst_summary: str, statements: List[DocStatement], limitations: List[str], warnings: List[str],
                   clarification: Optional[str], abstain: bool, lead: str = "", extra_blocks: Optional[List[str]] = None,
                   show_prediction: bool = True, show_unavailable: bool = True) -> str:
    where = f"{ctx.dataset_split} engine {ctx.engine_id}, cycle {ctx.cutoff_cycle}"
    if clarification and not lead and not statements:
        return clarification
    blocks: List[str] = []
    if clarification and statements:
        blocks.append(clarification)
    if lead:
        blocks.append(lead)
    if not show_prediction:
        pred = None
    if pred is not None:
        blocks.append(f"**Predicted RUL: {pred.predicted_rul_cycles:.1f} cycles** ({where}) · demo band: **{pred.demo_band}** (illustrative, not an aviation limit)\n"
                      f"Support model: {pred.support_rul_cycles:.1f} cycles · {AGREEMENT_TEXT[pred.model_agreement]} (gap {pred.model_agreement_gap_cycles:.1f}) · model `{pred.model_version}`")
    elif reason and show_unavailable:
        blocks.append(f"**Prediction unavailable** ({where}): {reason} No value has been estimated or made up.")
    if abstain:
        blocks.append("**I can't answer the maintenance, repair or diagnosis part of this question.** The sources contain no maintenance procedures, "
                      "repair actions, part-level diagnoses or safety limits, and I will not invent them.")
    if findings.sensor_findings:
        blocks.append(f"**Recent sensor evidence** (through cycle {ctx.cutoff_cycle}; generic sensor identifiers, units not verified)\n"
                      + "\n".join(f"- {f.statement}" for f in findings.sensor_findings))
    blocks.extend(extra_blocks or [])
    if explanation:
        blocks.append("**How to read this estimate**\n" + explanation)
    if analyst_summary:
        blocks.append("**Analyst summary** (model-written, numbers checked against tool results)\n" + analyst_summary)
    if statements:
        blocks.append("**From the project documents**\n" + "\n".join(f"- {s.statement} [{', '.join(s.source_ids)}]" for s in statements))
    if warnings:
        blocks.append("**Warnings**\n" + "\n".join(f"- {w}" for w in warnings))
    if limitations:
        blocks.append("**Limitations**\n" + "\n".join(f"- {l}" for l in limitations[:4]))
    blocks.append(f"_Status: {status}_")
    return "\n\n".join(blocks)


def build_response(*, ctx: RequestContext, route: str, agent_mode: str, status: str, pred: Optional[PredictRULOutput], reason: Optional[str],
                   findings: EngineFindings, explanation: str, statements: List[DocStatement], passages: Dict[str, Passage],
                   limitations: List[str], warnings: List[str], chart: Optional[ChartData], trace: List[TraceEvent], tool_calls: int,
                   latency_s: float, clarification: Optional[str] = None, abstain: bool = False, lead: str = "",
                   extra_blocks: Optional[List[str]] = None, show_prediction: bool = True, show_unavailable: bool = True,
                   intent: str = "assess", response_mode: str = "answer", table: Optional[List[Dict[str, Any]]] = None,
                   evidence: Optional[List[str]] = None, life: Optional[LifeView] = None,
                   life_charts: Optional[List[LifeChart]] = None) -> AssessmentResponse:
    md = build_markdown(ctx, status, pred, reason, findings, explanation, findings.summary, statements, limitations, warnings, clarification,
                        abstain, lead, extra_blocks, show_prediction, show_unavailable)
    return AssessmentResponse(
        context=ctx, route=route, agent_mode=agent_mode, status=status,
        prediction=prediction_block(pred) if (pred and show_prediction) else None,
        intent=intent, response_mode=response_mode, table=table or [], evidence=evidence or [],
        prediction_unavailable_reason=None if (pred and show_prediction) else reason, sensor_findings=findings.sensor_findings,
        explanation=(clarification or explanation or ""), doc_statements=statements, citations=_citations(statements, passages),
        limitations=limitations, warnings=warnings, answer_markdown=md, chart=chart, life=life, life_charts=life_charts or [], trace_id=f"tr-{ctx.request_id}", trace=trace,
        tool_calls=tool_calls, latency_s=round(latency_s, 3))


# ---------------------------------------------------------------------------------- intent blocks (numbers come from tool fields)
def boundary_lead(headline: str, body: str, offer: str) -> str:
    return f"**{headline}**\n\n{body}" + (f"\n\n{offer}" if offer else "")


def data_quality_block(tr: SensorTrendsOutput) -> str:
    d = tr.data_quality
    if not d:
        return ""
    lines = [f"- Rows checked: {d['rows_checked']} (cycles {d['first_cycle']} to {d['last_cycle']}, nothing after the cutoff)",
             f"- Missing values in the settings and sensor columns: {d['missing_values']}",
             f"- Duplicated cycles: {d['duplicated_cycles']}; cycle-order violations: {d['cycle_order_violations']}",
             f"- Cycle gaps: {d['cycle_gaps']} (missing cycles inside gaps: {d['missing_cycles_in_gaps']})"]
    if d.get("constant_model_sensors"):
        lines.append(f"- Constant model sensors over the observed history: {', '.join(d['constant_model_sensors'])}")
    else:
        lines.append("- Constant model sensors over the observed history: none")
    return ("**Data quality checks on the observed history** (measured on the rows through the cutoff)\n" + "\n".join(lines) +
            "\nThese checks are the only evidence of data quality here: a successful prediction does not show the history is clean, and the loader rejects "
            "rows it cannot validate rather than repairing them.")


def range_check_block(tr: SensorTrendsOutput, settings: bool = False) -> str:
    r = tr.range_check
    if not r:
        return ""
    if not r.get("available"):
        return f"**Training-range check not available**: {r.get('reason', 'reference statistics are missing')}. No range claim is made."
    lines = [f"Reference: {r['reference']} ({r['n_reference_rows']} rows). Checked {r['sensors_checked']} model sensors."]
    if settings:
        lines.append("")
        lines.append("| Setting | Latest | Training min | Training max | Within range |")
        lines.append("|---|---|---|---|---|")
        for k, v in r["settings"].items():
            lines.append(f"| {k} | {v['latest']:.4g} | {v['training_min']:.4g} | {v['training_max']:.4g} | {'yes' if v['within_training_range'] else 'NO'} |")
        lines.append("")
        lines.append("The settings are compared with the training range only. They are not model inputs, and a match supports this one domain check, not real-world applicability.")
        return "**Operating-condition check against the training data**\n" + "\n".join(lines)
    out = r["sensors_with_values_outside"]
    if not out:
        lines.append("No model sensor reading in the observed history lies outside the training min-max range.")
    else:
        lines.append(f"Latest cycle: {r['latest_cycle_sensors_outside']} sensor(s) outside the training range.")
        for o in out:
            lines.append(f"- {o['sensor']}: latest {o['latest']:.4g} (training {o['training_min']:.4g} to {o['training_max']:.4g}); "
                         f"{o['observed_rows_outside']} observed row(s) outside")
    lines.append("An out-of-range reading is a flag for review, not a confirmed failure, and a reading inside the range is not a guarantee of health.")
    return "**Readings versus the model's training range**\n" + "\n".join(lines)


def trend_rate_block(tr: SensorTrendsOutput) -> str:
    rows = [s for s in tr.sensors if s.slope_per_cycle is not None and s.recent_slope_per_cycle is not None]
    lines = [f"- {s.sensor}: slope over the last {tr.window_used} cycles {s.slope_per_cycle:+.4f} per cycle, over the most recent {s.recent_window} cycles "
             f"{s.recent_slope_per_cycle:+.4f} per cycle" for s in rows]
    body = "\n".join(lines) if lines else "- Not enough observed cycles to compare a recent slope with the window slope."
    return ("**Recent versus window sensor slopes** (observed patterns only)\n" + body +
            "\nThis compares two overlapping windows of one engine and shows observed patterns only. It does not prove a faster physical degradation rate and does not measure a physical degradation rate, and one noisy window or one prediction "
            "is not evidence that damage is accelerating.")


def sensor_plot_block(tr: SensorTrendsOutput) -> str:
    c = tr.chart
    span = f"cycles {c.cycles[0]} to {c.cycles[-1]}" if c.cycles else "the observed cycles"
    axis = ("Each line shows a sensor's change since the first plotted cycle, in units of that sensor's own spread (unitless), so sensors with different "
            "scales share one axis." if c.mode == "change" else
            "Each line shows raw sensor values. Sensors differ in scale, so compare the shape of each line, not its height.")
    return (f"**Sensor chart** ({len(c.series)} sensors, {span})\n{axis}\nSelection rule: {tr.selection_rule}.\n"
            "These are the sensors that moved most for this engine, not sensors shown to account for degradation. A trend describes what was observed; it does not "
            "identify a cause, a failing part or a maintenance action, and a sensor that moves little can still matter to the model.")


AGREEMENT_TEXT = {"close": "the models agree closely", "moderate": "the models differ moderately", "wide": "the models disagree widely"}  # "wide agreement" read as strong agreement

ROLE_TEXT = {
    "train_in_sample": "in-sample: this engine is part of the training data of the saved models, so predictions on it are optimistic and not a fair test",
    "validation_holdout": "held out from the training data of the saved models",
    "official_test": "official test engine",
}


def life_block(v: LifeView, top_n: int = 6) -> str:
    out = (f"**Life overview** (retrospective: reads this train engine's full run to failure and ignores the selected cutoff)\n"
           f"- Total life: **{v.total_cycles} cycles** (the engine ran to failure at cycle {v.total_cycles}).\n"
           f"- Data role: {ROLE_TEXT.get(v.data_role, v.data_role)}.")
    if v.ranking:
        rows = ["", "**Sensors most associated with progression to failure** (over the full life)", "",
                "| Rank | Sensor | Direction | Spearman rho vs cycle | Late minus early (x own spread) |", "|---|---|---|---|---|"]
        for i, r in enumerate(v.ranking[:top_n], 1):
            rows.append(f"| {i} | {r.sensor} | {r.direction} | {r.spearman_rho:+.3f} | {r.late_minus_early:+.2f} |")
        out += "\n" + "\n".join(rows)
        out += ("\n\nThis ranks the model sensors by how consistently each one rises or falls as this engine ages. It is an observed association, "
                "not evidence that a sensor caused the degradation, and sensor names and units are not verified.")
    return out


def scenario_block(pred: PredictRULOutput, cycles_per_day: float) -> str:
    days = pred.predicted_rul_cycles / cycles_per_day
    return (f"**Scenario conversion (your assumption)**\nAssuming {cycles_per_day:g} operating cycle{'s' if cycles_per_day != 1 else ''} per day: {pred.predicted_rul_cycles:.1f} cycles / "
            f"{cycles_per_day:g} = **{days:.1f} days**. This is an illustrative scenario from an assumption you supplied, not a guaranteed date and not a maintenance plan; "
            "the cycles estimate itself still has no uncertainty interval.")


def fleet_block(out: ScreenFleetOutput, purpose: str) -> str:
    head = []
    if out.threshold_cycles is not None:
        head.append(f"**{out.n_below_threshold} of {out.n_engines} {out.rows[0].dataset_split if out.rows else 'test'}-split engines have a predicted RUL below "
                    f"{out.threshold_cycles:g} cycles.** This is the screening threshold you set; it is a screening criterion, not an approved safety limit.")
    else:
        lead_in = "Screening list for engineer review: lowest" if purpose == "review" else "Lowest"
        head.append(f"**{lead_in} predicted RUL among {out.n_engines} test engines** at each engine's own latest observed cycle.")
    if not out.rows:
        return "\n".join(head) + "\n\nNo engine meets the criterion."
    lines = ["", "| Rank | Split | Engine | Observed cycle | Predicted RUL (cycles) | Demo band | Flags |", "|---|---|---|---|---|---|---|"]
    for r in out.rows:
        lines.append(f"| {r.rank} | {r.dataset_split} | {r.engine_id} | {r.observed_cycle} | {r.predicted_rul_cycles:.1f} | {r.demo_band} | {'; '.join(r.flags) or '-'} |")
    notes = [f"Ranking rule: {out.ranking_rule}. Showing up to {len(out.rows)} rows.",
             "Each engine is read only up to its own latest observed cycle, and those cycles do not represent the same real-world time."]
    if purpose == "review":
        notes.append("This ordering uses predicted RUL and the data flags shown. Real review priority also depends on criticality, consequences of failure and "
                     "operating context, which this application does not have, so treat it as a starting list for an engineer, not a maintenance decision.")
    return "\n".join(head + lines + [""] + [f"- {n}" for n in notes])


def evidence_block(ctx: RequestContext, pred: Optional[PredictRULOutput], trace: List[TraceEvent], data_checksum: str) -> str:
    lines = [f"- Request `{ctx.request_id}` (trace `tr-{ctx.request_id}`): {ctx.dataset_split} engine {ctx.engine_id}, observed cycles up to {ctx.cutoff_cycle}",
             f"- Primary model `{ctx.model_version}`" + (f"; support model `{pred.support_model_version}`; feature schema `{pred.feature_schema_version}`" if pred else ""),
             f"- Data checksum `{data_checksum}`; document corpus `{ctx.corpus_version}`"]
    for e in trace:
        lines.append(f"- Tool call {e.seq}: `{e.tool}` by {e.agent} -> {e.status} ({e.result_ref or 'no result'}"
                     + (f", cache {e.cache_status}" if e.cache_status else "") + f", {e.duration_ms:.0f} ms)")
    return "**Evidence behind this assessment**\n" + "\n".join(lines)
