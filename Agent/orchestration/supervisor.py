"""Deterministic supervisor: routes a question to an intent handler, a documentation lookup, an engine assessment, a
fleet screen, a fixed boundary response (decline / reject / not implemented / needs inputs) or a clarification.

No LLM is involved, so routing is testable and cannot be talked into a different tool set. The supervisor also refuses
to treat chat text as a way to change engine, split or cutoff, and it rejects invalid engine or cycle references with a
specific message BEFORE any prediction tool runs (business test pack Q41-Q43).
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from Agent.orchestration.boundaries import BOUNDARIES
from Agent.schemas.contracts import RequestContext

ASSESS_RE = re.compile(
    r"\b(assess\w*|predict\w*|forecast|how long|how many cycles|how much life|remaining life|this engine|selected engine|"
    r"current engine|the engine|my engine|this unit|health\w*|condition|status|sensors?|trends?|readings?|recent|compare|estimate\w*|"
    r"summary|summari[sz]e|reliability|overview|this one|that one)\b")
DOC_RE = re.compile(
    r"\b(what is|what are|what does|(which|what) sensors|define|definition|explain|meaning|means?|dataset|fd001|c-?mapss|hpc|high pressure|compressor|"
    r"model card|limitations?|caveats?|how was|trained|training|features?|glossary|units?|calendar|citation|source|why|trust|reliable)\b")
UNSUPPORTED_RE = re.compile(
    r"\b(repair\w*|fix|replace\w*|overhaul|maintenance (schedule|plan|action|interval|procedure|task)|schedule maintenance|work order|"
    r"inspect\w*|borescope|root cause|why did it fail|which (blade|part|component|bearing)|diagnos\w+|safe to fly|airworthiness|torque|part number)\b")
ENGINE_RE = re.compile(r"\b(?:engine|unit)\s*(?:#|no\.?|number|id)?\s*(\d+)\b")
CYCLE_RE = re.compile(r"\bcycle\s*(?:#|no\.?|number)?\s*(\d+)\b")
NEG_CYCLE_RE = re.compile(r"\bcycle\s*(?:#|no\.?|number)?\s*(?:(-\s*\d+)|minus\s+(\w+))")
FLAG_PATTERNS = {
    "cutoff_bypass_attempt": re.compile(
        r"(ignore|bypass|skip|remove|disable|override)\s+(the\s+|your\s+)?(cutoff|selection|selected)|"
        r"(beyond|after|past|later than|ahead of)\s+(the\s+)?(cutoff|selected cycle)|future\s+(cycles?|data|rows|readings)|"
        r"\b(full|entire)\s+(history|run|trajectory)|\ball cycles\b|until failure|later readings"),
    "instruction_override_attempt": re.compile(
        r"(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier)?\s*(instructions|rules|restrictions|guidelines)|"
        r"system prompt|developer mode|jailbreak|you are now"),
    "code_execution_request": re.compile(r"\b(run|execute|eval)\b.{0,40}\b(code|python|shell|script|command|bash)\b|subprocess|os\.system"),
}

# --- intents, tested in this order (the first match wins). Each entry: (key, regex)
BOUNDARY_PATTERNS = [
    ("control_action", re.compile(r"\b(disable|turn off|silence|mute|override|reset)\b.{0,30}\b(alarms?|alerts?|warnings?|interlocks?|protections?)\b|"
                                  r"change (a |the )?control setting|keep (it |the engine )?running")),
    ("approval_bypass", re.compile(r"bypass.{0,30}(approval|sign.?off|authori[sz]ation)|skip (the )?(maintenance )?approval|"
                                   r"\b(schedule|book|order|raise)\b.{0,30}\b(service|maintenance|work order)\b")),
    ("fabricate", re.compile(r"\b(invent|make up|fabricate|fake|pretend)\b.{0,40}\b(rul|value|number|prediction|estimate|result)\b|plausible (rul|value)")),
    ("safe_to_fly", re.compile(r"safe to fly|airworth|flight release|cleared to fly|fit to fly|safe to (operate|run)")),
    ("part_diagnosis", re.compile(r"\b(which|what)\b.{0,25}\b(blade|bearing|part|component|stage)s?\b.{0,30}\b(damag\w*|fail\w*|faulty|worn|crack\w*|broken|defect\w*|affected|degrad\w*)|"
                                  r"exact (blade|bearing|part|component)")),
    ("root_cause", re.compile(r"root cause|confirmed cause|what caused|cause of the (deterioration|degradation|failure)")),
    ("guarantee", re.compile(r"guarantee|exact (failure )?(date|day)|when exactly will")),
    ("procedure_spec", re.compile(r"torque|approved (settings|procedure|manual)|dismantl|disassembl|part number|tolerances?")),
    ("image_health", re.compile(r"\b(photos?|photographs?|images?|pictures?|x-?rays?)\b")),
    ("fault_vs_degradation", re.compile(r"sensor (fault|faults|error|failure|malfunction)|faulty sensor|(fault|faulty).{0,30}\b(or|versus|vs)\b.{0,30}(degrad|wear)|"
                                        r"(degrad|wear).{0,30}\b(or|versus|vs)\b.{0,30}(sensor|fault)")),
    ("causal", re.compile(r"(chang|adjust|reduc|lower|alter)\w*.{0,40}(operating conditions?|throttle|load|usage|settings?).{0,40}(extend|prolong|improv|increas|lengthen|help|save)|"
                          r"\b(extend|prolong)\w*\b.{0,20}(this|the) engine'?s? life")),
    ("probability", re.compile(r"probabilit|chance of fail|likelihood|risk of fail|(percent|%) (chance|likely)")),
    ("interval", re.compile(r"(prediction|confidence|credible) interval|error bars?|confidence band|uncertainty (band|range)|\b95 ?(%|percent)")),
    ("fleet_reliability", re.compile(r"\bmtbf\b|mean time between|\bavailability\b")),
    ("financial", re.compile(r"downtime savings|financial|cost savings?|\bsavings\b|\broi\b|return on investment|business case")),
    ("local_explanation", re.compile(r"sensors?\b.{0,25}\b(explain|drive|driving|contribut\w+|responsible)\b|particular (rul|estimate|prediction)|"
                                     r"feature (contribution|attribution)|\bshap\b")),
    ("cycle_comparison", re.compile(r"between (cycle )?\d+ and (cycle )?\d+|from cycle \d+ to (cycle )?\d+|cycle \d+ (and|vs\.?|versus) (cycle )?\d+")),
    ("compare_engines", re.compile(r"\bcompare\b.{0,60}\bengines?\b|\bengines?\s*\d+\s*(and|vs\.?|versus|with)\s*\d+|which (engine|one) has (less|more|lower|higher|shorter|longer)")),
]
FLEET_RE = re.compile(r"\bwhich engines?\b|\bengines? (should|to) (an engineer )?review|lowest rul|rank(ing)? (the )?engines|\ball engines\b|\bfleet\b")
CONVERT_RE = re.compile(r"\b(days?|weeks?|months?|hours?|calendar)\b")
RATE_RE = re.compile(r"(\d+(?:\.\d+)?|one|two|three|four|five|six|eight|ten)\s+(?:operating\s+)?cycles?\s*(?:per|a|each|/)\s*day")
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "ten": 10}
THRESH_RE = re.compile(r"(?:below|under|less than|fewer than|<)\s*(\d+(?:\.\d+)?)\s*(?:predicted\s*)?(?:operating\s*)?cycles?")
LAST_N_RE = re.compile(r"last\s+(\d+)\s*(?:observed\s*)?(?:cycles?|observations?|readings?)")
SENSOR_RE = re.compile(r"sensor[_ ]?(\d{1,2})\b")

# --- retrospective life view (TRAIN engines only): total life, sensors that track progression to failure, full-life charts
LIFE_TOTAL_RE = re.compile(
    r"\b(total|whole|entire|full|complete|overall)\s+(life|lifetime|life ?span|history|run|trajectory|cycles)\b|\blife ?(time|span)\b|life overview|"
    r"how (long|many cycles) did (it|this engine|the engine) (last|run|survive|operate)|when did (it|this engine) fail|"
    r"run[- ]to[- ]failure (history|trajectory|curve|overview)")
LIFE_DEGRAD_RE = re.compile(
    r"sensors?\b.{0,40}\b(contribut\w*|degrad\w*|deteriorat\w*|account\w* for|associated|progress\w*|track\w*)|"
    r"\b(contribut\w*|degrad\w*|deteriorat\w*|account\w* for)\b.{0,40}\bsensors?\b")
LIFE_CHART_RE = re.compile(
    r"\b(plot|chart|graph|draw)\b.{0,80}\brul\b|\b(show|display)\b.{0,40}\b(plot|chart|graph|curve|trajectory)s?\b.{0,60}\brul\b|"
    r"\brul\b.{0,50}\b(plot|chart|graph|curve|trajectory|over (its |the )?(life|time))\b|"
    r"actual (rul )?(vs\.?|versus|and) (the )?predicted|predicted (rul )?(vs\.?|versus|and) (the )?actual")
LIFE_RECENT_GUARD = re.compile(r"faster|quicker|recent\w*|accelerat\w*|worsen\w*|last\s+\d+|what (does|is|are)\b|\bdefin\w*|\bmeaning\b|\bmeans?\b")
# ^ cutoff-bound trend questions and documentation questions keep their existing routes
LIFE_TEST_RE = re.compile(r"\b(total|whole|entire|overall)\s+(life|lifetime|cycles)\b|how long did (it|this engine) (last|run|survive)|"
                          r"when did (it|this engine) fail|\blifetime of (this|the) engine\b")
SCALE_RE = re.compile(r"\b(scale|zoom\w*|rescal\w*|normali[sz]\w*|y[- ]?axis|axes|axis|clearer|more clear|easier to (?:read|see)|hard to (?:read|see))\b")
NORMALISE_RE = re.compile(r"normali[sz]\w*|same (?:axis|scale)|relative to (?:its|their) (?:own )?(?:spread|range)")
OVERRIDE_RE = re.compile(r"(ignore|bypass|skip|remove|disable|override)\s+(the\s+|your\s+)?(cutoff|selection|selected)")


def _life_decision(q: str, ctx: RequestContext) -> Optional["RouteDecision"]:
    """Route to the retrospective life view. Train split only, and never for an explicit attempt to override the cutoff."""
    if ctx.dataset_split != "train" or OVERRIDE_RE.search(q):
        return None
    sensors = list(dict.fromkeys(f"sensor_{s}" for s in SENSOR_RE.findall(q)))
    chart, total = bool(LIFE_CHART_RE.search(q)), bool(LIFE_TOTAL_RE.search(q))
    degrad = bool(LIFE_DEGRAD_RE.search(q)) and not LIFE_RECENT_GUARD.search(q)
    scale = bool(SCALE_RE.search(q)) and bool(sensors)  # "adjust the scale of sensor_4 ...": redraw that chart, not the whole overview again
    if not (chart or total or degrad or scale):
        return None
    if scale:
        return RouteDecision("life", intent="life_rescale", params={"sensors": sensors, "normalise": bool(NORMALISE_RE.search(q)), "rul": chart})
    return RouteDecision("life", intent="life_chart" if (chart or sensors) else "life_overview", params={"sensors": sensors})


DOC_INTENTS = [
    # key, regex, query, cite (verified against what the search tool returns), mode
    ("estimate_change", re.compile(r"(why|how).{0,30}(rul|estimate|prediction).{0,40}(increase|went up|rose|rise|go up|jump|change|differ)|"
                                   r"(rul|estimate|prediction).{0,20}(increase|went up|rose)"),
     "estimate need not fall steadily as new cycles arrive point estimate limitations", ["MC-06", "MC-10"]),
    ("short_history", re.compile(r"(only|just)\s+(five|5|a few|few|three|ten)\s+(observations?|cycles?|readings?)|short (histor|series)|few observations|"
                                 r"(five|5) observations"),
     "short histories rolling features fewer observations than window no earlier observations created warning", ["MC-12", "SG-04", "MC-06"]),
    ("before_maintenance", re.compile(r"(extra|more|additional|what other) information.{0,40}(maintenance|plan\w*)|before (planning|scheduling) maintenance|"
                                      r"information (is )?(needed|required).{0,30}maintenance"),
     "before planning maintenance applicability criticality approved procedures inspection evidence uncertainty", ["MC-11", "MC-09"]),
    ("overestimation", re.compile(r"overestimat|underestimat|optimistic|\bbias\w*|tend to (over|under)"),
     "overestimation signed error positive means overestimation risky direction long remaining life", ["MC-03", "MC-05", "MC-06"]),
    ("accuracy", re.compile(r"how accurate|accuracy|how good|\bmae\b|\brmse\b|typical error|performance (on|of)|unseen engines"),
     "evaluation final observed cycle test engines MAE RMSE endpoint all test cycles error", ["MC-03", "MC-04", "MC-05"]),
    ("hpc_definition", re.compile(r"\bhpc\b.{0,30}(mean|stand|located|location|where)|(what|where).{0,20}\bhpc\b|where is the (hpc|high pressure)|high pressure compressor"),
     "location of the HPC high pressure compressor upstream of the combustor degradation", ["DG-09", "DG-06"]),
    ("component_vs_engine", re.compile(r"blade life|component life|engine life|component.{0,25}(label|life)|compressor blade"),
     "engine-level target not component life blade bearing labels cycles until failure endpoint", ["DG-10", "DG-06", "DG-05"]),
    ("other_designs", re.compile(r"another engine|different engine|other engine|fd00[234]|\bfd004\b|transfer|other (dataset|fleet|subset)|generali[sz]"),
     "other engine designs FD004 operating conditions fault modes transfer validation", ["MC-13", "MC-06"]),
    ("actual_vs_predicted", re.compile(r"actual (rul )?(vs\.?|versus|and|or|from)|difference between actual|\bactual rul\b|ground truth|true rul"),
     "actual versus predicted RUL retrospective label supplied RUL error predicted minus actual", ["MC-10", "DG-05"]),
]
ENGINE_INTENTS = [
    # key, regex  (engine tools run with intent-specific arguments)
    ("data_quality", re.compile(r"missing values?|duplicat|\bgaps?\b|data quality|completeness|history (clean|complete|valid)")),
    ("range_check", re.compile(r"out(side)?[ -]of[ -](the )?(model'?s? )?(training )?range|outside.{0,25}training|training range|unusual readings?")),
    ("domain_check", re.compile(r"operating conditions?.{0,40}(match|same|similar|those used)|(match|same).{0,30}operating conditions?|training (domain|conditions)")),
    ("trend_rate", re.compile(r"deteriorat|degrad\w*.{0,15}(faster|quicker)|faster recently|accelerat|worsen")),
    ("sensor_plot", re.compile(r"\b(plot|chart|graph|draw)\b")),
    ("sensor_change", re.compile(r"(which|what) sensors?.{0,40}(changed|moved|varied)|changed most|most (change|movement)|sensor readings? changed")),
    ("evidence", re.compile(r"\bevidence\b|model version|trace|provenance|audit|which model")),
]


@dataclass
class RouteDecision:
    route: str  # assessment | documentation | mixed | clarify | boundary | fleet | reject
    clarification: Optional[str] = None
    abstain_topics: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    notices: List[str] = field(default_factory=list)
    intent: str = "assess"
    boundary: Optional[str] = None
    tool_args: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    doc_query: Optional[str] = None
    cite: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


def _num_word(w: str) -> Optional[float]:
    return float(WORDS[w]) if w in WORDS else (float(w) if re.fullmatch(r"\d+(\.\d+)?", w) else None)


def _reject_ref(ctx: RequestContext, flags, notices, message, intent) -> RouteDecision:
    return RouteDecision("reject", clarification=message, flags=flags, notices=notices, intent=intent)


def route_question(question: str, ctx: RequestContext, store=None) -> RouteDecision:
    q = question.lower().strip()
    where = f"{ctx.dataset_split} engine {ctx.engine_id} at cycle {ctx.cutoff_cycle}"
    flags = [name for name, rx in FLAG_PATTERNS.items() if rx.search(q)]
    notices: List[str] = []
    if flags:
        notices.append(f"Text in a question cannot change the selected context or tool permissions. Only observed cycles up to {ctx.cutoff_cycle} are used.")

    # 1. rule override or code execution typed by the user: reject outright
    if "instruction_override_attempt" in flags or "code_execution_request" in flags:
        return RouteDecision("boundary", flags=flags, notices=notices, intent="rule_override", boundary="rule_override")

    # 2. invalid references are rejected with a specific message before any prediction tool is invoked
    neg = NEG_CYCLE_RE.search(q)
    if neg:
        return _reject_ref(ctx, flags, notices, f"Cycle {neg.group(1).replace(' ', '') if neg.group(1) else 'minus ' + neg.group(2)} is not valid: cycles are positive "
                           f"whole numbers starting at 1. No prediction was run. The selected context is {where}.", "invalid_cycle")
    mentioned_engines = sorted({int(m) for m in ENGINE_RE.findall(q)})
    mentioned_cycles = sorted({int(m) for m in CYCLE_RE.findall(q)})
    compare = any(rx.search(q) for k, rx in BOUNDARY_PATTERNS if k in ("compare_engines", "cycle_comparison"))
    if store is not None and not compare:
        for e in mentioned_engines:
            if e != ctx.engine_id:
                try:
                    store.max_cycle(ctx.dataset_split, e)
                except Exception:
                    ids = store.engine_ids(ctx.dataset_split)
                    return _reject_ref(ctx, flags, notices, f"Engine {e} does not exist in the {ctx.dataset_split} split (engine ids run from {ids[0]} to {ids[-1]}). "
                                       "No prediction was run and none was made up.", "unknown_engine")
        target = ctx.engine_id
        try:
            last = store.max_cycle(ctx.dataset_split, target)
        except Exception:
            last = None
        if last is not None:
            for c in mentioned_cycles:
                if c > last and not [e for e in mentioned_engines if e != ctx.engine_id]:
                    return _reject_ref(ctx, flags, notices, f"Cycle {c} is after the last observed cycle ({last}) of {ctx.dataset_split} engine {ctx.engine_id}. "
                                       "No later data exists, so I will not extrapolate or silently use a different cutoff. Pick a cutoff from 1 to "
                                       f"{last} in the sidebar.", "cycle_after_history")

    # 3. boundary intents (decline / reject / not implemented / needs inputs)
    life = _life_decision(q, ctx)
    for key, rx in BOUNDARY_PATTERNS:
        if rx.search(q):
            if life is not None and key == "local_explanation":  # "which sensors contributed to the degradation" on a train engine
                return life
            return RouteDecision("boundary", flags=flags, notices=notices, intent=key, boundary=key)
    if life is not None:  # before the cutoff-bypass check: this retrospective view is labelled and limited to train engines
        return life
    if ctx.dataset_split == "test" and LIFE_TEST_RE.search(q) and "cutoff_bypass_attempt" not in flags:
        return RouteDecision(
            "clarify", flags=flags, notices=notices, intent="life_unavailable",
            clarification=(f"The total life is only known for train engines, whose histories run to failure. Test engine histories stop before failure, so the data "
                           f"contain no total life for {where}. Switch the dataset split to train to see a full run to failure, or ask for the predicted RUL "
                           "at the selected cutoff."))
    if "cutoff_bypass_attempt" in flags:
        return RouteDecision("boundary", flags=flags, notices=notices, intent="cutoff_bypass", boundary="cutoff_bypass")

    # 4. references to another (existing) engine or cycle cannot be honoured from chat
    other_engines = sorted({e for e in mentioned_engines if e != ctx.engine_id})
    other_cycles = sorted({c for c in mentioned_cycles if c != ctx.cutoff_cycle})
    if other_engines or other_cycles:
        refs = ([f"engine {e}" for e in other_engines] + [f"cycle {c}" for c in other_cycles])
        return RouteDecision(
            "clarify", flags=flags, notices=notices, intent="context_switch",
            clarification=(f"Your question mentions {', '.join(refs)}, but the selected context is {where}. "
                           "I cannot switch engines or cycles from the chat. Change the selection in the sidebar and ask again, "
                           "or ask about the selected engine."))

    # 5. unit conversion needs an explicit usage assumption
    if CONVERT_RE.search(q) and (RATE_RE.search(q) or re.search(r"how many (days|weeks|months)|convert|represent|in days|calendar", q)):
        m = RATE_RE.search(q)
        rate = _num_word(m.group(1)) if m else None
        if rate and rate > 0:
            return RouteDecision("assessment", flags=flags, notices=notices, intent="unit_scenario", params={"cycles_per_day": rate},
                                 tool_args={"get_sensor_trends": {}})
        return RouteDecision(
            "clarify", flags=flags, notices=notices, intent="unit_clarify", doc_query="cycles are not calendar time external assumption about how the engine is used",
            cite=["DG-07"],
            clarification=("Operating cycles are not days. To convert I need an assumption about how many operating cycles this engine flies or runs per day "
                           "(for example \"assume two cycles per day\"). With that I will divide the predicted cycles by it and label the result a scenario, not a maintenance date."))

    # 6. fleet screening (test split only)
    if FLEET_RE.search(q) and not re.search(r"\b(this|selected|current) engine\b", q):
        tm = THRESH_RE.search(q)
        params: Dict[str, Any] = {"threshold_cycles": float(tm.group(1))} if tm else {}
        if tm:
            params["user_threshold"] = True
        return RouteDecision("fleet", flags=flags, notices=notices, intent="fleet", params=params)

    # 7. documentation intents with fixed, verified citations
    for key, rx, query, cite in DOC_INTENTS:
        if rx.search(q):
            return RouteDecision("documentation", flags=flags, notices=notices, intent=key, doc_query=query, cite=cite)

    # 8. engine intents with intent-specific tool arguments
    for key, rx in ENGINE_INTENTS:
        if rx.search(q):
            args: Dict[str, Any] = {}
            lm = LAST_N_RE.search(q)
            if lm:
                args["window"] = max(5, min(100, int(lm.group(1))))
            sm = SENSOR_RE.findall(q)
            if sm and key in ("sensor_plot", "sensor_change", "trend_rate"):
                args["sensors"] = list(dict.fromkeys(f"sensor_{s}" for s in sm))
            if key == "sensor_plot":
                args["chart_mode"] = "change"
                if "sensors" not in args:
                    args["top_n"] = 6
                if lm:
                    args["chart_last_n"] = args["window"]
            if key == "trend_rate":
                args.setdefault("window", 30)
            if key == "evidence":
                return RouteDecision("mixed", flags=flags, notices=notices, intent="evidence",
                                     doc_query="models primary XGBoost support Random Forest trained on engines intended use educational demonstration",
                                     cite=["MC-02", "MC-01"])
            return RouteDecision("assessment", flags=flags, notices=notices, intent=key, tool_args={"get_sensor_trends": args})

    abstain = ["maintenance, repair or diagnosis"] if UNSUPPORTED_RE.search(q) else []
    a, d = bool(ASSESS_RE.search(q)), bool(DOC_RE.search(q))
    if a and d:
        route = "mixed"
    elif a:
        route = "assessment"
    elif d:
        route = "documentation"
    elif abstain:
        route = "documentation"
    else:
        return RouteDecision(
            "clarify", flags=flags, notices=notices, intent="clarify",
            clarification=(f"I can assess {where} (remaining useful life estimate, recent sensor trends, limitations) or answer questions "
                           "about the FD001 dataset, the sensors and the models. What would you like to know?"))
    return RouteDecision(route, abstain_topics=abstain, flags=flags, notices=notices, intent="assess")
