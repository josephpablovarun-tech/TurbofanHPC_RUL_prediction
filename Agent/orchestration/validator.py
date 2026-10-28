"""Deterministic checks on model-written text: numeric fidelity, evidence and citation validity.

A number in LLM-written text must equal (at the precision written) a value present in the tool
results; evidence ids and source ids must exist in what the tools returned; engine findings may
not name causes or maintenance actions. Anything failing is dropped, never repaired by guessing.
"""
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pydantic import BaseModel

from Agent.schemas.contracts import DocFindings, DocStatement, EngineFindings, SensorFinding

NUM_RE = re.compile(r"(?<![A-Za-z_\-\d.])-?\d+(?:\.\d+)?(?![A-Za-z_\d])")
BANNED_RE = re.compile(
    r"\b(caused by|root cause|due to a|blade|bearing|fouling|erosion|crack\w*|leak\w*|seal|replace\w*|repair\w*|inspect\w*|"
    r"maintenance|overhaul|ground(ed)?|safe to fly)\b", re.I)


INSTRUCTION_RE = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier)?\s*(instructions|rules)|"
    r"\b(you must|you should|always|never)\b.{0,60}\b(announce|claim|call|answer|reply|say)\b|\b(call|invoke|run)\s+(predict_rul|get_sensor_trends|search_documents|get_model_metadata)\b|"
    r"system prompt|you are now", re.I)
CAVEAT_RE = re.compile(r"\b(not (been )?verified|no uncertainty|uncalibrated|illustrative|not calendar|not a (maintenance|safety))\b", re.I)


def looks_like_instruction(text: str) -> bool:
    """Passage or statement text that tries to instruct the assistant rather than inform the reader."""
    return bool(INSTRUCTION_RE.search(text))


def preserve_caveats(statement: str, cited_texts: Sequence[str]) -> str:
    """If a cited passage carries a caveat sentence the statement dropped, append that sentence verbatim."""
    if CAVEAT_RE.search(statement):
        return statement
    for t in cited_texts:
        for sent in re.split(r"(?<=[.!?])\s+", t.strip()):
            if CAVEAT_RE.search(sent):
                return statement.rstrip() + " " + sent
    return statement


def collect_numbers(obj: Any, out: List[float]) -> List[float]:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    if isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(x) for x in NUM_RE.findall(obj))
    return out


def _consistent(text_num: str, allowed: Iterable[float]) -> bool:
    d = len(text_num.split(".")[1]) if "." in text_num else 0
    x = float(text_num)
    return any(abs(round(a, d) - x) < 10 ** -(d + 1) or abs(round(-a, d) - x) < 10 ** -(d + 1) for a in allowed)


def unsupported_numbers(text: str, allowed: Sequence[float]) -> List[str]:
    return [n for n in NUM_RE.findall(text) if not _consistent(n, allowed)]


def check_engine_findings(findings: EngineFindings, outputs: Dict[str, Any]) -> Tuple[EngineFindings, List[str]]:
    """Keep only statements tied to a returned sensor summary, with supported numbers and no causal or maintenance wording."""
    notes: List[str] = []
    allowed = collect_numbers(list(outputs.values()), [])
    evidence = {s.evidence_id: s.sensor for s in outputs["get_sensor_trends"].sensors} if "get_sensor_trends" in outputs else {}
    kept: List[SensorFinding] = []
    for f in findings.sensor_findings:
        if evidence.get(f.evidence_id) != f.sensor:
            notes.append(f"dropped finding for {f.sensor}: evidence id {f.evidence_id!r} was not returned by a tool")
        elif unsupported_numbers(f.statement, allowed):
            notes.append(f"dropped finding for {f.sensor}: numbers not found in tool results {unsupported_numbers(f.statement, allowed)}")
        elif BANNED_RE.search(f.statement):
            notes.append(f"dropped finding for {f.sensor}: causal or maintenance wording")
        else:
            kept.append(f)
    summary = findings.summary
    if summary and (unsupported_numbers(summary, allowed) or BANNED_RE.search(summary)):
        notes.append("dropped model-written summary: unsupported numbers or causal/maintenance wording")
        summary = ""
    return EngineFindings(summary=summary, sensor_findings=kept), notes


def check_doc_findings(findings: DocFindings, passages: Dict[str, str]) -> Tuple[DocFindings, List[str]]:
    """Keep only statements citing retrieved chunk ids whose numbers occur in the cited text."""
    notes: List[str] = []
    kept: List[DocStatement] = []
    for st in findings.statements:
        if not st.source_ids or any(s not in passages for s in st.source_ids):
            notes.append(f"dropped statement: citation {st.source_ids} does not resolve to a retrieved passage")
            continue
        cited_numbers = collect_numbers(" ".join(passages[s] for s in st.source_ids), [])
        bad = unsupported_numbers(st.statement, cited_numbers)
        if bad:
            notes.append(f"dropped statement: numbers {bad} not in the cited passage")
            continue
        if looks_like_instruction(st.statement) or any(looks_like_instruction(passages[x]) for x in st.source_ids):
            notes.append(f"dropped statement: cited passage or statement contains instruction-like text {st.source_ids}")
            continue
        kept.append(DocStatement(statement=preserve_caveats(st.statement, [passages[x] for x in st.source_ids]), source_ids=st.source_ids))
    return DocFindings(statements=kept), notes
