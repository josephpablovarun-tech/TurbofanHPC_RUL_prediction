"""Streamlit dashboard for the RUL agent. Run from the repo root: ``streamlit run Agent/app/app.py``."""
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from Agent.app.charts import line_chart  # noqa: E402
from Agent.app.explorer import render_explorer  # noqa: E402
from Agent.app.fleet import render_fleet  # noqa: E402
from Agent.config import load_config  # noqa: E402
from Agent.orchestration.orchestrator import AgentSystem  # noqa: E402
from Agent.orchestration.session import Session  # noqa: E402
from Agent.schemas.contracts import PredictRULInput, RequestContext  # noqa: E402

st.set_page_config(page_title="Turbofan HPC Remaining Useful Life Agent", layout="wide")
BAND_COLOUR = {"Normal": "#2e7d32", "Watch": "#b26a00", "Review": "#b3261e"}


@st.cache_resource(show_spinner="Loading models, data and documents...")
def get_system():
    return AgentSystem.from_config(load_config())


def main():
    try:
        system = get_system()
    except Exception as e:  # missing models/data: say so plainly
        st.error(f"Could not start the agent: {e}")
        st.stop()
    store = system.suite.store
    if "session" not in st.session_state:
        st.session_state.session, st.session_state.chat, st.session_state.sel = Session(), [], None

    with st.sidebar:
        with st.container(key="agent_controls"):  # kept rendered so the selection survives a switch of view; hidden below for the other views
            st.header("Selection")
            split = st.selectbox("Dataset split", ["test", "train"])
            eid = st.selectbox("Engine", store.engine_ids(split))
            mx = store.max_cycle(split, eid)
            default = max(1, mx // 2) if split == "train" else mx  # train: half of the known total life; test: the last recorded cycle (official endpoint)
            cutoff = st.slider("Cutoff cycle (data after this is never read)", 1, mx, default)
            at_end = (" (the last recorded cycle, the official evaluation endpoint)" if split == "test" else " (the last recorded cycle, where the engine failed)") if cutoff == mx else ""
            st.caption(f"Engine {eid} has {mx} observed cycles in the {split} file. Replaying as of cycle {cutoff} of {mx}{at_end}.")
            ok, msg = system.llm_status()
            mode = st.radio("Agent mode", ["deterministic", "llm"], help="LLM mode uses a local Ollama model; deterministic needs no model.")
            st.caption(("LLM available: " if ok else "LLM unavailable: ") + str(msg))
            outage = st.checkbox("Simulate prediction outage (fault injection)")
            system.suite.faults.clear()
            if outage:
                system.suite.faults["predict_rul"] = "unavailable"
            if st.button("Reset conversation"):
                st.session_state.session.reset(); st.session_state.chat = []
        st.caption(f"Primary: {system.suite.model_version}\n\nSupport: {system.suite.predictor.support_version}\n\nCorpus: {system.suite.corpus_version}")
        st.caption(system.alerts.status_text() if system.alerts else "Slack alerts: off (set alerts.enabled in configs/agent.yaml).")
        if st.session_state.get("view", "RUL agent") != "RUL agent":  # the selection belongs to the RUL agent view only
            st.html("<style>.st-key-agent_controls{display:none;}</style>")
            st.caption("Dataset split, engine, cutoff and agent mode apply to the RUL agent view only.")

    sel = (split, eid, cutoff)
    if st.session_state.sel not in (None, sel):  # context changed: no carry-over of chat or evidence
        st.session_state.chat, st.session_state.session = [], Session()
    st.session_state.sel = sel

    st.title("Turbofan HPC Remaining Useful Life Agent")
    st.caption("NASA C-MAPSS FD001 (simulated). Estimates are in operating cycles and are a demonstration, not a maintenance decision.")
    view = st.radio("View", ["RUL agent", "Data explorer", "Fleet scoring"], horizontal=True, key="view", label_visibility="collapsed")
    if view == "Data explorer":
        render_explorer(system)
        return
    if view == "Fleet scoring":
        render_fleet(system)
        return
    render_estimate_box(system, split, eid, cutoff)
    if split == "train":
        render_life_panel(system, eid)
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            render(m)

    q = st.chat_input(f"Ask about {split} engine {eid} at cycle {cutoff}")
    if q:
        st.session_state.chat.append({"role": "user", "text": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"), st.spinner("Working..."):
            try:
                r = system.run(eid, split, cutoff, q, mode=mode, session=st.session_state.session)
                m = {"role": "assistant", "resp": r}
            except Exception as e:
                m = {"role": "assistant", "text": f"Request failed: {e}"}
            render(m)
        st.session_state.chat.append(m)


ROLE_LABEL = {"train_in_sample": "In-sample", "validation_holdout": "Held out", "official_test": "Official test"}


def render_estimate_box(system, split, eid, cutoff):
    """RUL estimate as of the cutoff slider, for both splits. Uses the same tool path as the chat, so the numbers match; only cycles up to the cutoff are read."""
    ctx = RequestContext(request_id=uuid.uuid4().hex[:12], engine_id=eid, dataset_split=split, cutoff_cycle=cutoff, question="",
                         model_version=system.suite.model_version, corpus_version=system.suite.corpus_version)
    with st.container(border=True):
        st.markdown(f"**RUL estimate as of cycle {cutoff}** (updates as you move the cutoff slider; the model sees only cycles 1 to {cutoff})")
        try:
            out = system.suite.call("predict_rul", ctx, PredictRULInput())
        except Exception as e:
            st.warning(f"Prediction unavailable: {e}. No value has been estimated or made up.")
            return
        cols = st.columns(4)
        cols[0].metric("Predicted RUL (XGBoost)", f"{out.predicted_rul_cycles:.1f} cycles")
        cols[1].metric("Support (Random Forest)", f"{out.support_rul_cycles:.1f} cycles", f"{out.support_rul_cycles - out.predicted_rul_cycles:+.1f}", delta_color="off")
        cols[2].markdown(f"**Demo band**<br><span style='font-size:1.6rem;color:{BAND_COLOUR[out.demo_band]}'>{out.demo_band}</span>", unsafe_allow_html=True)
        try:  # evaluation display only: the true RUL is never given to the model or to the chat agent
            truth = system.suite.store.true_rul_at(split, eid, cutoff)
        except Exception as e:
            truth = None
            st.caption(f"True RUL unavailable: {e}")
        if truth is None:
            cols[3].metric("True RUL at this cycle", "n/a", help="The official label file RUL_FD001.txt was not found.")
        else:
            cols[3].metric("True RUL at this cycle", f"{truth} cycles", f"error {out.predicted_rul_cycles - truth:+.1f}", delta_color="off",
                           help="Error = predicted - true. Positive means the model overestimated the remaining life, the risky direction.")
        for w in out.warnings:
            st.caption(w)
        _alert_notice(system, split, eid, cutoff)
        if split == "test":
            st.caption("The true RUL of a test engine comes from the official label file RUL_FD001.txt: the label is the true RUL at the last recorded cycle, and at an earlier cycle "
                       "it is the label plus the cycles stepped back. It is shown for evaluation only and is never given to the model or the chat agent.")


def _alert_notice(system, split, eid, cutoff):
    """Say what the 80%-of-life alert did for this prediction (the alert itself is raised by the prediction tool, not by this page)."""
    ev = system.alerts.last_event if system.alerts else None
    if ev is None or ev.key != f"{split}:{eid}:{system.suite.model_version}" or ev.cycle != cutoff:
        return
    pct = f"{ev.life_used * 100:.0f}%"
    if ev.status == "alerted":
        st.info(f"Slack alert sent: estimated life used is {pct} (threshold {system.alerts.threshold * 100:.0f}%).")
    elif ev.status == "dry_run":
        st.info(f"Alert raised in dry-run: estimated life used is {pct}. It would be posted to Slack once a webhook URL is configured.")
    elif ev.status == "send_failed":
        st.warning(f"Slack alert could not be sent ({ev.error}). It will be retried on the next prediction.")
    elif ev.status == "already_alerted":
        st.caption(f"Estimated life used is {pct}. An alert was already raised for this engine; it re-arms if the estimate falls below {system.alerts.rearm_below * 100:.0f}%.")


def render_life_panel(system, eid):
    """Retrospective overview of a TRAIN engine's full run to failure; charts are requested from the chat."""
    with st.container(border=True):
        st.markdown("**Life overview of this train engine** (retrospective: reads the full run to failure and ignores the cutoff)")
        try:
            v = system.life.get("train", eid).summary
        except Exception as e:
            st.warning(f"Life overview unavailable: {e}")
            return
        c1, c2, c3 = st.columns(3)
        c1.metric("Total life", f"{v.total_cycles} cycles")
        c2.metric("Data role", ROLE_LABEL.get(v.data_role, v.data_role))
        c3.metric("Top sensor by association", v.ranking[0].sensor if v.ranking else "n/a")
        if v.ranking:
            st.dataframe(pd.DataFrame([{"Rank": i, "Sensor": r.sensor, "Direction": r.direction, "Spearman rho vs cycle": r.spearman_rho,
                                        "Late minus early (x own spread)": r.late_minus_early} for i, r in enumerate(v.ranking[:6], 1)]),
                         hide_index=True)
        top = v.ranking[0].sensor if v.ranking else "sensor_11"
        st.caption("Sensors are ranked by how consistently they rise or fall as this engine ages. That is an observed association, not a cause. "
                   f"To draw the charts, ask in the chat, for example: \"plot {top} and the RUL\".")


def render(m):
    if "resp" not in m:
        st.markdown(m["text"]); return
    r = m["resp"]
    if r.prediction:
        p = r.prediction
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted RUL (XGBoost)", f"{p.predicted_rul_cycles:.1f} cycles")
        c2.metric("Support model (Random Forest)", f"{p.support_rul_cycles:.1f} cycles", f"{p.support_rul_cycles - p.predicted_rul_cycles:+.1f}", delta_color="off")
        c3.markdown(f"**Demo band**<br><span style='font-size:1.6rem;color:{BAND_COLOUR[p.demo_band]}'>{p.demo_band}</span>", unsafe_allow_html=True)
    if r.chart:
        df = pd.DataFrame(r.chart.series, index=r.chart.cycles); df.index.name = "cycle"
        st.line_chart(df, height=220)
        if getattr(r.chart, "mode", "raw") == "change":
            st.caption("Change since the first plotted cycle, in units of each sensor's own spread (unitless).")
    for ch in getattr(r, "life_charts", []) or []:
        st.markdown(f"**{ch.title}**")
        line_chart(ch.cycles, ch.series, 240, ch.zero_baseline)
        if ch.caption:
            st.caption(ch.caption)
    st.markdown(r.answer_markdown)
    if r.citations:
        with st.expander(f"Sources ({len(r.citations)})"):
            for c in r.citations:
                st.markdown(f"**{c.chunk_id}** · {c.title} · {c.section} (rev {c.revision})")
    with st.expander(f"Trace · {r.tool_calls} tool calls · {r.latency_s:.2f}s · {r.agent_mode}"):
        st.caption(f"trace id {r.trace_id}")
        if r.trace:
            st.dataframe(pd.DataFrame([{"seq": e.seq, "agent": e.agent, "tool": e.tool, "status": e.status, "ms": e.duration_ms,
                                        "cache": e.cache_status, "error": e.error} for e in r.trace]), hide_index=True)


main()
