"""Fleet scoring view: import readings for many engines, estimate each engine's RUL with the model, and flag those at or above 80% of life."""
import hashlib

import pandas as pd
import streamlit as st

from Agent.app import charts
from Agent.tools import explore
from Agent.tools.fleet_upload import UploadError, parse_labels, parse_upload, score_fleet

SOURCES = ["Upload a file", "Official test engines (100)"]
HELP = ("Import readings for many engines in the FD001 format: a CSV with a header row (engine_id, cycle, setting_1..3, sensor_1..21), or 26 whitespace-separated "
        "columns without a header like test_FD001. The model scores the **last cycle of each engine**. Try `data/simulated/batch1.csv`, `batch2.csv` or `batch3.csv`.")


def render_fleet(system):
    st.markdown("**Fleet scoring**: estimate the RUL of every engine in a batch and flag the ones at or above the alert threshold of estimated life used.")
    source = st.radio("Readings", SOURCES, horizontal=True, key="fleet_source")
    try:
        loaded = _load_official(system) if source == SOURCES[1] else _load_upload()
    except UploadError as e:
        st.error(str(e))
        return
    if loaded is None:
        st.caption(HELP)
        return
    df, name, fingerprint, labels, warnings = loaded
    for w in warnings:
        st.caption(w)
    th = system.cfg["thresholds"]
    threshold = system.alerts.threshold if system.alerts else float(system.cfg.get("alerts", {}).get("life_used_threshold", 0.80))
    key = ("fleet", fingerprint, system.suite.model_version, threshold, None if labels is None else tuple(labels.to_numpy().tolist()))
    if st.session_state.get("fleet_key") != key:
        with st.spinner(f"Scoring {df['engine_id'].nunique()} engines..."):
            st.session_state["fleet_result"] = score_fleet(df, system.suite.predictor, th, threshold, labels)
            st.session_state["fleet_key"] = key
    res = st.session_state["fleet_result"]
    _summary(res, name, threshold)
    _results(res, threshold)
    _slack(system, res, name, fingerprint)


def _load_official(system):
    store = system.suite.store
    df = store.frame("test").copy()
    notes = ["The official test engines were never used to fit the models; true RUL labels are used for the evaluation table only."]
    try:
        labels = store.rul_labels()
    except Exception as e:  # a missing or misaligned label file should not stop the scoring
        labels, notes = None, [f"True RUL labels unavailable ({e}); scoring without the evaluation table."]
    return df, "official test engines", "official-test", labels, notes


def _load_upload():
    up = st.file_uploader("Fleet readings (.csv or .txt)", type=["csv", "txt"], key="fleet_upload")
    lab = st.file_uploader("True RUL labels (optional, for evaluation only)", type=["csv", "txt"], key="fleet_labels")
    if up is None:
        return None
    data = up.getvalue()
    df, warnings = parse_upload(data, up.name)
    labels = parse_labels(lab.getvalue(), df["engine_id"].unique(), lab.name) if lab is not None else None
    return df, up.name, hashlib.sha256(data).hexdigest()[:16], labels, warnings


def _summary(res, name, threshold):
    a = st.columns(4)
    a[0].metric("Engines scored", len(res.table))
    a[1].metric(f"At or above {threshold * 100:.0f}% life used", res.flagged)
    a[2].metric("Rejected engines", len(res.rejected))
    a[3].metric("Source", name)
    if res.rejected:
        with st.expander(f"{len(res.rejected)} engine(s) rejected: the rest were still scored"):
            st.dataframe(pd.DataFrame(res.rejected, columns=["engine_id", "reason"]), hide_index=True)
    if res.table.empty:
        st.warning("No engine could be scored.")


def _results(res, threshold):
    if res.table.empty:
        return
    st.markdown("**Results, most life used first**")
    view = res.table.copy()
    styled = view.style.apply(lambda r: ["background-color: rgba(255, 99, 71, 0.22)" if r["flagged"] else "" for _ in r], axis=1)
    st.dataframe(styled, hide_index=True)
    st.download_button("Download results (CSV)", res.table.to_csv(index=False).encode("utf-8"), file_name="fleet_scoring_results.csv", mime="text/csv",
                       key="fleet_dl")
    st.markdown("**Estimated life used across the fleet**")
    charts.histogram_chart(explore.histogram(res.table["life_used_pct"], 20, (0, 100)), "estimated life used (%)", y_title="engines")
    st.caption(f"Estimated life used = cycle / (cycle + predicted RUL). Engines at or above {threshold * 100:.0f}% are flagged for an inspection review (demo, not a "
               "maintenance decision). The model typically errs by about 15 cycles at an engine's last cycle and about 30 at arbitrary cycles.")
    if res.evaluation:
        _evaluation(res.evaluation, res.table, threshold)


def _evaluation(ev, table, threshold):
    st.markdown("**Evaluation against the labels you supplied** (never given to the model or to Slack)")
    a = st.columns(4)
    a[0].metric("MAE", f"{ev['mae']:.1f} cycles")
    a[1].metric("Mean error", f"{ev['mean_signed_error']:+.1f}", help="Predicted minus true. Positive means overestimated remaining life.")
    a[2].metric("Alert precision", "n/a" if ev["precision"] is None else f"{ev['precision'] * 100:.0f}%", help="Of the engines flagged, the share truly at or above the threshold.")
    a[3].metric("Alert recall", "n/a" if ev["recall"] is None else f"{ev['recall'] * 100:.0f}%", help="Of the engines truly at or above the threshold, the share flagged.")
    st.caption(f"Flagged and truly at or above {threshold * 100:.0f}%: {ev['true_positive']} · flagged but not truly: {ev['false_positive']} · missed: "
               f"{ev['false_negative']} · correctly not flagged: {ev['true_negative']}.")
    charts.scatter_chart(table[["engine_id", "true_rul", "predicted_rul"]], "true_rul", "predicted_rul", "true RUL (cycles)", "predicted RUL (cycles)")


def _slack(system, res, name, fingerprint):
    st.markdown("**Slack summary**")
    if system.alerts is None:
        st.caption("Slack alerts are off (alerts.enabled in configs/agent.yaml).")
        return
    st.caption(system.alerts.status_text())
    flagged = res.table[res.table["flagged"]]
    if flagged.empty:
        st.success(f"No engine is at or above {system.alerts.threshold * 100:.0f}% estimated life used, so nothing is sent.")
        return
    rows = flagged[["engine_id", "last_cycle", "life_used_pct", "predicted_rul", "band"]].to_dict("records")
    auto = st.checkbox("Post one summary to Slack when scoring finishes (once per file)", value=True, key="fleet_auto")
    force = st.button("Send the summary again", key="fleet_resend")
    if not (auto or force):
        return
    ev = system.alerts.send_fleet_summary(fingerprint, name, rows, len(res.table), res.model_version, force=force)
    if ev.status == "alerted":
        st.info(f"Slack summary sent for {len(rows)} engine(s).")
    elif ev.status == "dry_run":
        st.info(f"Dry run: a summary for {len(rows)} engine(s) would be posted once a webhook URL is configured.")
        with st.expander("Message that would be sent"):
            st.markdown(ev.message)
    elif ev.status == "send_failed":
        st.warning(f"The Slack summary could not be sent ({ev.error}). Use 'Send the summary again' to retry.")
    elif ev.status == "already_alerted":
        st.caption("A summary for this file was already sent. Use 'Send the summary again' to post it again.")
