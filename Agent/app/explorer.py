"""Data explorer view: descriptive and exploratory analysis of the three FD001 files (train, test, official RUL labels)."""
import pandas as pd
import streamlit as st

from Agent.app import charts
from Agent.tools import explore
from src.schema import NUMERIC_COLS, SENSOR_COLS, SETTING_COLS

DATASETS = {
    "train_FD001 · run to failure": "train",
    "test_FD001 · truncated histories": "test",
    "RUL_FD001 · true RUL of the test engines": "rul",
}
ABOUT = {
    "train": "Each engine runs until it fails, so its last cycle is its total life and RUL = last cycle - current cycle. The agent's models were fitted on engines from this file.",
    "test": "Each engine's history stops before failure. The true remaining life of each engine at its last cycle is in RUL_FD001, not in this file.",
}


def render_explorer(system):
    st.markdown("**Data explorer**: descriptive and exploratory analysis of the three FD001 files.")
    choice = st.radio("Dataset", list(DATASETS), horizontal=True, key="explorer_dataset")
    kind = DATASETS[choice]
    store = system.suite.store
    try:
        if kind == "rul":
            _render_labels(system)
        else:
            _render_frame(store.frame(kind), kind)
    except Exception as e:  # a missing or malformed file should say so plainly, not crash the page
        st.error(f"Could not analyse {choice}: {e}")


# ---------------------------------------------------------------- train / test frames
def _render_frame(df: pd.DataFrame, kind: str):
    is_train = kind == "train"
    st.caption(ABOUT[kind])
    names = ["Overview", "Statistics", "Distributions", "Trends", "Correlations", "Data quality"]
    t_over, t_stats, t_dist, t_trend, t_corr, t_qual = st.tabs(names)
    with t_over:
        _overview(df, kind)
    with t_stats:
        _statistics(df, kind, is_train)
    with t_dist:
        _distributions(df, kind, is_train)
    with t_trend:
        _trends(df, kind, is_train)
    with t_corr:
        _correlations(df, kind, is_train)
    with t_qual:
        _quality(df, kind)


def _overview(df, kind):
    o = explore.overview(df)
    a = st.columns(4)
    a[0].metric("Rows", f"{o['rows']:,}")
    a[1].metric("Engines", o["engines"])
    a[2].metric("Columns", o["columns"])
    a[3].metric("Missing cells", o["missing_cells"])
    what = "Total life" if kind == "train" else "Observed cycles"
    b = st.columns(4)
    b[0].metric(f"{what}: shortest", o["cycles_min"])
    b[1].metric(f"{what}: median", f"{o['cycles_median']:.0f}")
    b[2].metric(f"{what}: mean", f"{o['cycles_mean']:.1f}")
    b[3].metric(f"{what}: longest", o["cycles_max"])
    st.markdown("**Cycles per engine**" + (" (total life)" if kind == "train" else " (observed history; the engines have not failed yet)"))
    bins = st.slider("Histogram bins", 5, 60, 20, key=f"{kind}_life_bins")
    life = explore.lifetimes(df)
    charts.histogram_chart(explore.histogram(life, bins), "cycles per engine", y_title="engines")
    if o["constant_columns"]:
        st.caption(f"Constant columns (no variation, so no information): {', '.join(o['constant_columns'])}.")
    else:
        st.caption("No constant columns.")
    st.caption(f"Duplicate (engine, cycle) rows: {o['duplicate_keys']}.")
    st.markdown("**First rows**")
    st.dataframe(df.head(25), hide_index=True)
    with st.expander("Column list"):
        st.dataframe(pd.DataFrame({"column": df.columns, "type": [str(t) for t in df.dtypes]}), hide_index=True)


def _statistics(df, kind, is_train):
    group = st.radio("Columns", ["Sensors", "Operating settings", "All"], horizontal=True, key=f"{kind}_stat_group")
    cols = {"Sensors": SENSOR_COLS, "Operating settings": SETTING_COLS, "All": NUMERIC_COLS}[group]
    table = explore.describe_table(df, cols)
    st.dataframe(table, hide_index=True)
    st.download_button("Download this table (CSV)", table.to_csv(index=False).encode("utf-8"), file_name=f"{kind}_FD001_summary_statistics.csv", mime="text/csv",
                       key=f"{kind}_stat_dl")
    st.caption("Constant columns (std = 0) carry no information. Skew near 0 means a symmetric distribution.")
    if is_train:
        st.markdown("**Early vs late life** (mean in the last 20% of each engine's life minus the first 20%, in units of the column's overall std)")
        shift = explore.early_late_shift(df, [c for c in cols if c not in ("cycle",)])
        st.dataframe(shift, hide_index=True)
        charts.bar_chart(shift, "column", "shift (std units)", sort_by=list(shift["column"]), height=280)
        st.caption("A large shift means the column moves systematically as the engine ages. This is an observed association with life consumed, not a cause.")


def _distributions(df, kind, is_train):
    col = st.selectbox("Column", NUMERIC_COLS, index=NUMERIC_COLS.index("sensor_11") if "sensor_11" in NUMERIC_COLS else 0, key=f"{kind}_dist_col")
    bins = st.slider("Bins", 10, 80, 30, key=f"{kind}_dist_bins")
    compare = is_train and st.checkbox("Compare early life (first 20%) with late life (last 20%)", key=f"{kind}_dist_phase")
    if compare:
        charts.histogram_chart(explore.phase_histograms(df, col, bins), col, y="percent", y_title="% of rows in the phase", color="phase")
        st.caption("Separated humps mean the sensor's values shift as the engine degrades; overlapping humps mean it barely changes.")
    else:
        charts.histogram_chart(explore.histogram(df[col], bins), col)
    st.dataframe(df[[col]].describe().T, hide_index=False)


def _trends(df, kind, is_train):
    engines = sorted(int(e) for e in df["engine_id"].unique())
    default_sensors = [s for s in ("sensor_11", "sensor_4") if s in SENSOR_COLS]
    sel_e = st.multiselect("Engines (up to 5)", engines, default=engines[:1], max_selections=5, key=f"{kind}_trend_engines")
    sel_s = st.multiselect("Sensors (up to 4)", SENSOR_COLS, default=default_sensors, max_selections=4, key=f"{kind}_trend_sensors")
    c1, c2 = st.columns(2)
    x_choice = c1.radio("X axis", ["cycle", "remaining life (RUL)"] if is_train else ["cycle"], horizontal=True, key=f"{kind}_trend_x")
    standardise = c2.checkbox("Standardise (z-score, so sensors share a scale)", value=len(sel_s) > 1, key=f"{kind}_trend_std")
    if not sel_e or not sel_s:
        st.info("Pick at least one engine and one sensor.")
        return
    use_rul = x_choice != "cycle"
    frame = explore.trend_frame(df, sel_e, sel_s, "rul" if use_rul else "cycle", standardise)
    for s in sel_s:
        st.markdown(f"**{s}**" + (" (standardised)" if standardise else ""))
        charts.long_line_chart(frame[frame["sensor"] == s], "remaining life (cycles, time runs left to right)" if use_rul else "cycle", reverse_x=use_rul)
    if is_train:
        st.caption("With the RUL axis every engine ends at 0 on the right, which lines the engines up by how close to failure they are.")


def _correlations(df, kind, is_train):
    c1, c2 = st.columns(2)
    method = c1.radio("Method", ["pearson", "spearman"], horizontal=True, key=f"{kind}_corr_method")
    with_settings = c2.checkbox("Include operating settings", value=False, key=f"{kind}_corr_settings")
    cols = (SETTING_COLS if with_settings else []) + SENSOR_COLS
    corr = explore.correlation_matrix(df, cols, method)
    st.markdown(f"**{method.capitalize()} correlation matrix** (constant columns are left out)")
    charts.heatmap_chart(corr)
    thr = st.slider("Show pairs with |correlation| at least", 0.5, 0.99, 0.9, 0.01, key=f"{kind}_corr_thr")
    pairs = explore.correlation_pairs(corr, thr)
    st.markdown(f"**Strongly related pairs** ({len(pairs)})")
    st.dataframe(pairs, hide_index=True)
    st.caption("Highly correlated sensors carry overlapping information, which is one reason a model can use fewer sensors.")
    if is_train:
        st.markdown("**Correlation of each column with RUL** (Spearman, over every row of the train file)")
        r = explore.rul_correlations(explore.add_rul(df), cols, "spearman")
        charts.bar_chart(r, "column", "correlation with RUL", sort_by=list(r["column"]), height=300)
        st.caption("A strong correlation means the column tracks remaining life across engines. It is an observed association, not a cause.")


def _quality(df, kind):
    o = explore.overview(df)
    problems = []
    if o["missing_cells"]:
        problems.append(f"{o['missing_cells']} missing cells")
    if o["duplicate_keys"]:
        problems.append(f"{o['duplicate_keys']} duplicate (engine, cycle) rows")
    gaps = explore.cycle_gap_table(df)
    if len(gaps):
        problems.append(f"{len(gaps)} engines with cycle gaps")
    if problems:
        st.warning("Found: " + "; ".join(problems) + ".")
    else:
        st.success("No missing cells, no duplicate (engine, cycle) rows and no cycle gaps.")
    st.markdown("**Per column**")
    st.dataframe(explore.quality_table(df), hide_index=True)
    st.caption("Outliers use the 1.5 x IQR rule. In simulated sensor data they are often noise, not faults; they are not removed here.")
    if len(gaps):
        st.markdown("**Engines with cycle gaps**")
        st.dataframe(gaps, hide_index=True)


# ---------------------------------------------------------------- RUL label file
def _render_labels(system):
    store = system.suite.store
    rul = store.rul_labels()
    if rul is None:
        st.info("RUL_FD001.txt was not found in the data folder, so the label file cannot be analysed.")
        return
    st.caption("The official true remaining useful life of each test engine at its last observed cycle (one value per engine, in engine order).")
    joined = explore.label_join(store.frame("test"), rul)
    t_over, t_dist, t_join = st.tabs(["Overview", "Distribution", "Joined with test_FD001"])
    with t_over:
        a = st.columns(3)
        a[0].metric("Labels", len(rul))
        a[1].metric("Smallest", int(rul.min()))
        a[2].metric("Largest", int(rul.max()))
        b = st.columns(3)
        b[0].metric("Median", f"{rul.median():.0f}")
        b[1].metric("Mean", f"{rul.mean():.1f}")
        b[2].metric("Std", f"{rul.std(ddof=0):.1f}")
        st.markdown("**True RUL by test engine**")
        df = rul.reset_index()
        df.columns = ["engine_id", "true RUL"]
        charts.bar_chart(df, "engine_id", "true RUL", sort_by=list(df["engine_id"]), height=260)
        st.dataframe(df, hide_index=True)
    with t_dist:
        bins = st.slider("Histogram bins", 5, 40, 15, key="rul_bins")
        charts.histogram_chart(explore.histogram(rul, bins), "true RUL at the last observed cycle (cycles)", y_title="test engines")
        th = system.cfg["thresholds"]
        counts = explore.band_counts(rul, th["band_normal_above"], th["band_watch_above"])
        st.markdown("**Test engines by demo band at their last cycle** (illustrative bands, not aviation limits)")
        st.dataframe(pd.DataFrame({"band": list(counts), "engines": list(counts.values())}), hide_index=True)
        st.dataframe(rul.describe().to_frame("true RUL").T, hide_index=True)
    with t_join:
        st.markdown("**Observed cycles + true RUL = implied total life** of each test engine")
        st.dataframe(joined, hide_index=True)
        charts.scatter_chart(joined, "observed_cycles", "rul_label", "observed cycles", "true RUL at the last cycle")
        st.caption("Each point is a test engine. Engines observed for many cycles do not necessarily have little life left, because the files were truncated at different points.")
        life_train = explore.lifetimes(store.frame("train"))
        lo, hi = float(min(life_train.min(), joined["implied_total_life"].min())), float(max(life_train.max(), joined["implied_total_life"].max()))
        parts = []
        for name, values in (("train total life", life_train), ("test implied total life", joined["implied_total_life"])):
            h = explore.histogram(values, 20, (lo, hi))
            h["percent"] = 100.0 * h["count"] / max(1, len(values))
            h["group"] = name
            parts.append(h)
        st.markdown("**Total life: train engines vs test engines (implied)**")
        charts.histogram_chart(pd.concat(parts, ignore_index=True), "total life (cycles)", y="percent", y_title="% of engines", color="group")
        st.caption("If the two overlap, the test engines were drawn from a similar life distribution as the training engines.")
