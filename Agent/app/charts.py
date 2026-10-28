"""Altair chart helpers for the dashboard. Sensor values sit far from zero, so most charts fit their y-axis to the data."""
from typing import Optional, Sequence

import altair as alt
import pandas as pd
import streamlit as st


def line_chart(cycles, series, height=240, zero=True):
    """Line chart over cycles. ``zero=False`` fits the y-axis to the data; Streamlit's default starts at zero and flattens sensors far from it."""
    long = pd.DataFrame(series, index=cycles); long.index.name = "cycle"
    long = long.reset_index().melt("cycle", var_name="series", value_name="value")
    ch = (alt.Chart(long).mark_line()
          .encode(x=alt.X("cycle:Q", scale=alt.Scale(domain=[min(cycles), max(cycles)], nice=False)),
                  y=alt.Y("value:Q", scale=alt.Scale(zero=zero), title=None),
                  color=alt.Color("series:N", legend=alt.Legend(orient="bottom", title=None)),
                  tooltip=["cycle:Q", "series:N", alt.Tooltip("value:Q", format=".4g")])
          .properties(height=height).interactive())
    st.altair_chart(ch, use_container_width=True)


def long_line_chart(df: pd.DataFrame, x_title: str, y_title: Optional[str] = None, height: int = 260, reverse_x: bool = False, zero: bool = False):
    """Lines from a long frame with columns x, value, engine (one line per engine)."""
    ch = (alt.Chart(df).mark_line()
          .encode(x=alt.X("x:Q", title=x_title, scale=alt.Scale(reverse=reverse_x, nice=False)),
                  y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=zero)),
                  color=alt.Color("engine:N", legend=alt.Legend(orient="bottom", title="engine")),
                  tooltip=["engine:N", alt.Tooltip("x:Q", title=x_title), alt.Tooltip("value:Q", format=".4g")])
          .properties(height=height).interactive())
    st.altair_chart(ch, use_container_width=True)


def histogram_chart(h: pd.DataFrame, x_title: str, y: str = "count", y_title: str = "rows", color: Optional[str] = None, height: int = 240):
    enc = dict(x=alt.X("start:Q", title=x_title), x2="end:Q", y=alt.Y(f"{y}:Q", title=y_title, stack=None),
               tooltip=[alt.Tooltip("start:Q", title="from", format=".4g"), alt.Tooltip("end:Q", title="to", format=".4g"), alt.Tooltip(f"{y}:Q", format=".4g")])
    if color:
        enc["color"] = alt.Color(f"{color}:N", legend=alt.Legend(orient="bottom", title=None))
        ch = alt.Chart(h).mark_bar(opacity=0.55).encode(**enc)
    else:
        ch = alt.Chart(h).mark_bar().encode(**enc)
    st.altair_chart(ch.properties(height=height), use_container_width=True)


def heatmap_chart(corr: pd.DataFrame, height: int = 520):
    names = list(corr.columns)
    long = corr.stack().rename("correlation").reset_index()
    long.columns = ["a", "b", "correlation"]
    ch = (alt.Chart(long).mark_rect()
          .encode(x=alt.X("a:N", sort=names, title=None), y=alt.Y("b:N", sort=names, title=None),
                  color=alt.Color("correlation:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
                  tooltip=["a:N", "b:N", alt.Tooltip("correlation:Q", format=".3f")])
          .properties(height=height))
    st.altair_chart(ch, use_container_width=True)


def bar_chart(df: pd.DataFrame, x: str, y: str, height: int = 300, sort_by: Optional[Sequence[str]] = None, horizontal: bool = False):
    sort = list(sort_by) if sort_by is not None else None
    if horizontal:
        enc = dict(y=alt.Y(f"{x}:N", sort=sort, title=None), x=alt.X(f"{y}:Q"))
    else:
        enc = dict(x=alt.X(f"{x}:N", sort=sort, title=None), y=alt.Y(f"{y}:Q"))
    st.altair_chart(alt.Chart(df).mark_bar().encode(tooltip=[f"{x}:N", alt.Tooltip(f"{y}:Q", format=".4g")], **enc).properties(height=height), use_container_width=True)


def scatter_chart(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, height: int = 300):
    ch = (alt.Chart(df).mark_circle(size=60, opacity=0.7)
          .encode(x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(zero=False)), y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False)),
                  tooltip=list(df.columns))
          .properties(height=height).interactive())
    st.altair_chart(ch, use_container_width=True)
