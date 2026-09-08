import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from core import config, results_log

st.set_page_config(page_title="Results Log", layout="wide")
st.title("Results Log")
st.caption(
    f"Every run ever executed, in one flat table — the raw material for your report. "
    f"Backed by `{results_log.LOG_PATH.relative_to(config.REPO_ROOT)}`, which you can also "
    f"open directly in Excel/Sheets without this app."
)

if not results_log.LOG_PATH.exists():
    st.info("No runs logged yet.")
    st.stop()

df = pd.read_csv(results_log.LOG_PATH, encoding="utf-8")

col1, col2, col3, col4 = st.columns(4)
with col1:
    stage_filter = st.multiselect("Stage", sorted(df["stage"].unique()))
with col2:
    model_filter = st.multiselect("Model", sorted(df["model"].unique()))
with col3:
    prompt_filter = st.multiselect("Prompt", sorted(df["prompt_name"].unique()))
with col4:
    valid_only = st.checkbox("Valid JSON only")

filtered = df.copy()
if stage_filter:
    filtered = filtered[filtered["stage"].isin(stage_filter)]
if model_filter:
    filtered = filtered[filtered["model"].isin(model_filter)]
if prompt_filter:
    filtered = filtered[filtered["prompt_name"].isin(prompt_filter)]
if valid_only:
    filtered = filtered[filtered["is_valid_json"] == True]  # noqa: E712

st.markdown(f"**{len(filtered)} / {len(df)} rows**")
st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered rows as CSV",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="results_log_filtered.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Quick rollup: validity rate & latency by model × stage")
if len(filtered):
    rollup = filtered.groupby(["stage", "model"]).agg(
        runs=("run_id", "count"),
        valid_rate=("is_valid_json", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
    ).reset_index()
    rollup["valid_rate"] = (rollup["valid_rate"] * 100).round(1)
    rollup["avg_latency_ms"] = rollup["avg_latency_ms"].round(0)
    st.dataframe(rollup, use_container_width=True, hide_index=True)
