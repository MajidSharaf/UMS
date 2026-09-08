import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import objectives

st.set_page_config(page_title="Objectives", layout="wide")
st.title("Objectives")
st.caption("Note what a batch of experiments is trying to find out. Purely for your own "
           "reference — objectives can be attached to runs so you can filter results later.")

with st.form("new_objective", clear_on_submit=True):
    title = st.text_input("Title", placeholder="e.g. Does schema-constrained decoding beat freeform JSON on project_brief?")
    description = st.text_area("Description / hypothesis", height=100)
    submitted = st.form_submit_button("Save objective")
    if submitted and title.strip():
        objectives.save_objective(title.strip(), description.strip())
        st.success("Saved.")
        st.rerun()

st.divider()

items = objectives.list_objectives()
if not items:
    st.info("No objectives yet.")
for obj in items:
    with st.expander(obj.title):
        st.write(obj.description or "_no description_")
        st.caption(f"id: {obj.id}")
