import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import data_sources as ds
from core import db, stages

st.set_page_config(page_title="Compare", layout="wide")
st.title("Compare & Score")
st.caption(
    "Pick a stage and an input to see every run against it side by side, with the "
    "actual source page(s) alongside so you can judge accuracy and hallucinations "
    "against the real document — not against another model's guess. Scores feed the "
    "leaderboard at the bottom."
)

stage_ids = [s.id for s in stages.STAGES]
stage_labels = {s.id: f"{s.order}. {s.name}" for s in stages.STAGES}
stage_id = st.selectbox("Stage", stage_ids, format_func=lambda s: stage_labels[s])

all_runs = db.list_runs(stage=stage_id)
input_refs = sorted({r["input_ref"] for r in all_runs})
if not input_refs:
    st.info("No runs logged yet for this stage. Go run some experiments first.")
    st.stop()

input_ref = st.selectbox("Input", input_refs)
runs = [r for r in all_runs if r["input_ref"] == input_ref]


def render_source(ref: str) -> None:
    """Shows the actual source material (page image + text) behind an
    input_ref, so scoring happens against the real document, not memory."""
    base, document_id = ds.split_ref(ref)
    if base.startswith("page:"):
        page = ds.get_page(int(base.split(":", 1)[1]), document_id)
        if page:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(str(page.image_path), caption=f"page {page.page_number}", use_container_width=True)
            with c2:
                if page.heading:
                    st.markdown(f"**Heading:** {page.heading}")
                st.text(page.prose_text[:2000] or "(no extracted text)")
            return
    if base.startswith("image:"):
        image = ds.get_image(base.split(":", 1)[1], document_id)
        if image:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(str(image.image_path), caption=image.filename, use_container_width=True)
            with c2:
                st.markdown(f"**Page {image.page_number} heading:** {image.nearest_heading or '(none)'}")
                st.text(image.nearby_text[:1000] or "(no nearby text)")
            return
    if base.startswith("pages:"):
        try:
            page_numbers = ast.literal_eval(base.split(":", 1)[1])
        except (ValueError, SyntaxError):
            page_numbers = []
        if page_numbers:
            st.caption(f"Source pages: {page_numbers}")
            thumb_cols = st.columns(min(8, len(page_numbers)))
            for i, n in enumerate(page_numbers):
                page = ds.get_page(n, document_id)
                if page:
                    with thumb_cols[i % len(thumb_cols)]:
                        st.image(str(page.image_path), caption=f"p{n}", use_container_width=True)
            with st.expander("Full extracted text for these pages"):
                for n in page_numbers:
                    page = ds.get_page(n, document_id)
                    if page:
                        st.markdown(f"**Page {n}**" + (f" — {page.heading}" if page.heading else ""))
                        st.text(page.prose_text[:1500] or "(no text)")
            return
    st.caption(f"No direct source view for `{ref}` — this input is itself derived from a prior stage's output; "
               f"open that stage's Compare view to trace it back to a page/image.")


with st.expander("Source material", expanded=True):
    render_source(input_ref)

st.markdown(f"**{len(runs)} run(s)** for `{input_ref}`")

cols = st.columns(min(3, len(runs)) or 1)
for i, run in enumerate(runs):
    with cols[i % len(cols)]:
        badge = "✅" if run["is_valid_json"] else "❌"
        st.markdown(f"**{badge} {run['prompt_name']} · {run['model']}**")
        st.caption(f"T={run['temperature']} · {(run['latency_ms'] or 0):.0f}ms")
        if run["error"]:
            st.error(run["error"])
        if run["parsed_json"]:
            st.json(json.loads(run["parsed_json"]))
        else:
            st.text((run["raw_output"] or "")[:500])

        existing_score = db.get_score(run["id"])
        with st.form(f"score-{run['id']}"):
            st.markdown("_Score this run (1-5)_")
            accuracy = st.slider("Accuracy", 1, 5, existing_score["accuracy"] if existing_score else 3, key=f"acc-{run['id']}")
            completeness = st.slider("Completeness", 1, 5, existing_score["completeness"] if existing_score else 3, key=f"comp-{run['id']}")
            hallucination = st.slider("Hallucination (5 = none)", 1, 5, existing_score["hallucination"] if existing_score else 3, key=f"hal-{run['id']}")
            structure = st.slider("Structure / usable JSON", 1, 5, existing_score["structure"] if existing_score else (5 if run["is_valid_json"] else 1), key=f"struct-{run['id']}")
            overall = st.slider("Overall", 1, 5, existing_score["overall"] if existing_score else 3, key=f"ovr-{run['id']}")
            notes = st.text_input("Notes", existing_score["notes"] if existing_score else "", key=f"notes-{run['id']}")
            if st.form_submit_button("Save score"):
                db.upsert_score(run["id"], accuracy, completeness, hallucination, structure, overall, notes)
                st.success("Saved.")

st.divider()
st.subheader("Leaderboard for this stage")
scored = db.scores_for_stage(stage_id)
if not scored:
    st.info("No scores yet.")
else:
    from collections import defaultdict
    groups = defaultdict(list)
    for row in scored:
        key = (row["prompt_name"], row["model"])
        groups[key].append(row["overall"])
    leaderboard = sorted(
        ({"prompt": k[0], "model": k[1], "avg_overall": round(sum(v) / len(v), 2), "n": len(v)}
         for k, v in groups.items()),
        key=lambda r: r["avg_overall"], reverse=True,
    )
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)
