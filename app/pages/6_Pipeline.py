"""The clean pipeline: Brief Extract -> Slide Generation, run stage by
stage with the active prompt+model from Pipeline Settings — no pickers
here. The only human decision point mirrors production's actual gate
(confirm which extracted images matter before they're analyzed); every
other stage is run, reviewed, approved-or-edited, then moved past. Trying
alternative prompts/models happens in the Experiment Lab, not here.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import active_config as ac
from core import config
from core import data_sources as ds
from core import db, objectives, ollama_client
from core import runner, stages

st.set_page_config(page_title="Pipeline", layout="wide")
st.title("Pipeline")
st.caption(
    "Runs Brief Extract through Slide Content using whatever is set active in "
    "Pipeline Settings. Review and approve (or edit) each stage's output before it "
    "feeds the next one."
)

if not ollama_client.is_reachable():
    st.error("Ollama is not reachable. Start it (`ollama serve`) and reload this page.")
    st.stop()

ORDER = stages.ordered_stage_ids()
missing_config = [sid for sid in ORDER if ac.get_active(sid) is None]
if missing_config:
    st.warning(
        f"No active prompt/model set for: {', '.join(missing_config)}. "
        f"Set them on the **Pipeline Settings** page first."
    )
    st.stop()

if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = {"started": False}
state = st.session_state.pipeline_state

with st.sidebar:
    st.subheader("New run")
    doc_ids = [None] + config.list_document_ids()
    doc_id = st.selectbox(
        "Document", doc_ids,
        format_func=lambda d: ds.document_filename(d) + (" (original)" if d is None else ""),
    )
    obj_list = objectives.list_objectives()
    obj_choice = st.selectbox("Objective (optional)", ["(none)"] + [o.title for o in obj_list])
    objective_id = None if obj_choice == "(none)" else next(o.id for o in obj_list if o.title == obj_choice)
    pages = ds.list_pages(doc_id)
    page_numbers = st.multiselect(
        "Pages in this document", [p.page_number for p in pages],
        default=[p.page_number for p in pages],
    )
    if st.button("Start new pipeline run", type="primary"):
        chain_run_id = db.insert_chain_run(objective_id, "pipeline run",
                                            f"{len(page_numbers)} pages: {sorted(page_numbers)}")
        st.session_state.pipeline_state = {
            "started": True, "chain_run_id": chain_run_id, "page_numbers": page_numbers,
            "document_id": doc_id,
            "objective_id": objective_id, "gate_done": False, "selected_images": [],
            "cursor": 0, "approved": {}, "history": {},
        }
        st.rerun()
    if state.get("started") and st.button("Abandon this run"):
        st.session_state.pipeline_state = {"started": False}
        st.rerun()

if not state.get("started"):
    st.info("Pick pages in the sidebar and click **Start new pipeline run**.")
    st.stop()

st.caption(f"chain_run_id: `{state['chain_run_id']}`")


def active_prompt_and_model(stage_id: str):
    cfg = ac.get_active(stage_id)
    from core.prompts import get_prompt
    return get_prompt(stage_id, cfg.prompt_name), cfg.model


# --- Step 0: human image review gate (mirrors production Step 3) ----------
if not state["gate_done"]:
    st.subheader("Review images before analysis")
    st.caption(
        "Production pauses here for a human to confirm which extracted images are "
        "worth analyzing. Uncheck anything irrelevant (logos, dividers, noise)."
    )
    page_images = [i for n in state["page_numbers"] for i in ds.images_on_page(n, state["document_id"])]
    if not page_images:
        st.info("No images on the selected pages.")
        state["selected_images"] = []
    else:
        cols = st.columns(4)
        selected = []
        for idx, img in enumerate(page_images):
            with cols[idx % 4]:
                st.image(str(img.image_path), caption=f"p{img.page_number} {img.filename}", use_container_width=True)
                if st.checkbox("include", value=True, key=f"gate-{img.filename}"):
                    selected.append(img.filename)
        state["selected_images"] = selected
    if st.button("Confirm images and start analysis", type="primary"):
        state["gate_done"] = True
        st.rerun()
    st.stop()

# --- Progress strip ---------------------------------------------------------
cols = st.columns(len(ORDER))
for i, sid in enumerate(ORDER):
    sdef = stages.STAGES_BY_ID[sid]
    with cols[i]:
        if sid in state["approved"]:
            st.markdown(f"✅ **{sdef.order}**  \napproved")
        elif i == state["cursor"]:
            st.markdown(f"▶️ **{sdef.order}**  \ncurrent")
        else:
            st.markdown(f"⏳ **{sdef.order}**")
st.divider()

if state["cursor"] >= len(ORDER):
    st.success("Pipeline complete.")
    for sid in ORDER:
        st.markdown(f"**{stages.STAGES_BY_ID[sid].name}**")
        st.json(state["approved"][sid])
    st.stop()

current_sid = ORDER[state["cursor"]]
current_def = stages.STAGES_BY_ID[current_sid]
prompt, model = active_prompt_and_model(current_sid)
st.subheader(f"Stage {current_def.order}: {current_def.name}")
st.caption(f"Using: **{prompt.display_name}** on **{model}** — change in Pipeline Settings.")

history = state["history"].setdefault(current_sid, [])


def upstream_context() -> tuple[dict, str]:
    for sid in reversed(ORDER[:state["cursor"]]):
        if sid in state["approved"]:
            return state["approved"][sid], sid
    return {}, "raw pages"


def raw_pages_text() -> str:
    pages = [ds.get_page(n, state["document_id"]) for n in state["page_numbers"]]
    return "\n\n".join(ds.page_context_full(p) for p in pages if p)


run_label = "Re-run this stage" if history else "Run this stage"
if st.button(run_label, type="primary"):
    with st.spinner(f"Running {current_def.name}…"):
        if current_def.input_kind == "image":
            images = [i for n in state["page_numbers"] for i in ds.images_on_page(n, state["document_id"])
                      if i.filename in state["selected_images"]]
            batch = []
            for img in images:
                result = runner.run_stage(
                    current_sid, prompt, model, {"CONTEXT": ds.image_prompt_context(img)},
                    temperature=ac.PIPELINE_TEMPERATURE, response_format=ac.PIPELINE_RESPONSE_FORMAT,
                    images=[img.image_path], input_ref=img.ref, input_summary=f"page {img.page_number}",
                    context_variant="image_metadata", objective_id=state["objective_id"],
                    chain_run_id=state["chain_run_id"], chain_sequence=state["cursor"],
                )
                batch.append(result)
            history.append(batch)
        elif current_def.input_kind == "page":
            batch = []
            for n in state["page_numbers"]:
                page = ds.get_page(n, state["document_id"])
                page_image_refs = {i.ref for i in ds.images_on_page(n, state["document_id"])}
                approved_images = state["approved"].get("image_analysis", {})
                image_evidence = json.dumps([
                    v for ref, v in approved_images.items() if ref in page_image_refs
                ]) if isinstance(approved_images, dict) else "[]"
                result = runner.run_stage(
                    current_sid, prompt, model,
                    {"CONTEXT": ds.page_context_full(page), "IMAGE_EVIDENCE": image_evidence},
                    temperature=ac.PIPELINE_TEMPERATURE, response_format=ac.PIPELINE_RESPONSE_FORMAT,
                    input_ref=page.ref, input_summary=page.heading or f"page {n}",
                    context_variant="full", objective_id=state["objective_id"],
                    chain_run_id=state["chain_run_id"], chain_sequence=state["cursor"],
                )
                batch.append(result)
            history.append(batch)
        else:
            ctx, source_label = upstream_context()
            context_str = json.dumps(ctx) if ctx else raw_pages_text()
            if current_sid == "slide_content" and isinstance(ctx, dict) and "slides" in ctx:
                batch = []
                for slide in ctx.get("slides", [])[:8]:
                    result = runner.run_stage(
                        current_sid, prompt, model,
                        {"CONTEXT": json.dumps({"slide": slide, "narrative": state["approved"].get("narrative", {})})},
                        temperature=ac.PIPELINE_TEMPERATURE, response_format=ac.PIPELINE_RESPONSE_FORMAT,
                        input_ref=f"slide:{slide.get('title', '')[:30]}", input_summary=slide.get("title", ""),
                        context_variant="slide_plan_item", objective_id=state["objective_id"],
                        chain_run_id=state["chain_run_id"], chain_sequence=state["cursor"],
                    )
                    batch.append(result)
                history.append(batch)
            else:
                result = runner.run_stage(
                    current_sid, prompt, model, {"CONTEXT": context_str},
                    temperature=ac.PIPELINE_TEMPERATURE, response_format=ac.PIPELINE_RESPONSE_FORMAT,
                    input_ref=f"pages:{sorted(state['page_numbers'])}", input_summary=f"source={source_label}",
                    context_variant=source_label, objective_id=state["objective_id"],
                    chain_run_id=state["chain_run_id"], chain_sequence=state["cursor"],
                )
                history.append([result])
    st.rerun()

if history:
    latest_batch = history[-1]
    st.markdown(f"**Output** ({len(latest_batch)} item(s))" + (f" — {len(history) - 1} earlier attempt(s)" if len(history) > 1 else ""))
    all_valid = all(r.parsed_json is not None for r in latest_batch)
    edited_values = {}
    for r in latest_batch:
        if r.error:
            st.error(r.error)
        if r.schema_errors:
            st.warning(" / ".join(r.schema_errors))
        label = r.input_ref if len(latest_batch) > 1 else "Output"
        with st.expander(label, expanded=len(latest_batch) == 1):
            text = st.text_area(
                "Edit before approving if needed",
                json.dumps(r.parsed_json, indent=2) if r.parsed_json is not None else r.output_text,
                height=220, key=f"edit-{current_sid}-{len(history)}-{r.run_id}",
                label_visibility="collapsed",
            )
            edited_values[r.input_ref] = text

    if st.button("Approve and continue", type="primary", disabled=not all_valid):
        try:
            parsed = {ref: json.loads(text) for ref, text in edited_values.items()}
            if len(latest_batch) == 1:
                parsed = next(iter(parsed.values()))
            state["approved"][current_sid] = parsed
            state["cursor"] += 1
            st.rerun()
        except json.JSONDecodeError as exc:
            st.error(f"Not valid JSON, can't approve: {exc}")
else:
    st.info(f"Click **{run_label}** to produce this stage's output.")
