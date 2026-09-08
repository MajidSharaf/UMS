import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from core import config, data_sources as ds, db, ollama_client, seed_prompts, stages

st.set_page_config(page_title="Pipeline Experiment Lab", layout="wide")

db.init_db()
seed_prompts.seed_all()

st.title("Pipeline Experiment Lab")
st.caption(
    "Local experimentation harness for the UMS document-intelligence pipeline. "
    "Every run goes through your local Ollama — nothing leaves this machine."
)

col1, col2, col3 = st.columns(3)

reachable = ollama_client.is_reachable()
with col1:
    st.metric("Ollama", "reachable" if reachable else "not reachable",
               help=f"Checked {config.OLLAMA_HOST}/api/tags")
with col2:
    st.metric("Pages loaded", len(ds.list_pages()))
with col3:
    st.metric("Images loaded", len(ds.list_images()))

st.divider()

if not reachable:
    st.warning(
        f"Could not reach Ollama at `{config.OLLAMA_HOST}`. Start it with `ollama serve` "
        "(or just open the Ollama app) before running experiments. The rest of the app "
        "still works for browsing prompts and data."
    )
else:
    st.subheader("Models available in Ollama")
    models = ollama_client.list_models()
    if not models:
        st.info("No models pulled yet. Run e.g. `ollama pull llama3.1:8b` in a terminal.")
    else:
        rows = []
        for m in models:
            size_gb = m.size_bytes / (1024 ** 3)
            rows.append({
                "model": m.name,
                "vision": "yes" if m.is_vision else "",
                "params": m.parameter_size,
                "quant": m.quantization,
                "size (GB)": round(size_gb, 1),
                "heavy for a laptop": "⚠" if size_gb > 10 else "",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        vision_models = [m.name for m in models if m.is_vision]
        if not vision_models:
            st.warning(
                "No vision-capable model detected. The Image Analysis stage needs one "
                "(e.g. `ollama pull llava:7b` or `ollama pull llama3.2-vision`)."
            )

st.divider()
st.subheader("Pipeline stages")
for s in sorted(stages.STAGES, key=lambda s: s.order):
    vision_tag = " 👁 vision" if s.vision_required else ""
    st.markdown(f"**{s.order}. {s.name}**{vision_tag}  \n{s.description}")

st.divider()
st.markdown(
    "**Two separate things here:**\n\n"
    "- **Pipeline** runs the clean 7-stage pipeline end to end with whatever prompt+model "
    "is set active in **Pipeline Settings** — review/approve/edit at each stage, image "
    "review gate up front, no pickers cluttering the run itself.\n"
    "- **Experiment Lab** (**Run Experiment**, **Compare**, **Results Log**) is where you "
    "try prompt variants (and models, if needed) against each other, batch-run them, and "
    "manually review/rank the results. When one wins, promote it in Pipeline Settings.\n\n"
    "**Objectives** and **Prompts** support both: note what you're testing, and browse/"
    "edit/version the prompt library itself."
)
