"""Where you set which prompt + model the clean Pipeline page uses at each
stage. This is the only place that changes — the pipeline run screen never
shows a picker. When the experiment lab finds something better, it gets
promoted here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import active_config, ollama_client
from core import prompts as prompts_mod
from core import stages

st.set_page_config(page_title="Pipeline Settings", layout="wide")
st.title("Pipeline Settings")
st.caption(
    "One prompt and one model per stage — this is what the Pipeline page runs with. "
    "Try alternatives in the Experiment Lab first; promote the winner here."
)

if not ollama_client.is_reachable():
    st.error("Ollama is not reachable. Start it (`ollama serve`) and reload this page.")
    st.stop()

models = ollama_client.list_models()

for s in sorted(stages.STAGES, key=lambda s: s.order):
    active = active_config.get_active(s.id)
    all_prompts = prompts_mod.list_prompts(s.id)
    prompt_options = {p.display_name: p for p in all_prompts}
    stage_models = [m for m in models if m.is_vision] if s.vision_required else models

    with st.container(border=True):
        st.markdown(f"**{s.order}. {s.name}**")
        st.caption(s.description)
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            default_key = None
            if active:
                default_key = next((k for k, p in prompt_options.items()
                                     if p.name == active.prompt_name), None)
            keys = list(prompt_options.keys())
            idx = keys.index(default_key) if default_key in keys else 0
            chosen_key = st.selectbox("Prompt", keys, index=idx, key=f"prompt-{s.id}")
        with col2:
            model_names = [m.name for m in stage_models]
            m_idx = model_names.index(active.model) if active and active.model in model_names else 0
            chosen_model = st.selectbox("Model", model_names, index=m_idx if model_names else 0, key=f"model-{s.id}") if model_names else None
        with col3:
            st.write("")
            st.write("")
            if st.button("Save", key=f"save-{s.id}", disabled=not chosen_model):
                chosen_prompt = prompt_options[chosen_key]
                active_config.set_active(s.id, chosen_prompt.name, chosen_model)
                st.success("Saved.")
                st.rerun()

st.divider()
st.caption(f"Stored in `pipeline_config.yaml` at the repo root. Temperature is fixed at "
           f"{active_config.PIPELINE_TEMPERATURE} and decoding mode at "
           f"`{active_config.PIPELINE_RESPONSE_FORMAT}` for the clean pipeline run.")
