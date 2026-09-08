import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import ollama_client

st.set_page_config(page_title="Models", layout="wide")
st.title("Models")
st.caption("Pulled from your local Ollama (`GET /api/tags`). Add more with `ollama pull <name>` "
           "in a terminal, then refresh below.")

if st.button("Refresh"):
    st.rerun()

if not ollama_client.is_reachable():
    st.error(f"Ollama is not reachable at the configured host. Start it and refresh.")
    st.stop()

models = ollama_client.list_models()
if not models:
    st.info("No models pulled yet.")
    st.stop()

text_models = [m for m in models if not m.is_vision]
vision_models = [m for m in models if m.is_vision]

st.subheader(f"Vision-capable ({len(vision_models)}) — usable for Image Analysis")
if vision_models:
    st.dataframe(
        [{"model": m.name, "params": m.parameter_size, "quant": m.quantization,
          "size (GB)": round(m.size_bytes / 1024**3, 1)} for m in vision_models],
        use_container_width=True, hide_index=True,
    )
else:
    st.info("None detected. Pull one, e.g. `ollama pull llava:7b` or "
            "`ollama pull llama3.2-vision`.")

st.subheader(f"Text models ({len(text_models)}) — usable for every other stage")
if text_models:
    st.dataframe(
        [{"model": m.name, "params": m.parameter_size, "quant": m.quantization,
          "size (GB)": round(m.size_bytes / 1024**3, 1),
          "heavy for a laptop": "⚠" if m.size_bytes / 1024**3 > 10 else ""}
         for m in text_models],
        use_container_width=True, hide_index=True,
    )
else:
    st.info("None detected.")

with st.expander("Pull a new model"):
    model_name = st.text_input("Model tag, e.g. qwen2.5:7b-instruct")
    if st.button("Pull") and model_name.strip():
        placeholder = st.empty()
        try:
            for progress in ollama_client.pull_model(model_name.strip()):
                status = progress.get("status", "")
                placeholder.text(status)
            st.success(f"Pulled {model_name}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Pull failed: {exc}")
