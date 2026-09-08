import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import data_sources as ds
from core import db, ollama_client, objectives
from core import prompts as prompts_mod
from core import runner, stages

st.set_page_config(page_title="Run Experiment", layout="wide")
st.title("Run Experiment")
st.caption(
    "Batch-run one stage across prompt variants (the main lever — model choice is "
    "usually already settled) and log every output for you to review and rank. "
    "Every combination is one call to your local Ollama."
)

if not ollama_client.is_reachable():
    st.error("Ollama is not reachable. Start it (`ollama serve`) and reload this page.")
    st.stop()

stage_ids = [s.id for s in stages.STAGES]
stage_labels = {s.id: f"{s.order}. {s.name}" for s in stages.STAGES}
stage_id = st.selectbox("Stage", stage_ids, format_func=lambda s: stage_labels[s])
stage_def = stages.STAGES_BY_ID[stage_id]
st.info(stage_def.description)

obj_list = objectives.list_objectives()
obj_choice = st.selectbox(
    "Objective (optional)", ["(none)"] + [o.title for o in obj_list],
)
objective_id = None
if obj_choice != "(none)":
    objective_id = next(o.id for o in obj_list if o.title == obj_choice)

st.subheader("1. Input")

context_kwargs_list: list[dict] = []  # each item: {"context": {...}, "input_ref":..., "input_summary":..., "images": [...], "context_variant": ...}

if stage_def.input_kind == "image":
    images = ds.list_images()
    choices = st.multiselect(
        "Images", [i.filename for i in images],
        default=images[0].filename if images else None,
        format_func=lambda f: f"{f} (page {next(i.page_number for i in images if i.filename == f)})",
    )
    for filename in choices:
        img = ds.get_image(filename)
        context_kwargs_list.append({
            "context": {"CONTEXT": ds.image_prompt_context(img)},
            "input_ref": img.ref, "input_summary": f"page {img.page_number}",
            "images": [img.image_path], "context_variant": "image_metadata",
        })

elif stage_def.input_kind == "page":
    pages = ds.list_pages()
    variant_name = st.selectbox("Context packaging", list(ds.CONTEXT_VARIANTS.keys()))
    variant_fn = ds.CONTEXT_VARIANTS[variant_name]
    choices = st.multiselect(
        "Pages", [p.page_number for p in pages],
        default=[pages[0].page_number] if pages else [],
        format_func=lambda n: f"page {n}" + (f" - {ds.get_page(n).heading[:40]}" if ds.get_page(n).heading else ""),
    )
    for page_number in choices:
        page = ds.get_page(page_number)
        page_images = ds.images_on_page(page_number)
        image_evidence = "[]"
        context_kwargs_list.append({
            "context": {"CONTEXT": variant_fn(page), "IMAGE_EVIDENCE": image_evidence},
            "input_ref": page.ref, "input_summary": page.heading or f"page {page_number}",
            "images": None, "context_variant": variant_name,
        })

else:
    st.markdown("This stage consumes an upstream stage's output. Choose where the "
                "`{{CONTEXT}}` value comes from:")
    source_mode = st.radio("Input source", ["Raw pages from document", "Existing run output(s)"])
    if source_mode == "Raw pages from document":
        pages = ds.list_pages()
        choices = st.multiselect(
            "Pages to concatenate as context", [p.page_number for p in pages],
            default=[p.page_number for p in pages[:5]],
        )
        if choices:
            selected_pages = [ds.get_page(n) for n in choices]
            combined = "\n\n".join(ds.page_context_full(p) for p in selected_pages)
            context_kwargs_list.append({
                "context": {"CONTEXT": combined},
                "input_ref": f"pages:{sorted(choices)}", "input_summary": "raw pages",
                "images": None, "context_variant": "raw_pages",
            })
    else:
        upstream_runs = db.list_runs()
        run_labels = {
            r["id"]: f"{r['stage']} / {r['prompt_name']} / {r['model']} / {r['input_ref']}"
            for r in upstream_runs if r["is_valid_json"]
        }
        chosen = st.multiselect(
            "Existing valid runs to use as input", list(run_labels.keys()),
            format_func=lambda rid: run_labels[rid],
        )
        if chosen:
            payloads = [json.loads(r["parsed_json"]) for r in upstream_runs if r["id"] in chosen]
            merged = payloads if len(payloads) > 1 else payloads[0]
            context_kwargs_list.append({
                "context": {"CONTEXT": json.dumps(merged)},
                "input_ref": f"runs:{','.join(chosen)}", "input_summary": "from prior run(s)",
                "images": None, "context_variant": "prior_run_output",
            })

st.subheader("2. Prompts")
all_prompts = prompts_mod.list_prompts(stage_id)
prompt_options = {p.display_name: p for p in all_prompts}
chosen_prompt_names = st.multiselect(
    "Prompts to test", list(prompt_options.keys()),
    default=[k for k in prompt_options if "production" in k] or list(prompt_options.keys())[:1],
)
chosen_prompts = [prompt_options[k] for k in chosen_prompt_names]

st.subheader("3. Models")
models = ollama_client.list_models()
if stage_def.vision_required:
    models = [m for m in models if m.is_vision]
model_names = st.multiselect("Models to test", [m.name for m in models],
                              default=[models[0].name] if models else [])

st.subheader("4. Sampling & decoding")
col1, col2 = st.columns(2)
with col1:
    temperatures = st.multiselect("Temperatures", [0.0, 0.2, 0.4, 0.7, 1.0], default=[0.0])
with col2:
    response_formats = st.multiselect(
        "Decoding mode", list(runner.RESPONSE_FORMATS), default=["json_schema"],
        help="freeform = no constraint; json_loose = Ollama 'format: json'; "
             "json_schema = Ollama structured outputs constrained to this stage's schema.",
    )

total_runs = len(context_kwargs_list) * len(chosen_prompts) * len(model_names) * len(temperatures) * len(response_formats)
st.markdown(f"**Total calls this will make: {total_runs}**")
if total_runs > 30:
    st.warning("That's a lot of calls — this may take a while on a laptop. Consider narrowing the matrix.")

if st.button("Run", type="primary", disabled=total_runs == 0):
    progress = st.progress(0.0)
    results_area = st.container()
    done = 0
    combos = list(itertools.product(context_kwargs_list, chosen_prompts, model_names, temperatures, response_formats))
    for input_kwargs, prompt, model, temperature, response_format in combos:
        with st.spinner(f"{prompt.display_name} · {model} · T={temperature} · {response_format} · {input_kwargs['input_ref']}"):
            result = runner.run_stage(
                stage_id, prompt, model, input_kwargs["context"],
                temperature=temperature, response_format=response_format,
                images=input_kwargs.get("images"),
                input_ref=input_kwargs["input_ref"], input_summary=input_kwargs["input_summary"],
                context_variant=input_kwargs.get("context_variant", ""),
                objective_id=objective_id,
            )
        done += 1
        progress.progress(done / total_runs)
        badge = "✅ valid" if result.is_valid_json else "❌ invalid"
        with results_area.expander(
            f"{badge} · {prompt.display_name} · {model} · T={temperature} · {response_format} · "
            f"{input_kwargs['input_ref']} · {result.latency_ms:.0f}ms"
        ):
            if result.error:
                st.error(result.error)
            if result.schema_errors:
                st.warning(" / ".join(result.schema_errors))
            if result.parsed_json is not None:
                st.json(result.parsed_json)
            else:
                st.text(result.output_text)
    st.success(f"Done. {done} runs logged to experiments/lab.db and experiments/results_log.csv.")
