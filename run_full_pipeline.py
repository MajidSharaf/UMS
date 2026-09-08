#!/usr/bin/env python3
"""Faithful recreation of the real production pipeline, brief extraction
through slide generation, theme colors excluded per instruction. This is
the baseline run: one prompt per stage (the real, verbatim production
prompt, seeded from reference/ums_pipeline_reference.md), the real
production models. Once this runs cleanly end to end, prompt variation
becomes the next round of testing on top of this same chain.

Stage chain (production stage -> lab function):
    1. Meaningful image analysis   (brief-image)        -> vision model
    2. Project Brief                (brief-project)      -> text model
    3. Executive Summary            (brief-summary)       -> text model
    4. Normalized source context    (publish-normalize)   -> presentation model
    5. Narrative skeleton / slide plan (publish-skeleton) -> presentation model
    6. Per-slide enrichment         (publish-enrich)      -> presentation model
    7. Schema-bound slide copy      (publish-copy)        -> presentation model

Deterministic production steps (PDF intake, static extraction, human image
review gate, semantic layout selection, image resolution via ComfyUI,
persistence) are out of scope for this lab - the human review gate is
auto-approved (every meaningful image goes through), and layout selection
uses one generic content contract since the real layout definitions aren't
available here.

Usage:
    python run_full_pipeline.py --dry-run
    python run_full_pipeline.py
    python run_full_pipeline.py --text-model gemma4:26b --presentation-model qwen3.6:35b --vision-model qwen3-vl:8b
    python run_full_pipeline.py --max-images 5   # sanity pass before the full run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from core import data_sources as ds  # noqa: E402
from core import ollama_client  # noqa: E402
from core import prompts  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "experiments" / "full_pipeline"

# ---------------------------------------------------------------------------
# Seed the 3 new stages with their real, verbatim production prompts.
# image_analysis / project_brief / slide_plan / slide_content already exist
# in prompts_library/ from the earlier round.
# ---------------------------------------------------------------------------

def seed_new_stage_prompts():
    prompts.seed_if_missing(
        "executive_summary", "production",
        system_prompt=(
            "You are a careful document analyst. Use only supplied page evidence. "
            "Never invent missing details. Mark uncertainty explicitly and cite "
            "important page numbers. Return only valid JSON matching the supplied schema."
        ),
        user_template=(
            "Summarize this completed project brief. The brief is the authoritative "
            "input; retain important source page references and uncertainty:\n{{CONTEXT}}"
        ),
        notes="Verbatim production brief-summary prompt. Lab uses plain text output, not the full schema.",
        is_production_baseline=True,
    )
    prompts.seed_if_missing(
        "normalized_context", "production",
        system_prompt=(
            "Normalize the presentation request and selected UMS Brief into concise "
            "structured planning context. Preserve explicit facts, constraints and "
            "provenance. Do not invent information."
        ),
        user_template=(
            "Presentation purpose: general project overview. Audience: internal team. "
            "Tone: professional. Language: English.\n\n{{CONTEXT}}"
        ),
        notes="Verbatim production publish-normalize prompt, with a fixed presentation-intent stand-in since there's no real user selection UI here.",
        is_production_baseline=True,
    )
    prompts.seed_if_missing(
        "slide_enrichment", "production",
        system_prompt=(
            "Expand this single slide's plan into a focused content brief: the "
            "specific points it must cover, the evidence that supports them, and "
            "what should NOT be included. Do not write final slide copy. Return "
            "only valid JSON matching: content_brief (string), supporting_points "
            "(array of strings), exclusions (array of strings)."
        ),
        user_template=(
            "Slide plan entry:\n{{SLIDE}}\n\nPresentation context:\n{{CONTEXT}}"
        ),
        notes="Adapted from production publish-enrich's per-slide job description (exact production prompt text wasn't in the extracted docs).",
        is_production_baseline=True,
    )


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _call(model, system_prompt, user_prompt, *, images=None, num_predict=800,
          timeout=240.0, response_format="json", label=""):
    try:
        result = ollama_client.generate(
            model=model, system_prompt=system_prompt, user_prompt=user_prompt,
            temperature=0.0, response_format=response_format, num_predict=num_predict,
            timeout=timeout, images=images,
        )
        return result.output_text.strip(), None
    except ollama_client.OllamaError as exc:
        print(f"  WARNING: call failed ({label}): {exc}", file=sys.stderr)
        return "", str(exc)


def run_image_analysis(images, vision_model, log):
    prompt = prompts.get_prompt("image_analysis", "production")
    captions = []
    for img in images:
        user_prompt = prompt.render_user_prompt(CONTEXT=ds.image_prompt_context(img))
        text, err = _call(vision_model, prompt.system_prompt, user_prompt, images=[img.image_path],
                           num_predict=300, timeout=180.0, response_format="json",
                           label=f"image {img.filename}")
        captions.append({"page": img.page_number, "file": img.filename,
                          "output": text if not err else None, "error": err})
    log(f"image analysis: {sum(1 for c in captions if c['output'])}/{len(captions)} succeeded")
    return captions


def run_project_brief(pages, image_results, text_model, log):
    prompt = prompts.get_prompt("project_brief", "production")
    evidence = "\n\n".join(ds.page_context_full(p) for p in pages)
    image_evidence = "\n".join(
        f"[Image, page {c['page']}]: {c['output']}" for c in image_results if c["output"]
    )
    context = evidence + "\n\nImage evidence:\n" + image_evidence
    user_prompt = prompt.render_user_prompt(CONTEXT=context)
    text, err = _call(text_model, prompt.system_prompt, user_prompt, num_predict=2000,
                       timeout=300.0, label="project brief")
    log("project brief: " + ("ok" if not err else f"FAILED ({err})"))
    return text


def run_executive_summary(brief_text, text_model, log):
    prompt = prompts.get_prompt("executive_summary", "production")
    user_prompt = prompt.render_user_prompt(CONTEXT=brief_text)
    text, err = _call(text_model, prompt.system_prompt, user_prompt, num_predict=900,
                       timeout=240.0, response_format=None, label="executive summary")
    log("executive summary: " + ("ok" if not err else f"FAILED ({err})"))
    return text


def run_normalized_context(brief_text, summary_text, presentation_model, log):
    prompt = prompts.get_prompt("normalized_context", "production")
    context = f"Project brief:\n{brief_text}\n\nExecutive summary:\n{summary_text}"
    user_prompt = prompt.render_user_prompt(CONTEXT=context)
    text, err = _call(presentation_model, prompt.system_prompt, user_prompt, num_predict=900,
                       timeout=240.0, response_format=None, label="normalized context")
    log("normalized context: " + ("ok" if not err else f"FAILED ({err})"))
    return text


def run_slide_plan(normalized_text, presentation_model, log):
    prompt = prompts.get_prompt("slide_plan", "production")
    user_prompt = prompt.render_user_prompt(CONTEXT=normalized_text)
    text, err = _call(presentation_model, prompt.system_prompt, user_prompt, num_predict=1800,
                       timeout=300.0, label="slide plan")
    log("slide plan: " + ("ok" if not err else f"FAILED ({err})"))
    if err:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log("slide plan: response was not valid JSON, downstream slide stages will be skipped")
        return None


def run_slide_enrichment(slide, normalized_text, presentation_model, log):
    prompt = prompts.get_prompt("slide_enrichment", "production")
    user_prompt = prompt.render_user_prompt(SLIDE=json.dumps(slide), CONTEXT=normalized_text)
    text, err = _call(presentation_model, prompt.system_prompt, user_prompt, num_predict=500,
                       timeout=180.0, label=f"enrich slide {slide.get('title', '?')}")
    return text if not err else None


def run_slide_content(slide, enrichment_text, presentation_model, log):
    prompt = prompts.get_prompt("slide_content", "production")
    context = json.dumps(slide) + "\n\nEnrichment:\n" + (enrichment_text or "")
    user_prompt = prompt.render_user_prompt(CONTEXT=context)
    text, err = _call(presentation_model, prompt.system_prompt, user_prompt, num_predict=500,
                       timeout=180.0, label=f"slide copy {slide.get('title', '?')}")
    return text if not err else None


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-model", default="gemma4:26b",
                         help="brief-project / brief-summary model (production default: Gemma 4 26B)")
    parser.add_argument("--presentation-model", default="qwen3.5:35b-a3b",
                         help="publish-* stages model (production default: Qwen 3.6 35B A3B; "
                              "substituted with Qwen 3.5 35B A3B, same family/size class, because "
                              "the 3.6 tag's manifest consistently failed with EOF on this registry, "
                              "confirmed via direct curl testing to be specific to that manifest, "
                              "not a network/DNS issue)")
    parser.add_argument("--vision-model", default="qwen3-vl:8b-instruct",
                         help="brief-image model (production default: Qwen3-VL 8B Instruct)")
    parser.add_argument("--max-images", type=int, default=None,
                         help="cap meaningful images processed, for a fast sanity pass")
    parser.add_argument("--out", default=str(OUT_DIR / "full_pipeline_result.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seed_new_stage_prompts()

    pages = list(ds.list_pages())
    images = list(ds.list_images())
    if args.max_images is not None:
        images = images[: args.max_images]

    print(f"Plan: {len(pages)} pages, {len(images)} images")
    print(f"text model: {args.text_model}   presentation model: {args.presentation_model}   vision model: {args.vision_model}")
    print("Stages: image_analysis -> project_brief -> executive_summary -> "
          "normalized_context -> slide_plan -> slide_enrichment -> slide_content")
    print("(theme colors excluded)")
    if args.dry_run:
        return

    if not ollama_client.is_reachable():
        print("ERROR: Ollama is not reachable. Start it with `ollama serve` first.")
        sys.exit(1)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    started = time.time()
    image_results = run_image_analysis(images, args.vision_model, log)
    brief = run_project_brief(pages, image_results, args.text_model, log)
    summary = run_executive_summary(brief, args.text_model, log)
    normalized = run_normalized_context(brief, summary, args.presentation_model, log)
    slide_plan = run_slide_plan(normalized, args.presentation_model, log)

    slides_out = []
    if slide_plan and isinstance(slide_plan.get("slides"), list):
        for i, slide in enumerate(slide_plan["slides"], start=1):
            log(f"slide {i}/{len(slide_plan['slides'])}: enrich + copy")
            enrichment = run_slide_enrichment(slide, normalized, args.presentation_model, log)
            copy = run_slide_content(slide, enrichment, args.presentation_model, log)
            slides_out.append({"plan": slide, "enrichment": enrichment, "copy": copy})

    result = {
        "models": {"text": args.text_model, "presentation": args.presentation_model, "vision": args.vision_model},
        "image_analysis": image_results,
        "brief": brief,
        "executive_summary": summary,
        "normalized_context": normalized,
        "slide_plan": slide_plan,
        "slides": slides_out,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {result['elapsed_seconds']}s. Wrote {out_path}")


if __name__ == "__main__":
    main()
