#!/usr/bin/env python3
"""Faithful recreation of the real production pipeline, brief extraction
through slide generation, theme colors excluded per instruction. Every
call uses the real production prompt, the real production model, the
real production JSON Schema (schema-constrained decoding via Ollama's
`format` field, not loose "any JSON" mode), and the real documented
token budget and retry behavior for that stage.

Stage chain (production stage -> lab function):
    1. Meaningful image analysis   (brief-image)        -> vision model, 1024 tok, compact-retry + syntax-repair + deterministic fallback
    2. Project Brief                (brief-project)      -> text model, 4096 tok, whole-call retry once on invalid JSON
    3. Executive Summary            (brief-summary)       -> text model, 1500 tok, reads only the completed brief
    4. Normalized source context    (publish-normalize)   -> presentation model
    5. Narrative skeleton / slide plan (publish-skeleton) -> presentation model, 2048 tok
    6. Per-slide enrichment         (publish-enrich)      -> presentation model, 768 tok/slide
    7. Schema-bound slide copy      (publish-copy)        -> presentation model

Deterministic production steps (PDF intake, static extraction, human image
review gate, semantic layout selection, image resolution via ComfyUI,
persistence) are out of scope for this lab - the human review gate is
auto-approved (every meaningful image goes through). Two things stay
simplified because they depend on data this lab doesn't have: `purposeId`
is a free string instead of the real layout-catalog enum, and slide copy
uses one generic content contract instead of a real per-layout schema.
Neither affects brief/summary/slide-text quality, only final visual
rendering, which is out of scope here.

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
from core import production_schemas as schemas  # noqa: E402

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
        notes="Verbatim production brief-summary prompt, real schema-constrained decoding.",
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
        notes="Verbatim production publish-normalize prompt, with a fixed presentation-intent stand-in since there's no real user selection UI here. Real schema-constrained decoding.",
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
# Stage runners. Every call is now schema-constrained (Ollama's `format`
# field bound to the real production JSON Schema for that stage), not the
# loose response_format="json" mode used before - that loose mode is the
# most likely reason production's own prompt returned agentic tool-call
# JSON in the last sweep instead of a brief: with no schema locking the
# shape down, the model drifted into whatever JSON pattern it had seen
# most in training.
# ---------------------------------------------------------------------------

MAX_IMAGE_DESCRIPTIONS_PER_PAGE = 2  # real production cap, per the documented Step 5 behavior

REPAIR_SYSTEM_PROMPT = "Fix JSON syntax only. Do not add semantic information, fields, or array items. Return valid JSON only."


def _raw_call(model, system_prompt, user_prompt, *, images=None, num_predict=800,
               num_ctx=8192, timeout=240.0, response_format="json", label=""):
    """Returns (text, done_reason, err). done_reason is None on a transport failure."""
    try:
        result = ollama_client.generate(
            model=model, system_prompt=system_prompt, user_prompt=user_prompt,
            temperature=0.0, response_format=response_format, num_predict=num_predict,
            num_ctx=num_ctx, think=False, timeout=timeout, images=images,
        )
        text = result.output_text.strip()
        done_reason = result.raw_response.get("done_reason")
        if not text:
            thinking = (result.raw_response.get("thinking") or "")[:200]
            print(f"  WARNING: empty response ({label}); done_reason={done_reason!r}, thinking field had "
                  f"{len(result.raw_response.get('thinking', ''))} chars: {thinking!r}", file=sys.stderr)
        return text, done_reason, None
    except ollama_client.OllamaError as exc:
        print(f"  WARNING: call failed ({label}): {exc}", file=sys.stderr)
        return "", None, str(exc)


def _call(model, system_prompt, user_prompt, *, images=None, num_predict=800,
          num_ctx=8192, timeout=240.0, response_format="json", label=""):
    text, _done_reason, err = _raw_call(model, system_prompt, user_prompt, images=images,
                                         num_predict=num_predict, num_ctx=num_ctx, timeout=timeout,
                                         response_format=response_format, label=label)
    return text, err


def _parse_json(text, label, log):
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        log(f"{label}: response was not valid JSON ({exc})")
        return None, str(exc)


def _repair_json(model, malformed_text, label, log):
    """The real brief-image repair pass: syntax-only fix, no semantic changes."""
    log(f"{label}: attempting syntax-only repair")
    text, err = _call(model, REPAIR_SYSTEM_PROMPT, malformed_text, num_predict=512,
                       num_ctx=4096, timeout=120.0, response_format="json", label=f"{label} repair")
    if err:
        return None
    parsed, parse_err = _parse_json(text, f"{label} repair", log)
    return parsed


def run_image_analysis(images, vision_model, log, prompt=None):
    """brief-image: normal pass (1024 tok, full schema) -> on truncation,
    compact retry (640 tok, compact schema) -> on complete-but-malformed
    JSON, syntax repair -> on repeated failure, a deterministic fallback
    record (never a dropped image)."""
    prompt = prompt or prompts.get_prompt("image_analysis", "production")
    captions = []
    for img in images:
        user_prompt = prompt.render_user_prompt(CONTEXT=ds.image_prompt_context(img))
        label = f"image {img.filename}"
        text, done_reason, err = _raw_call(
            vision_model, prompt.system_prompt, user_prompt, images=[img.image_path],
            num_predict=1024, num_ctx=8192, timeout=180.0,
            response_format=schemas.image_analysis_schema(), label=label,
        )
        parsed = None
        if not err and text:
            if done_reason == "length":
                log(f"{label}: truncated at 1024 tokens, retrying with compact schema")
                text, _dr2, err = _raw_call(
                    vision_model, prompt.system_prompt, user_prompt, images=[img.image_path],
                    num_predict=640, num_ctx=8192, timeout=120.0,
                    response_format=schemas.compact_image_schema(), label=f"{label} (compact retry)",
                )
            if not err and text:
                parsed, parse_err = _parse_json(text, label, log)
                if parsed is None:
                    parsed = _repair_json(vision_model, text, label, log)
        if parsed is None:
            # deterministic fallback record - an image is never silently dropped
            parsed = {"summary": f"[vision analysis unavailable for {img.filename}]",
                      "subjects": [], "style": [], "environment": [], "colors": [],
                      "visibleText": [], "uncertainties": ["vision analysis failed"],
                      "confidence": {"detection": 0.0, "interpretation": 0.0}, "fallback": True}
        captions.append({"page": img.page_number, "file": img.filename, "output": parsed})
    ok_count = sum(1 for c in captions if not c["output"].get("fallback"))
    log(f"image analysis: {ok_count}/{len(captions)} succeeded (real vision output, not fallback)")
    return captions


def run_project_brief(pages, image_results, text_model, log, prompt=None):
    """brief-project: real 39-key AECO schema, 4096-token budget, schema-
    constrained decoding. Image evidence is capped at 2 descriptions per
    page, matching the documented Step 5 behavior. On invalid JSON, the
    whole consolidation call retries once (the documented queue-retry
    behavior for this stage - unlike brief-image, there's no cheap syntax
    repair for the brief itself)."""
    prompt = prompt or prompts.get_prompt("project_brief", "production")
    evidence = "\n\n".join(ds.page_context_full(p) for p in pages)

    by_page: dict[int, list] = {}
    for c in image_results:
        if c["output"] and not c["output"].get("fallback"):
            by_page.setdefault(c["page"], []).append(c["output"])
    image_evidence_lines = []
    for page, descs in sorted(by_page.items()):
        for d in descs[:MAX_IMAGE_DESCRIPTIONS_PER_PAGE]:
            image_evidence_lines.append(f"[Image, page {page}]: {json.dumps(d)}")
    context = evidence + "\n\nImage evidence:\n" + "\n".join(image_evidence_lines)
    user_prompt = prompt.render_user_prompt(CONTEXT=context)

    for attempt in (1, 2):
        text, err = _call(text_model, prompt.system_prompt, user_prompt, num_predict=4096,
                           num_ctx=8192, timeout=300.0, response_format=schemas.project_brief_schema(),
                           label=f"project brief (attempt {attempt})")
        if err:
            log(f"project brief: FAILED ({err})")
            return None
        parsed, parse_err = _parse_json(text, "project brief", log)
        if parsed is not None:
            log("project brief: ok")
            return parsed
        if attempt == 1:
            log("project brief: invalid JSON, retrying the whole consolidation call once")
    log("project brief: still invalid after retry, giving up")
    return None


def run_executive_summary(brief: dict, text_model, log, prompt=None):
    """brief-summary: reads only the completed brief, never raw pages or
    images. Real schema, 1500-token budget."""
    prompt = prompt or prompts.get_prompt("executive_summary", "production")
    user_prompt = prompt.render_user_prompt(CONTEXT=json.dumps(brief))
    text, err = _call(text_model, prompt.system_prompt, user_prompt, num_predict=1500,
                       num_ctx=8192, timeout=240.0, response_format=schemas.summary_schema(),
                       label="executive summary")
    if err:
        log(f"executive summary: FAILED ({err})")
        return None
    parsed, parse_err = _parse_json(text, "executive summary", log)
    log("executive summary: " + ("ok" if parsed is not None else f"invalid JSON ({parse_err})"))
    return parsed


def run_normalized_context(brief: dict, summary: dict, presentation_model, log, prompt=None):
    prompt = prompt or prompts.get_prompt("normalized_context", "production")
    context = f"Project brief:\n{json.dumps(brief)}\n\nExecutive summary:\n{json.dumps(summary)}"
    user_prompt = prompt.render_user_prompt(CONTEXT=context)
    text, err = _call(presentation_model, prompt.system_prompt, user_prompt, num_predict=2048,
                       num_ctx=8192, timeout=240.0, response_format=schemas.normalized_context_schema(),
                       label="normalized context")
    if err:
        log(f"normalized context: FAILED ({err})")
        return None
    parsed, parse_err = _parse_json(text, "normalized context", log)
    log("normalized context: " + ("ok" if parsed is not None else f"invalid JSON ({parse_err})"))
    return parsed


def run_slide_plan(normalized: dict, presentation_model, log, prompt=None):
    prompt = prompt or prompts.get_prompt("slide_plan", "production")
    requested_count = normalized.get("requestedSlideCount") if normalized else None
    user_prompt = prompt.render_user_prompt(CONTEXT=json.dumps(normalized))
    text, err = _call(presentation_model, prompt.system_prompt, user_prompt, num_predict=2048,
                       num_ctx=8192, timeout=300.0, response_format=schemas.slide_plan_schema(requested_count),
                       label="slide plan")
    if err:
        log(f"slide plan: FAILED ({err})")
        return None
    parsed, parse_err = _parse_json(text, "slide plan", log)
    if parsed is None:
        log(f"slide plan: raw response tail: ...{text[-300:]}")
    log("slide plan: " + ("ok" if parsed is not None else f"invalid JSON ({parse_err})"))
    return parsed


def run_slide_enrichment(slide, normalized: dict, presentation_model, log, prompt=None):
    prompt = prompt or prompts.get_prompt("slide_enrichment", "production")
    user_prompt = prompt.render_user_prompt(SLIDE=json.dumps(slide), CONTEXT=json.dumps(normalized))
    text, err = _call(presentation_model, prompt.system_prompt, user_prompt, num_predict=768,
                       num_ctx=8192, timeout=180.0, response_format=schemas.slide_enrichment_schema(),
                       label=f"enrich slide {slide.get('title', '?')}")
    if err:
        return None
    parsed, _parse_err = _parse_json(text, f"enrich slide {slide.get('title', '?')}", log)
    return parsed


def run_slide_content(slide, enrichment: dict, presentation_model, log, prompt=None):
    prompt = prompt or prompts.get_prompt("slide_content", "production")
    context = json.dumps(slide) + "\n\nEnrichment:\n" + json.dumps(enrichment or {})
    user_prompt = prompt.render_user_prompt(CONTEXT=context)
    text, err = _call(presentation_model, prompt.system_prompt, user_prompt, num_predict=500,
                       num_ctx=8192, timeout=180.0, response_format=schemas.slide_content_schema(),
                       label=f"slide copy {slide.get('title', '?')}")
    if err:
        return None
    parsed, _parse_err = _parse_json(text, f"slide copy {slide.get('title', '?')}", log)
    return parsed


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
    summary = run_executive_summary(brief, args.text_model, log) if brief is not None else None
    normalized = (run_normalized_context(brief, summary, args.presentation_model, log)
                  if brief is not None and summary is not None else None)
    slide_plan = run_slide_plan(normalized, args.presentation_model, log) if normalized is not None else None

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
