#!/usr/bin/env python3
"""Take-3 objective: produce the single best possible plain-text brief
summary from the extracted document (text + images), and compare pipeline
STRUCTURE rather than prompt wording.

Prompt style is fixed across every stage and every variant, using the one
finding that held up in the last round: state "reason before answering"
once up front, then repeat it again immediately after the input material,
right before the model has to produce output. Every stage outputs plain
text, not JSON - production's real deliverable here is a summary a person
reads, not a structured record.

What varies between variants is how many LLM hops the material passes
through before the final summary comes out:

  full        images -> per-page evidence (LLM) -> long brief (LLM) -> summary (LLM)
  condensed   images -> summary directly from raw text + image captions (LLM)
  minimal     summary directly from raw text alone, no vision step at all (LLM)

`minimal` exists partly to answer a real question (does the vision stage
even matter for summary quality) and partly as a guaranteed-to-finish
fallback, since image analysis was the single least reliable stage last
round - every image call is wrapped so one failure doesn't take down the
whole run.

Usage:
    python run_summary_pipeline.py                       # run all 3 variants
    python run_summary_pipeline.py --variants full minimal
    python run_summary_pipeline.py --text-model llama3.1:8b --vision-model moondream
    python run_summary_pipeline.py --max-images 10        # cap vision calls while testing
    python run_summary_pipeline.py --dry-run              # show the plan, call nothing
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

OUT_DIR = Path(__file__).resolve().parent / "experiments" / "summary_pipeline"


# ---------------------------------------------------------------------------
# Fixed prompt style - the one pattern that reliably beat everything else
# last round. Every stage system prompt states the reasoning instruction
# once; every user prompt repeats a short version of it right after the
# input material.
# ---------------------------------------------------------------------------

def _call(model, system_prompt, user_prompt, *, images=None, num_predict=800,
          timeout=180.0, label=""):
    try:
        result = ollama_client.generate(
            model=model, system_prompt=system_prompt, user_prompt=user_prompt,
            temperature=0.0, response_format=None, num_predict=num_predict,
            timeout=timeout, images=images,
        )
        return result.output_text.strip(), None
    except ollama_client.OllamaError as exc:
        print(f"  WARNING: call failed ({label}): {exc}", file=sys.stderr)
        return "", str(exc)


def analyze_image(image_ref, vision_model: str) -> str:
    """One meaningful image -> a short plain-text caption. Never raises -
    a failed image degrades to a placeholder instead of killing the run."""
    system_prompt = (
        "Describe only what is visibly evidenced in this image, considering the "
        "nearby heading and text for context. Think through what you can actually "
        "see before writing the description. Never invent dimensions, quantities, "
        "or labels you cannot read. Write two or three plain sentences, no JSON, "
        "no bullet points."
    )
    user_prompt = (
        f"Page: {image_ref.page_number}\n"
        f"Heading: {image_ref.nearest_heading}\n"
        f"Nearby text: {image_ref.nearby_text}\n\n"
        "Look at the image carefully first, then describe only what it actually shows."
    )
    text, err = _call(
        vision_model, system_prompt, user_prompt, images=[image_ref.image_path],
        num_predict=150, timeout=180.0, label=f"image {image_ref.filename}",
    )
    if err or not text:
        return f"[Image on page {image_ref.page_number} ({image_ref.filename}): vision analysis unavailable]"
    return f"[Image on page {image_ref.page_number}]: {text}"


def page_evidence(page, image_captions_for_page: list[str], text_model: str) -> str:
    system_prompt = (
        "Think through what this page actually states before answering - facts, "
        "requirements, constraints, risks. Then write a plain-text paragraph "
        "summarizing only what this page evidences. No JSON, no headers, just prose."
    )
    body = ds.page_context_full(page)
    if image_captions_for_page:
        body += "\n\nImage evidence on this page:\n" + "\n".join(image_captions_for_page)
    user_prompt = body + "\n\nReason through the material first, then write the paragraph."
    text, err = _call(text_model, system_prompt, user_prompt, num_predict=400,
                       label=f"page {page.page_number} evidence")
    return text if not err else f"[Page {page.page_number}: analysis unavailable]"


def write_brief(evidence_text: str, text_model: str) -> str:
    system_prompt = (
        "You are a senior AECO project brief analyst. Think through the evidence "
        "systematically - scope, requirements, constraints, risks, open questions - "
        "before writing. Then write one detailed, well-organized project brief in "
        "plain prose. Cover requirements, constraints, risks, and opportunities as "
        "paragraphs, not JSON or bullet lists. Never invent facts not present in the evidence."
    )
    user_prompt = (
        "Evidence:\n" + evidence_text +
        "\n\nReason through the evidence first, then write the full detailed brief in plain text."
    )
    text, err = _call(text_model, system_prompt, user_prompt, num_predict=1500,
                       timeout=240.0, label="project brief")
    return text if not err else "[Brief unavailable]"


def write_summary(source_text: str, text_model: str, *, from_brief: bool) -> str:
    label = "brief" if from_brief else "raw evidence"
    system_prompt = (
        f"You are summarizing a project {label} for someone who has not read it. "
        "Think through what actually matters before writing - the core project, its "
        "key requirements, constraints and risks - before drafting. Then write a "
        "concise, dense plain-text summary, a few short paragraphs, no headers, no "
        "JSON, no bullet lists. Never invent facts not present in the source."
    )
    user_prompt = (
        f"Source ({label}):\n" + source_text +
        "\n\nReason through what matters first, then write the concise summary."
    )
    text, err = _call(text_model, system_prompt, user_prompt, num_predict=700,
                       timeout=200.0, label="summary")
    return text if not err else "[Summary unavailable]"


# ---------------------------------------------------------------------------
# Pipeline variants
# ---------------------------------------------------------------------------

def run_full(pages, images, text_model, vision_model, log):
    log("full: analyzing images")
    captions_by_page: dict[int, list[str]] = {}
    for img in images:
        cap = analyze_image(img, vision_model)
        captions_by_page.setdefault(img.page_number, []).append(cap)

    log("full: per-page evidence")
    page_texts = []
    for page in pages:
        ev = page_evidence(page, captions_by_page.get(page.page_number, []), text_model)
        page_texts.append(f"--- Page {page.page_number} ---\n{ev}")
    evidence_blob = "\n\n".join(page_texts)

    log("full: writing brief")
    brief = write_brief(evidence_blob, text_model)

    log("full: writing summary")
    summary = write_summary(brief, text_model, from_brief=True)
    return {"variant": "full", "brief": brief, "summary": summary, "stage_count": 2 + len(pages) + len(images)}


def run_condensed(pages, images, text_model, vision_model, log):
    log("condensed: analyzing images")
    captions = [analyze_image(img, vision_model) for img in images]

    log("condensed: assembling raw evidence (no per-page LLM pass)")
    page_texts = [ds.page_context_full(p) for p in pages]
    evidence_blob = "\n\n".join(page_texts) + "\n\nImage evidence:\n" + "\n".join(captions)

    log("condensed: writing summary directly")
    summary = write_summary(evidence_blob, text_model, from_brief=False)
    return {"variant": "condensed", "brief": None, "summary": summary, "stage_count": 1 + len(images)}


def run_minimal(pages, images, text_model, log):
    log("minimal: assembling raw text only (no vision calls)")
    page_texts = [ds.page_context_full(p) for p in pages]
    evidence_blob = "\n\n".join(page_texts)

    log("minimal: writing summary directly")
    summary = write_summary(evidence_blob, text_model, from_brief=False)
    return {"variant": "minimal", "brief": None, "summary": summary, "stage_count": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="*", default=["full", "condensed", "minimal"],
                         choices=["full", "condensed", "minimal"])
    parser.add_argument("--text-model", default="llama3.1:8b")
    parser.add_argument("--vision-model", default="moondream")
    parser.add_argument("--max-images", type=int, default=None,
                         help="cap how many meaningful images get analyzed (full/condensed only)")
    parser.add_argument("--out", default=str(OUT_DIR / "summary_results.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pages = list(ds.list_pages())
    images = list(ds.list_images())
    if args.max_images is not None:
        images = images[: args.max_images]

    print(f"Plan: {len(pages)} pages, {len(images)} images, variants={args.variants}")
    print(f"text model: {args.text_model}   vision model: {args.vision_model}")
    if args.dry_run:
        return

    if not ollama_client.is_reachable():
        print("ERROR: Ollama is not reachable. Start it with `ollama serve` first.")
        sys.exit(1)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    results = {}
    for variant in args.variants:
        started = time.time()
        if variant == "full":
            results["full"] = run_full(pages, images, args.text_model, args.vision_model, log)
        elif variant == "condensed":
            results["condensed"] = run_condensed(pages, images, args.text_model, args.vision_model, log)
        elif variant == "minimal":
            results["minimal"] = run_minimal(pages, images, args.text_model, log)
        results[variant]["elapsed_seconds"] = round(time.time() - started, 1)
        log(f"{variant} done in {results[variant]['elapsed_seconds']}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("Upload this file (or paste its contents) for side-by-side comparison of the final summaries.")


if __name__ == "__main__":
    main()
