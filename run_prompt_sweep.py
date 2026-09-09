#!/usr/bin/env python3
"""Prompt-variation experiment on top of the real production pipeline.

Reuses the ~20 hand-authored prompt variants per stage already in
prompts_library/ (built for the earlier small-model round) and tests them
against the real production models, holding everything except the prompt
text fixed. Same methodology as before: one input, many prompts, so any
difference in output is attributable to wording, not to a moving input.

For each testable stage, a single baseline pass produces the shared input
material once (cached), then every prompt variant for that stage runs
against that same cached input:

    image_analysis        -> one representative image
    project_brief          -> the real evidence from all pages + images
    slide_plan              -> the real normalized context from the brief
    slide_content            -> one representative slide from the plan

Usage:
    python run_prompt_sweep.py --dry-run
    python run_prompt_sweep.py                       # every stage
    python run_prompt_sweep.py --stages project_brief slide_plan
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
import run_full_pipeline as pipe  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "experiments" / "prompt_sweep"


def sweep_stage(stage, run_fn, run_args, log, out, timing):
    variants = prompts.list_prompts(stage)
    log(f"{stage}: {len(variants)} prompt variants to test")
    stage_started = time.time()
    results = []
    for i, prompt in enumerate(variants, start=1):
        log(f"{stage} [{i}/{len(variants)}] {prompt.name}")
        started = time.time()
        output = run_fn(*run_args, prompt=prompt)
        elapsed = round(time.time() - started, 1)
        results.append({
            "prompt_name": prompt.name,
            "is_production_baseline": prompt.is_production_baseline,
            "output": output if isinstance(output, str) else json.dumps(output) if output else None,
            "elapsed_seconds": elapsed,
        })
    stage_elapsed = round(time.time() - stage_started, 1)
    out[stage] = results
    (OUT_DIR / f"{stage}_sweep.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{stage}_sweep.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    per_variant = [r["elapsed_seconds"] for r in results]
    timing[stage] = {
        "variants_tested": len(results),
        "sweep_seconds": stage_elapsed,
        "avg_seconds_per_variant": round(sum(per_variant) / len(per_variant), 1) if per_variant else 0,
        "min_seconds": min(per_variant) if per_variant else 0,
        "max_seconds": max(per_variant) if per_variant else 0,
    }
    log(f"{stage}: wrote {OUT_DIR / f'{stage}_sweep.json'} "
        f"({stage_elapsed}s total, avg {timing[stage]['avg_seconds_per_variant']}s/variant)")


def _timed_baseline(label, fn, timing, *args, **kwargs):
    started = time.time()
    result = fn(*args, **kwargs)
    elapsed = round(time.time() - started, 1)
    timing.setdefault("baseline_passes", {})[label] = elapsed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="*",
                         default=["image_analysis", "project_brief", "slide_plan", "slide_content"],
                         choices=["image_analysis", "project_brief", "slide_plan", "slide_content"])
    parser.add_argument("--text-model", default="gemma4:26b")
    parser.add_argument("--presentation-model", default="qwen3.5:35b-a3b")
    parser.add_argument("--vision-model", default="qwen3-vl:8b-instruct")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pipe.seed_new_stage_prompts()

    pages = list(ds.list_pages())
    images = list(ds.list_images())
    plan_counts = {s: len(prompts.list_prompts(s)) for s in args.stages}
    print(f"Plan: stages={args.stages}")
    for s, n in plan_counts.items():
        print(f"  {s}: {n} prompt variants")
    if args.dry_run:
        return

    if not ollama_client.is_reachable():
        print("ERROR: Ollama is not reachable. Start it with `ollama serve` first.")
        sys.exit(1)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    run_started = time.time()
    out: dict = {}
    timing: dict = {}

    # --- Image Analysis: one representative image ---
    if "image_analysis" in args.stages:
        rep_image = images[0]
        log(f"image_analysis baseline image: {rep_image.filename}")
        sweep_stage("image_analysis", pipe.run_image_analysis, ([rep_image], args.vision_model, log), log, out, timing)

    # --- Project Brief: real evidence from a real baseline image pass ---
    baseline_image_results = None
    if "project_brief" in args.stages:
        log("project_brief: running baseline image analysis for shared evidence (all images, production prompt)")
        baseline_image_results = _timed_baseline("project_brief:image_analysis", pipe.run_image_analysis,
                                                   timing, images, args.vision_model, log)
        sweep_stage("project_brief", pipe.run_project_brief,
                    (pages, baseline_image_results, args.text_model, log), log, out, timing)

    # --- Slide Plan: real normalized context from a real baseline brief ---
    baseline_normalized = None
    if "slide_plan" in args.stages:
        if baseline_image_results is None:
            log("slide_plan: running baseline image analysis for shared evidence (all images, production prompt)")
            baseline_image_results = _timed_baseline("slide_plan:image_analysis", pipe.run_image_analysis,
                                                       timing, images, args.vision_model, log)
        log("slide_plan: running baseline brief -> summary -> normalized context (production prompts)")
        baseline_brief = _timed_baseline("slide_plan:brief", pipe.run_project_brief,
                                          timing, pages, baseline_image_results, args.text_model, log)
        baseline_summary = _timed_baseline("slide_plan:summary", pipe.run_executive_summary,
                                            timing, baseline_brief, args.text_model, log)
        baseline_normalized = _timed_baseline("slide_plan:normalize", pipe.run_normalized_context,
                                               timing, baseline_brief, baseline_summary, args.presentation_model, log)
        sweep_stage("slide_plan", pipe.run_slide_plan, (baseline_normalized, args.presentation_model, log), log, out, timing)

    # --- Slide Content: one representative slide from a real baseline plan ---
    if "slide_content" in args.stages:
        if baseline_normalized is None:
            if baseline_image_results is None:
                log("slide_content: running baseline image analysis for shared evidence")
                baseline_image_results = _timed_baseline("slide_content:image_analysis", pipe.run_image_analysis,
                                                           timing, images, args.vision_model, log)
            log("slide_content: running baseline brief -> summary -> normalized context")
            baseline_brief = _timed_baseline("slide_content:brief", pipe.run_project_brief,
                                              timing, pages, baseline_image_results, args.text_model, log)
            baseline_summary = _timed_baseline("slide_content:summary", pipe.run_executive_summary,
                                                timing, baseline_brief, args.text_model, log)
            baseline_normalized = _timed_baseline("slide_content:normalize", pipe.run_normalized_context,
                                                   timing, baseline_brief, baseline_summary, args.presentation_model, log)
        log("slide_content: running baseline slide plan for a representative slide")
        baseline_plan = _timed_baseline("slide_content:slide_plan", pipe.run_slide_plan,
                                         timing, baseline_normalized, args.presentation_model, log)
        if baseline_plan and baseline_plan.get("slides"):
            rep_slide = baseline_plan["slides"][0]
            enrichment = _timed_baseline("slide_content:enrichment", pipe.run_slide_enrichment,
                                          timing, rep_slide, baseline_normalized, args.presentation_model, log)
            sweep_stage("slide_content", pipe.run_slide_content,
                        (rep_slide, enrichment, args.presentation_model, log), log, out, timing)
        else:
            log("slide_content: baseline slide plan failed, skipping this stage's sweep")

    total_elapsed = round(time.time() - run_started, 1)
    timing["total_seconds"] = total_elapsed
    timing["total_minutes"] = round(total_elapsed / 60, 1)

    summary_path = OUT_DIR / "sweep_summary.json"
    summary_path.write_text(json.dumps(
        {"per_stage": {s: timing.get(s, {}) for s in out}, "timing": timing}, indent=2), encoding="utf-8")
    print(f"\nAll done in {timing['total_minutes']} minutes ({total_elapsed}s). Per-stage results in {OUT_DIR}/<stage>_sweep.json")
    for s in out:
        t = timing.get(s, {})
        print(f"  {s}: {t.get('variants_tested', '?')} variants, {t.get('sweep_seconds', '?')}s "
              f"(avg {t.get('avg_seconds_per_variant', '?')}s/variant, range {t.get('min_seconds', '?')}-{t.get('max_seconds', '?')}s)")
    print("Upload those files for side-by-side comparison, same pattern as before.")


if __name__ == "__main__":
    main()
