#!/usr/bin/env python3
"""Unattended batch experiment: runs every prompt variant in the library
against every stage, using each stage's real natural input, and logs
everything to the same experiments/lab.db + results_log.csv + runs/*.json
the Streamlit app already reads. Run this, walk away, come back and open
the Compare / Results Log pages (or just read results_log.csv directly).

Design: for stages 3-7 (which consume an upstream stage's output rather
than a raw page/image), only the "production" prompt variant's output at
each stage is propagated forward to build the next stage's input - so
every variant of every stage is tested against the *same*, realistic
input, not a different one each time. Stages 1-2 (image_analysis,
page_analysis) run every variant against every real image/page - which
does NOT scale to a large prompt library. 100 variants x 47 images is
4,700 calls on its own. Two-phase workflow once your prompt library is
large:
  1. SEARCH on a small sample to find the winner fast:
       python run_batch_experiment.py --limit 5
  2. CONFIRM the winner (only) against the full document, e.g. via the
     Pipeline page after promoting it in Pipeline Settings, or:
       python run_batch_experiment.py --only-production

Usage:
    python run_batch_experiment.py                       # everything, all pages/images
    python run_batch_experiment.py --limit 5              # search phase: first 5 pages/images only
    python run_batch_experiment.py --only-production      # confirm phase: just the production variant, full document
    python run_batch_experiment.py --stages image_analysis page_analysis
    python run_batch_experiment.py --text-model qwen2.5:7b-instruct --vision-model llava:7b
    python run_batch_experiment.py --dry-run              # print the plan and a time estimate, call nothing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from core import data_sources as ds  # noqa: E402
from core import db, ollama_client  # noqa: E402
from core import prompts as prompts_mod  # noqa: E402
from core import runner, stages  # noqa: E402

MAX_SLIDE_CONTENT = 8
ASSUMED_SECONDS_PER_CALL = 8  # very rough - real speed depends heavily on your model/hardware


def pick_models(args) -> tuple[str | None, str | None]:
    models = ollama_client.list_models()
    text_models = [m.name for m in models if not m.is_vision]
    vision_models = [m.name for m in models if m.is_vision]
    return (
        args.text_model or (text_models[0] if text_models else None),
        args.vision_model or (vision_models[0] if vision_models else None),
    )


def raw_pages_text(pages) -> str:
    return "\n\n".join(ds.page_context_full(p) for p in pages)


def production_variant(variants):
    return next((p for p in variants if p.name == "production"), variants[0] if variants else None)


class Runner:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.total = 0
        self.started = time.time()

    def run(self, stage_id, prompt, model, context, **kwargs) -> "runner.RunResult | None":
        self.total += 1
        elapsed = time.time() - self.started
        label = f"[{self.total:>4}] {stage_id:<22} {prompt.display_name:<20} {model:<24}"
        if self.dry_run:
            print(f"{label} (dry run)")
            return None
        result = runner.run_stage(stage_id, prompt, model, context, temperature=0.0,
                                   response_format="json_schema", **kwargs)
        badge = "OK " if result.is_valid_json else "BAD"
        print(f"{label} {badge} {result.latency_ms:>7.0f}ms  ({elapsed/60:.1f}m elapsed)")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="max pages/images to test per leaf stage (default: all)")
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--stages", nargs="*", default=None, help="subset of stage ids to run (default: all 7)")
    parser.add_argument("--only-production", action="store_true",
                         help="use only the 'production' prompt variant per stage (the confirm phase)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and a time estimate, call nothing")
    args = parser.parse_args()

    def get_variants(stage_id: str):
        all_variants = prompts_mod.list_prompts(stage_id)
        if args.only_production:
            prod = production_variant(all_variants)
            return [prod] if prod else []
        return all_variants

    if not args.dry_run and not ollama_client.is_reachable():
        print("ERROR: Ollama is not reachable. Start it with `ollama serve` first.")
        sys.exit(1)

    db.init_db()
    text_model, vision_model = pick_models(args)
    print(f"text model:   {text_model or '(none found)'}")
    print(f"vision model: {vision_model or '(none found - image_analysis will be skipped)'}")

    stage_ids = args.stages or stages.ordered_stage_ids()
    pages = list(ds.list_pages())
    images = list(ds.list_images())
    if args.limit:
        pages, images = pages[:args.limit], images[:args.limit]

    plan = []
    for sid in stage_ids:
        sdef = stages.STAGES_BY_ID[sid]
        n_variants = len(get_variants(sid))
        model = vision_model if sdef.vision_required else text_model
        if sdef.input_kind == "image":
            n_items = len(images)
        elif sdef.input_kind == "page":
            n_items = len(pages)
        elif sdef.id == "slide_content":
            n_items = MAX_SLIDE_CONTENT  # upper bound; actual count depends on slide_plan's output
        else:
            n_items = 1
        plan.append((sid, n_variants, n_items, model))
    total_planned = sum(v * i for _, v, i, m in plan if m)
    est_hours = total_planned * ASSUMED_SECONDS_PER_CALL / 3600
    print(f"\nPlan ({total_planned} calls total, very roughly {est_hours:.1f}h at "
          f"{ASSUMED_SECONDS_PER_CALL}s/call - actual speed depends on your model/hardware):")
    for sid, v, i, model in plan:
        print(f"  {sid:<22} {v} variant(s) x {i} item(s) = {v*i:>4} calls  [{model or 'SKIPPED - no model'}]")
    if total_planned > 500:
        print(f"\nThat's a lot. Consider `--limit 5` to search on a small sample first, then "
              f"`--only-production` (after promoting the winner in Pipeline Settings) to confirm "
              f"against the full document.")
    print()

    if args.dry_run:
        return

    r = Runner(dry_run=False)
    propagated: dict[str, object] = {}  # stage_id -> production output (dict, or dict keyed by ref for leaf stages)

    if "image_analysis" in stage_ids and vision_model:
        variants = get_variants("image_analysis")
        out = {}
        for img in images:
            for prompt in variants:
                result = r.run("image_analysis", prompt, vision_model, {"CONTEXT": ds.image_prompt_context(img)},
                                images=[img.image_path], input_ref=img.ref,
                                input_summary=f"page {img.page_number}", context_variant="image_metadata")
                if prompt.name == "production" and result and result.is_valid_json:
                    out[img.ref] = result.parsed_json
        propagated["image_analysis"] = out

    if "page_analysis" in stage_ids and text_model:
        variants = get_variants("page_analysis")
        out = {}
        for page in pages:
            page_refs = {i.ref for i in ds.images_on_page(page.page_number)}
            image_evidence = json.dumps([v for ref, v in propagated.get("image_analysis", {}).items() if ref in page_refs])
            for prompt in variants:
                result = r.run("page_analysis", prompt, text_model,
                                {"CONTEXT": ds.page_context_full(page), "IMAGE_EVIDENCE": image_evidence},
                                input_ref=page.ref, input_summary=page.heading or f"page {page.page_number}",
                                context_variant="full")
                if prompt.name == "production" and result and result.is_valid_json:
                    out[page.ref] = result.parsed_json
        propagated["page_analysis"] = out

    if "evidence_consolidation" in stage_ids and text_model:
        variants = get_variants("evidence_consolidation")
        page_analyses = list(propagated.get("page_analysis", {}).values())
        context_str = json.dumps(page_analyses) if page_analyses else raw_pages_text(pages)
        best = None
        for prompt in variants:
            result = r.run("evidence_consolidation", prompt, text_model, {"CONTEXT": context_str},
                            input_ref=f"pages:{[p.page_number for p in pages]}",
                            input_summary="page_analyses" if page_analyses else "raw_pages",
                            context_variant="page_analyses" if page_analyses else "raw_pages")
            if prompt.name == "production" and result and result.is_valid_json:
                best = result.parsed_json
        propagated["evidence_consolidation"] = best

    if "project_brief" in stage_ids and text_model:
        variants = get_variants("project_brief")
        source = (propagated.get("evidence_consolidation")
                  or list(propagated.get("page_analysis", {}).values())
                  or None)
        context_str = json.dumps(source) if source else raw_pages_text(pages)
        best = None
        for prompt in variants:
            result = r.run("project_brief", prompt, text_model, {"CONTEXT": context_str},
                            input_ref=f"pages:{[p.page_number for p in pages]}", input_summary="upstream",
                            context_variant="evidence" if source else "raw_pages")
            if prompt.name == "production" and result and result.is_valid_json:
                best = result.parsed_json
        propagated["project_brief"] = best

    if "narrative" in stage_ids and text_model:
        variants = get_variants("narrative")
        source = propagated.get("project_brief") or propagated.get("evidence_consolidation")
        context_str = json.dumps(source) if source else raw_pages_text(pages)
        best = None
        for prompt in variants:
            result = r.run("narrative", prompt, text_model, {"CONTEXT": context_str},
                            input_ref=f"pages:{[p.page_number for p in pages]}", input_summary="upstream",
                            context_variant="project_brief" if source else "raw_pages")
            if prompt.name == "production" and result and result.is_valid_json:
                best = result.parsed_json
        propagated["narrative"] = best

    if "slide_plan" in stage_ids and text_model:
        variants = get_variants("slide_plan")
        source = propagated.get("narrative") or propagated.get("project_brief")
        context_str = json.dumps(source) if source else raw_pages_text(pages)
        best = None
        for prompt in variants:
            result = r.run("slide_plan", prompt, text_model, {"CONTEXT": context_str},
                            input_ref=f"pages:{[p.page_number for p in pages]}", input_summary="upstream",
                            context_variant="narrative" if source else "raw_pages")
            if prompt.name == "production" and result and result.is_valid_json:
                best = result.parsed_json
        propagated["slide_plan"] = best

    if "slide_content" in stage_ids and text_model and propagated.get("slide_plan"):
        variants = get_variants("slide_content")
        slides = propagated["slide_plan"].get("slides", [])[:MAX_SLIDE_CONTENT]
        for idx, slide in enumerate(slides, start=1):
            context_str = json.dumps({"slide": slide, "narrative": propagated.get("narrative") or {}})
            for prompt in variants:
                r.run("slide_content", prompt, text_model, {"CONTEXT": context_str},
                      input_ref=f"slide:{slide.get('title', '')[:30] or idx}",
                      input_summary=slide.get("title", ""), context_variant="slide_plan_item")

    print(f"\nDone. {r.total} runs logged to experiments/lab.db, experiments/results_log.csv, "
          f"experiments/runs/*.json.\nOpen the Streamlit app (Compare / Results Log pages) to review, "
          f"or read experiments/results_log.csv directly.")


if __name__ == "__main__":
    main()
