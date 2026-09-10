#!/usr/bin/env python3
"""Run the 5 standalone 'expert' prompts - each a full rewrite at
production's depth and structure, built around one core idea from a
top-scoring short variant (not production's text plus an addendum) -
through project_brief, then executive_summary on each resulting brief.

Usage:
    tmux new -s expert
    python run_expert_prompts.py --run-id expert
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

OUT_DIR = Path(__file__).resolve().parent / "experiments" / "expert_prompts"

VARIANTS = ["expert-cot3", "expert-strict2", "expert-risk1", "expert-complete2", "expert-risk2"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--text-model", default="gemma4:26b")
    parser.add_argument("--vision-model", default="qwen3-vl:8b-instruct")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_DIR / run_id

    pipe.seed_new_stage_prompts()
    pages = list(ds.list_pages())
    images = list(ds.list_images())

    print(f"Plan: {len(VARIANTS)} standalone expert prompts x (project_brief + executive_summary), "
          f"real evidence ({len(pages)} pages, {len(images)} images)")
    print(f"run_id: {run_id}   output: {run_dir}")
    if args.dry_run:
        return

    if run_dir.exists():
        print(f"ERROR: {run_dir} already exists.")
        sys.exit(1)
    if not ollama_client.is_reachable():
        print("ERROR: Ollama not reachable.")
        sys.exit(1)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    log("running shared baseline image analysis (production prompt, all images) - reused by every expert prompt")
    started = time.time()
    image_results = pipe.run_image_analysis(images, args.vision_model, log)
    log(f"baseline image analysis done in {round(time.time() - started, 1)}s")

    es_prompt = prompts.get_prompt("executive_summary", "production")
    results = {}

    for name in VARIANTS:
        brief_prompt = prompts.get_prompt("project_brief", name)
        if brief_prompt is None:
            log(f"WARNING: prompt '{name}' not found, skipping")
            continue
        log(f"{name}: running project_brief")
        t0 = time.time()
        brief = pipe.run_project_brief(pages, image_results, args.text_model, log, prompt=brief_prompt)
        brief_elapsed = round(time.time() - t0, 1)
        if brief is None:
            log(f"{name}: project_brief FAILED, skipping executive_summary")
            results[name] = {"brief": None, "brief_elapsed_seconds": brief_elapsed,
                              "executive_summary": None, "summary_elapsed_seconds": None}
            continue
        log(f"{name}: project_brief ok ({brief_elapsed}s), running executive_summary")
        t1 = time.time()
        summary = pipe.run_executive_summary(brief, args.text_model, log, prompt=es_prompt)
        summary_elapsed = round(time.time() - t1, 1)
        log(f"{name}: executive_summary ok ({summary_elapsed}s)")
        results[name] = {
            "brief": brief, "brief_elapsed_seconds": brief_elapsed,
            "executive_summary": summary, "summary_elapsed_seconds": summary_elapsed,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "expert_prompts_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    for name, r in results.items():
        ok = "ok" if r["brief"] else "FAILED"
        print(f"  {name}: brief {ok} ({r['brief_elapsed_seconds']}s), "
              f"summary {'ok' if r.get('executive_summary') else 'n/a'} ({r.get('summary_elapsed_seconds')}s)")


if __name__ == "__main__":
    main()
