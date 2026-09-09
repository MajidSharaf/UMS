#!/usr/bin/env python3
"""Re-verify determinism (temperature 0) under THIS round's real production
models and real schema-constrained decoding. The earlier local-model round
confirmed 46/46 repeat-run pairs were byte-for-byte identical, but that was
under different (smaller) models - this reruns the same check against
Gemma 4 26B / Qwen3-VL 8B / Qwen 3.5 35B A3B specifically, on the production
prompt at every swept stage, so the claim holds for the models actually used
in round2 rather than being inherited from the earlier round.

Usage:
    tmux new -s determinism
    python check_determinism.py --run-id determinism_recheck
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

OUT_DIR = Path(__file__).resolve().parent / "experiments" / "determinism_recheck"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--text-model", default="gemma4:26b")
    parser.add_argument("--presentation-model", default="qwen3.5:35b-a3b")
    parser.add_argument("--vision-model", default="qwen3-vl:8b-instruct")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_DIR / run_id

    pipe.seed_new_stage_prompts()
    pages = list(ds.list_pages())
    images = list(ds.list_images())
    rep_image = images[0]

    if args.dry_run:
        print(f"Plan: run_id={run_id}, output={run_dir}")
        print("Would run production prompt twice each for: image_analysis (1 image), "
              "project_brief, slide_plan, slide_content, and diff the pairs.")
        return

    if run_dir.exists():
        print(f"ERROR: {run_dir} already exists.")
        sys.exit(1)
    if not ollama_client.is_reachable():
        print("ERROR: Ollama not reachable.")
        sys.exit(1)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    results = {}

    log("image_analysis: run 1/2")
    ia1 = pipe.run_image_analysis([rep_image], args.vision_model, log)
    log("image_analysis: run 2/2")
    ia2 = pipe.run_image_analysis([rep_image], args.vision_model, log)
    results["image_analysis"] = {"run1": ia1, "run2": ia2, "identical": json.dumps(ia1) == json.dumps(ia2)}

    log("project_brief: run 1/2")
    pb1 = pipe.run_project_brief(pages, ia1, args.text_model, log)
    log("project_brief: run 2/2")
    pb2 = pipe.run_project_brief(pages, ia1, args.text_model, log)  # same evidence both times
    results["project_brief"] = {"run1": pb1, "run2": pb2, "identical": json.dumps(pb1) == json.dumps(pb2)}

    summary1 = pipe.run_executive_summary(pb1, args.text_model, log)
    norm1 = pipe.run_normalized_context(pb1, summary1, args.presentation_model, log)

    log("slide_plan: run 1/2")
    sp1 = pipe.run_slide_plan(norm1, args.presentation_model, log)
    log("slide_plan: run 2/2")
    sp2 = pipe.run_slide_plan(norm1, args.presentation_model, log)
    results["slide_plan"] = {"run1": sp1, "run2": sp2, "identical": json.dumps(sp1) == json.dumps(sp2)}

    if sp1 and sp1.get("slides"):
        slide = sp1["slides"][0]
        enrich1 = pipe.run_slide_enrichment(slide, norm1, args.presentation_model, log)
        log("slide_content: run 1/2")
        sc1 = pipe.run_slide_content(slide, enrich1, args.presentation_model, log)
        log("slide_content: run 2/2")
        sc2 = pipe.run_slide_content(slide, enrich1, args.presentation_model, log)
        results["slide_content"] = {"run1": sc1, "run2": sc2, "identical": json.dumps(sc1) == json.dumps(sc2)}

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "determinism_recheck.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Determinism recheck (this round's real production models) ===")
    for stage, r in results.items():
        print(f"  {stage}: {'IDENTICAL' if r['identical'] else 'DIFFERENT'}")
    print(f"Wrote {run_dir / 'determinism_recheck.json'}")


if __name__ == "__main__":
    main()
