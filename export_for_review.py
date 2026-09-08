#!/usr/bin/env python3
"""Exports logged runs into one clean JSON file for a human (or a more
capable model than the local judge) to review directly - full inputs and
full outputs, not the truncated excerpts in results_log.csv.

Usage:
    python export_for_review.py                        # every valid run
    python export_for_review.py --stages project_brief narrative
    python export_for_review.py --top-per-prompt 3       # cap runs per (stage, prompt) to keep the file small
    python export_for_review.py --out review.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from core import db, stages  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="*", default=None, help="subset of stage ids (default: all)")
    parser.add_argument("--top-per-prompt", type=int, default=None,
                         help="cap how many runs per (stage, prompt) get included, to keep the file small")
    parser.add_argument("--valid-only", action="store_true", default=True,
                         help="only include runs that passed schema validation (default: on)")
    parser.add_argument("--out", default="review_export.json")
    args = parser.parse_args()

    stage_ids = args.stages or stages.ordered_stage_ids()
    export = {}
    total = 0

    for stage_id in stage_ids:
        runs = db.list_runs(stage=stage_id)
        if args.valid_only:
            runs = [r for r in runs if r["is_valid_json"]]

        by_prompt: dict[str, list] = {}
        for r in runs:
            by_prompt.setdefault(r["prompt_name"], []).append(r)

        stage_export = []
        for prompt_name, prompt_runs in by_prompt.items():
            chosen = prompt_runs[:args.top_per_prompt] if args.top_per_prompt else prompt_runs
            for r in chosen:
                stage_export.append({
                    "run_id": r["id"],
                    "prompt_name": r["prompt_name"],
                    "model": r["model"],
                    "input_ref": r["input_ref"],
                    "input_given_to_model": r["user_prompt"],
                    "output": r["parsed_json"] or r["raw_output"],
                })
                total += 1
        export[stage_id] = stage_export

    out_path = Path(args.out)
    out_path.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"Exported {total} runs across {len(stage_ids)} stage(s) to {out_path} ({size_kb:.0f} KB).")
    print("Upload this file to Claude and ask it to review/compare/rank the outputs per stage.")


if __name__ == "__main__":
    main()
