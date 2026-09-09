#!/usr/bin/env python3
"""Rank prompt-sweep results without dumping raw JSON into a terminal/chat.

Every sweep stage now uses schema-constrained decoding, so the *shape* of a
successful output is already guaranteed correct (that was the whole point of
the schema fix) - what's left to compare is how much real, useful content
each prompt variant produced: how many fields actually got filled in, how
much detail is in them, and whether the call failed outright.

Usage:
    python analyze_sweep.py --run-id round2
    python analyze_sweep.py --run-id round2 --stage project_brief
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))
from core import production_schemas as schemas  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "experiments" / "prompt_sweep"

STAGE_KEYS = {
    "image_analysis": schemas.image_analysis_schema()["required"],
    "project_brief": schemas.BRIEF_KEYS,
    "slide_plan": ["presentationSummary", "slides"],
    "slide_content": ["title", "body", "speakerNotes"],
}


def _non_empty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, dict)):
        return len(value) > 0
    return True


def score_output(stage: str, parsed: dict) -> dict:
    keys = STAGE_KEYS.get(stage, list(parsed.keys()))
    filled = [k for k in keys if _non_empty(parsed.get(k))]
    return {
        "fields_filled": f"{len(filled)}/{len(keys)}",
        "fields_filled_frac": len(filled) / len(keys) if keys else 0.0,
        "char_length": len(json.dumps(parsed, ensure_ascii=False)),
        "has_action_key": "action" in parsed,  # sanity check for the old tool-call-hijack bug
    }


def analyze_stage(stage: str, run_dir: Path) -> None:
    path = run_dir / f"{stage}_sweep.json"
    if not path.exists():
        print(f"{stage}: no sweep file at {path}, skipping")
        return
    entries = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    failures = []
    hijacked = []
    for e in entries:
        name = e["prompt_name"]
        baseline = " [PRODUCTION]" if e.get("is_production_baseline") else ""
        if not e.get("output"):
            failures.append(name + baseline)
            continue
        try:
            parsed = json.loads(e["output"])
        except (json.JSONDecodeError, TypeError):
            failures.append(name + baseline + " (output not valid JSON)")
            continue
        s = score_output(stage, parsed)
        if s["has_action_key"]:
            hijacked.append(name + baseline)
        rows.append({
            "name": name + baseline,
            "elapsed": e.get("elapsed_seconds", "?"),
            **s,
        })

    rows.sort(key=lambda r: (r["fields_filled_frac"], r["char_length"]), reverse=True)

    print(f"\n=== {stage} ===  ({len(rows)} valid / {len(entries)} total)")
    print(f"{'variant':<28} {'fields':<8} {'chars':<7} {'sec':<6}")
    for r in rows:
        print(f"{r['name']:<28} {r['fields_filled']:<8} {r['char_length']:<7} {r['elapsed']:<6}")
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
    if hijacked:
        print(f"WARNING - tool-call-style 'action' key found in: {', '.join(hijacked)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=list(STAGE_KEYS), default=None,
                         help="omit to analyze every stage present in this run")
    args = parser.parse_args()

    run_dir = OUT_DIR / args.run_id
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)

    stages = [args.stage] if args.stage else list(STAGE_KEYS)
    for stage in stages:
        analyze_stage(stage, run_dir)


if __name__ == "__main__":
    main()
