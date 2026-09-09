#!/usr/bin/env python3
"""Split one sweep stage's packed JSON into one readable file per prompt
variant, for manual side-by-side reading (e.g. in a text editor or VS Code's
diff view) rather than scrolling one giant JSON array.

Usage:
    python extract_briefs.py --run-id round2                     # project_brief, the default
    python extract_briefs.py --run-id round2 --stage slide_plan
    python extract_briefs.py --run-id round2 --stage project_brief --out-dir ~/Desktop/briefs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SWEEP_DIR = Path(__file__).resolve().parent / "experiments" / "prompt_sweep"


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", default="project_brief",
                         choices=["image_analysis", "project_brief", "slide_plan", "slide_content"])
    parser.add_argument("--out-dir", default=None,
                         help="default: experiments/prompt_sweep/<run-id>/<stage>_readable/")
    args = parser.parse_args()

    sweep_path = SWEEP_DIR / args.run_id / f"{args.stage}_sweep.json"
    if not sweep_path.exists():
        print(f"ERROR: {sweep_path} does not exist")
        sys.exit(1)
    entries = json.loads(sweep_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir) if args.out_dir else SWEEP_DIR / args.run_id / f"{args.stage}_readable"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for e in entries:
        name = e["prompt_name"]
        tag = "PRODUCTION_BASELINE" if e.get("is_production_baseline") else "variant"
        fname = f"{_safe_filename(name)}__{tag}.json"
        if not e.get("output"):
            (out_dir / fname).write_text(f"// call failed, no output\n", encoding="utf-8")
            continue
        try:
            parsed = json.loads(e["output"])
            text = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            text = e["output"]  # not valid JSON - dump raw so it's still inspectable
        header = f"// prompt: {name}   elapsed: {e.get('elapsed_seconds', '?')}s   production_baseline: {e.get('is_production_baseline', False)}\n\n"
        (out_dir / fname).write_text(header + text, encoding="utf-8")
        written += 1

    print(f"Wrote {written}/{len(entries)} readable files to {out_dir}")
    print("Open that folder in a text editor / VS Code and compare side by side.")


if __name__ == "__main__":
    main()
