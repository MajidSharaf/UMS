#!/usr/bin/env python3
"""Full pipeline run using the winning prompt per stage from the round2 sweep,
instead of the production prompt everywhere. Produces a second, comparable
deliverable (brief + deck) built from the actual best-performing wording per
stage, so it can be shown side-by-side against the production-prompt run.

Winners, from experiments/prompt_sweep/round2 (ranked by schema-field
completeness, then content depth, among prompts with genuinely distinct
output - not just a top-of-list duplicate of another variant):
    image_analysis  -> strict-3   (8/8 fields, most detailed of the 23)
    project_brief   -> cot-3      (40/40 fields, most detailed of the 23)
    slide_plan      -> complete-2 (most detailed of only 8 truly distinct outputs)
    slide_content   -> complete-1 (most detailed of the 23)
executive_summary / normalized_context / slide_enrichment were never swept
(only the production prompt exists for them), so they stay on production.

Usage:
    tmux new -s optimized
    python run_optimized_pipeline.py --run-id optimized
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from core import prompts  # noqa: E402
import run_full_pipeline as pipe  # noqa: E402

WINNERS = {
    "image_analysis": "strict-3",
    "project_brief": "cot-3",
    "slide_plan": "complete-2",
    "slide_content": "complete-1",
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=None)
    args, extra = parser.parse_known_args()

    # Monkey-patch the four swept stages' default prompt to the winner,
    # by wrapping run_full_pipeline's run_* functions with the winner baked in.
    orig = {}
    for stage, winner_name in WINNERS.items():
        winner_prompt = prompts.get_prompt(stage, winner_name)
        if winner_prompt is None:
            print(f"ERROR: prompt '{winner_name}' not found for stage '{stage}'")
            sys.exit(1)
        fn_name = f"run_{stage}"
        orig_fn = getattr(pipe, fn_name)
        orig[fn_name] = orig_fn

        def make_wrapper(orig_fn=orig_fn, winner_prompt=winner_prompt):
            def wrapper(*a, **kw):
                kw.setdefault("prompt", winner_prompt)
                return orig_fn(*a, **kw)
            return wrapper

        setattr(pipe, fn_name, make_wrapper())
        print(f"{stage}: using winning prompt '{winner_name}' instead of production")

    sys.argv = [sys.argv[0]] + extra + (["--run-id", args.run_id] if args.run_id else ["--run-id", "optimized"])
    pipe.main()


if __name__ == "__main__":
    main()
