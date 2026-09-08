#!/usr/bin/env python3
"""AI prompt generator: asks your local Ollama text model to write new
prompt variants for each pipeline stage and saves them straight into
prompts_library/ (plain text, no YAML). Unattended - run it, then check
the Prompts page or the .txt files to see what it wrote.

Every generation call includes the stage's production baseline prompt as
a reference example ("here's one that already works, draw on its ideas"),
plus one of 7 distinct design angles (terse, anti-hallucination-strict,
chain-of-thought, checklist, persona, risk-first, completeness-first) so
variants differ on purpose rather than being reworded restatements of
each other. Batches stay small (a handful of variants per call) and cycle
through the angles repeatedly rather than asking for e.g. 100 prompts in
one response - a small local model asked for a huge batch in one shot is
likely to truncate or degrade in quality.

Usage:
    python generate_prompts.py                        # 100 per stage, 700 total
    python generate_prompts.py --count 20               # 20 per stage instead
    python generate_prompts.py --stages project_brief narrative
    python generate_prompts.py --model qwen2.5:7b-instruct
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import cycle, islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from core import ollama_client  # noqa: E402
from core import prompt_generation as pg  # noqa: E402
from core import prompts as prompts_mod  # noqa: E402
from core import stages  # noqa: E402

BATCH_SIZE = 4          # variants requested per Ollama call - kept small for reliability
MAX_CALL_MULTIPLIER = 3  # safety cap: give up on a stage after this many times the "ideal" call count


def pick_model(args) -> str | None:
    if args.model:
        return args.model
    models = [m.name for m in ollama_client.list_models() if not m.is_vision]
    return models[0] if models else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100, help="target variants per stage (default: 100, 700 total across 7 stages)")
    parser.add_argument("--stages", nargs="*", default=None, help="subset of stage ids (default: all 7)")
    parser.add_argument("--model", default=None, help="text model to use (default: first non-vision model found)")
    args = parser.parse_args()

    if not ollama_client.is_reachable():
        print("ERROR: Ollama is not reachable. Start it with `ollama serve` first.")
        sys.exit(1)

    model = pick_model(args)
    if not model:
        print("ERROR: no text model found via `ollama list`. Pull one first.")
        sys.exit(1)
    print(f"model: {model}")

    stage_ids = args.stages or stages.ordered_stage_ids()
    ideal_calls_per_stage = -(-args.count // BATCH_SIZE)  # ceil
    max_calls_per_stage = ideal_calls_per_stage * MAX_CALL_MULTIPLIER
    print(f"target: {args.count} variants/stage, {BATCH_SIZE} per call, up to {max_calls_per_stage} "
          f"calls/stage before giving up on a stage\n")

    started = time.time()
    total_saved, total_rejected, total_calls = 0, 0, 0

    for stage_id in stage_ids:
        existing_names = prompts_mod.list_prompt_names(stage_id)
        stage_saved, stage_calls = 0, 0
        print(f"--- {stage_id} (starting with {len(existing_names)} existing) ---")
        for angle_label, angle_instruction in islice(cycle(pg.ANGLES), max_calls_per_stage):
            if stage_saved >= args.count:
                break
            stage_calls += 1
            total_calls += 1
            saved, errors = pg.generate_batch(
                stage_id, model, angle_label, angle_instruction, BATCH_SIZE, existing_names,
            )
            existing_names.extend(p.name for p in saved)
            stage_saved += len(saved)
            total_saved += len(saved)
            total_rejected += len(errors)
            elapsed = time.time() - started
            print(f"  call {stage_calls:>3} [{angle_label:<24}] +{len(saved)} saved, {stage_saved}/{args.count}" +
                  (f", rejected: {'; '.join(errors)}" if errors else "") +
                  f"  ({elapsed/60:.1f}m elapsed)")
        if stage_saved < args.count:
            print(f"  gave up after {stage_calls} calls, only reached {stage_saved}/{args.count} "
                  f"(model may be struggling with the required format)")
        print(f"  stage total: {stage_saved}\n")

    print(f"Done. {total_saved} prompts saved across prompts_library/ in {total_calls} calls, "
          f"{total_rejected} rejected (missing placeholder, empty fields, or name collision - not saved).\n"
          f"Open the Prompts page, or Run Experiment / run_batch_experiment.py to try them.")


if __name__ == "__main__":
    main()
