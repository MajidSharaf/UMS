#!/usr/bin/env python3
"""Automated first-pass scoring: asks a local text model to judge each
logged run against the exact input it was given (pulled from the DB, not
re-derived), so scoring checks faithfulness to real evidence rather than
guessing. Writes scores into the same `scores` table the Compare page's
leaderboard already reads, plus a standalone CSV for pasting into a report.

This is a fast triage tool, not a rigorous evaluation: a small local model
judging another local model's output has real limits (it can miss subtle
hallucinations, and it wasn't itself validated as a good judge). Treat the
output as "which prompts look strongest," not a certified grade.

Usage:
    python evaluate_outputs.py                      # sample 8 runs/stage, judge them
    python evaluate_outputs.py --sample-size 15       # bigger sample per stage
    python evaluate_outputs.py --all                  # judge every valid run (slow - see the printed estimate)
    python evaluate_outputs.py --stages project_brief narrative
    python evaluate_outputs.py --dry-run               # show what would be judged, call nothing
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from core import db, ollama_client, stages  # noqa: E402
from pydantic import BaseModel  # noqa: E402

ASSUMED_SECONDS_PER_CALL = 76  # tonight's real observed average, not the tool's old 8s guess
LEADERBOARD_PATH = Path(__file__).resolve().parent / "experiments" / "leaderboard.csv"


class JudgeScore(BaseModel):
    accuracy: int
    completeness: int
    hallucination: int  # 5 = none found
    overall: int
    reason: str


JUDGE_SYSTEM = (
    "You are a strict, skeptical reviewer. You will be shown the exact input material "
    "an AI assistant was given, and the JSON output it produced from that input. Score "
    "the output ONLY against that input - never against outside knowledge. Penalize any "
    "claim, fact, or figure in the output that is not supported by the input (hallucination). "
    "Penalize missing information the input clearly supports (incompleteness). "
    "Return only JSON matching the schema: accuracy (1-5), completeness (1-5), "
    "hallucination (1-5, 5 = none found), overall (1-5), reason (one short sentence)."
)


def pick_judge_model(args) -> str | None:
    if args.model:
        return args.model
    models = [m.name for m in ollama_client.list_models() if not m.is_vision]
    return models[0] if models else None


def judge_run(judge_model: str, run) -> JudgeScore | None:
    user_prompt = (
        f"INPUT GIVEN TO THE MODEL:\n{run['user_prompt']}\n\n"
        f"MODEL'S OUTPUT:\n{run['raw_output']}\n\n"
        f"Score this output against the input above."
    )
    try:
        gen = ollama_client.generate(
            model=judge_model, system_prompt=JUDGE_SYSTEM, user_prompt=user_prompt,
            temperature=0.0, response_format=JudgeScore.model_json_schema(), num_predict=300,
        )
        import json
        return JudgeScore.model_validate(json.loads(gen.output_text))
    except Exception as exc:
        print(f"  judge call failed for run {run['id']}: {exc}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-size", type=int, default=8, help="runs to judge per (stage, prompt) - default 8")
    parser.add_argument("--all", action="store_true", help="judge every valid run instead of a sample (slow)")
    parser.add_argument("--stages", nargs="*", default=None, help="subset of stage ids (default: all)")
    parser.add_argument("--model", default=None, help="judge model (default: first non-vision model found)")
    parser.add_argument("--dry-run", action="store_true", help="show the plan, call nothing")
    parser.add_argument("--seed", type=int, default=0, help="random seed for sampling, for reproducibility")
    args = parser.parse_args()

    if not args.dry_run and not ollama_client.is_reachable():
        print("ERROR: Ollama is not reachable. Start it with `ollama serve` first.")
        sys.exit(1)

    judge_model = pick_judge_model(args)
    print(f"judge model: {judge_model or '(none found)'}")

    stage_ids = args.stages or stages.ordered_stage_ids()
    random.seed(args.seed)

    to_judge = []
    for stage_id in stage_ids:
        runs = [r for r in db.list_runs(stage=stage_id) if r["is_valid_json"]]
        # group by prompt_name so the sample covers every prompt, not just whichever ran most
        by_prompt: dict[str, list] = {}
        for r in runs:
            by_prompt.setdefault(r["prompt_name"], []).append(r)
        for prompt_name, prompt_runs in by_prompt.items():
            chosen = prompt_runs if args.all else random.sample(prompt_runs, min(args.sample_size, len(prompt_runs)))
            to_judge.extend(chosen)

    est_seconds = len(to_judge) * ASSUMED_SECONDS_PER_CALL
    print(f"\nPlan: {len(to_judge)} runs to judge, ~{est_seconds/3600:.1f}h at "
          f"{ASSUMED_SECONDS_PER_CALL}s/call (tonight's real observed average)\n")

    if args.dry_run or not judge_model:
        return

    started = time.time()
    results = []
    for i, run in enumerate(to_judge, start=1):
        score = judge_run(judge_model, run)
        elapsed = time.time() - started
        if score:
            db.upsert_score(run["id"], score.accuracy, score.completeness, score.hallucination,
                             5 if run["is_valid_json"] else 1, score.overall, score.reason)
            print(f"[{i:>4}/{len(to_judge)}] {run['stage']:<22} {run['prompt_name']:<20} "
                  f"overall={score.overall}  ({elapsed/60:.1f}m elapsed)")
            results.append({"stage": run["stage"], "prompt_name": run["prompt_name"], "run_id": run["id"],
                             "accuracy": score.accuracy, "completeness": score.completeness,
                             "hallucination": score.hallucination, "overall": score.overall,
                             "reason": score.reason})
        else:
            print(f"[{i:>4}/{len(to_judge)}] {run['stage']:<22} {run['prompt_name']:<20} JUDGE FAILED "
                  f"({elapsed/60:.1f}m elapsed)")

    # Leaderboard: average overall score per (stage, prompt_name), across whatever got judged
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r["stage"], r["prompt_name"])].append(r["overall"])
    leaderboard = sorted(
        ({"stage": k[0], "prompt_name": k[1], "avg_overall": round(sum(v) / len(v), 2), "n": len(v)}
         for k, v in groups.items()),
        key=lambda r: (r["stage"], -r["avg_overall"]),
    )
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEADERBOARD_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "prompt_name", "avg_overall", "n"])
        writer.writeheader()
        writer.writerows(leaderboard)

    print(f"\nDone. {len(results)}/{len(to_judge)} judged successfully. Scores saved to the DB "
          f"(visible in the Compare page's leaderboard) and to {LEADERBOARD_PATH}.")
    print("\nLeaderboard (best prompt per stage, by average judge score):")
    seen_stages = set()
    for row in leaderboard:
        if row["stage"] not in seen_stages:
            seen_stages.add(row["stage"])
            print(f"  {row['stage']:<22} winner: {row['prompt_name']:<20} avg={row['avg_overall']} (n={row['n']})")


if __name__ == "__main__":
    main()
