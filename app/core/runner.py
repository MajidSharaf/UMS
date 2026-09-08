"""Executes a single pipeline stage: render prompt -> call Ollama -> parse
-> validate -> persist (SQLite row + raw JSON blob + CSV log line).

This is the one place every experiment axis converges: which prompt,
which model, what decoding constraint (freeform / loose JSON /
schema-constrained), what sampling settings, and what context packaging
variant was used to build the input. Change any one of those and re-run
the same input to compare.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import config, db, ollama_client, results_log, schemas
from .prompts import Prompt

RESPONSE_FORMATS = ("freeform", "json_loose", "json_schema")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class RunResult:
    run_id: str
    stage: str
    model: str
    prompt: Prompt
    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    input_ref: str = ""
    output_text: str = ""
    parsed_json: Optional[dict] = None
    is_valid_json: bool = False
    schema_errors: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None


def _extract_json(text: str) -> tuple[Optional[dict], Optional[str]]:
    text = text.strip()
    if not text:
        return None, "empty response"
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip()), None
        except json.JSONDecodeError as exc:
            return None, f"fenced block not valid JSON: {exc}"
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1]), None
        except json.JSONDecodeError as exc:
            return None, f"brace-extracted text not valid JSON: {exc}"
    return None, "no JSON object found in output"


def _resolve_format(response_format: str, stage: str) -> Any:
    if response_format == "freeform":
        return None
    if response_format == "json_loose":
        return "json"
    if response_format == "json_schema":
        return schemas.json_schema_for(stage)
    raise ValueError(f"Unknown response_format {response_format!r}")


def run_stage(
    stage: str,
    prompt: Prompt,
    model: str,
    context: dict[str, str],
    *,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    response_format: str = "json_schema",
    images: Optional[list[Path]] = None,
    input_ref: str = "",
    input_summary: str = "",
    context_variant: str = "",
    objective_id: Optional[str] = None,
    chain_run_id: Optional[str] = None,
    chain_sequence: Optional[int] = None,
) -> RunResult:
    user_prompt = prompt.render_user_prompt(**context)
    run_id = db.new_id()

    result = RunResult(
        run_id=run_id, stage=stage, model=model, prompt=prompt,
        system_prompt=prompt.system_prompt, user_prompt=user_prompt,
        temperature=temperature, input_ref=input_ref,
    )

    fmt = _resolve_format(response_format, stage)
    schema_errors: list[str] = []
    try:
        gen = ollama_client.generate(
            model=model,
            system_prompt=prompt.system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            images=images,
            response_format=fmt,
        )
        result.output_text = gen.output_text
        result.latency_ms = gen.latency_ms
        parsed, parse_error = _extract_json(gen.output_text)
        result.parsed_json = parsed
        if parse_error:
            schema_errors.append(parse_error)
        elif isinstance(parsed, dict):
            valid, errors = schemas.validate_stage_output(stage, parsed)
            result.is_valid_json = valid
            schema_errors.extend(errors)
        else:
            schema_errors.append(f"output parsed but is not a JSON object (got {type(parsed).__name__})")
    except ollama_client.OllamaError as exc:
        result.error = str(exc)
        schema_errors.append(str(exc))

    result.schema_errors = schema_errors

    # Persistence is best-effort and must never take down an unattended run:
    # a transient SQLite lock (e.g. OneDrive syncing the DB file mid-write,
    # or the Streamlit app opening its own connection at the same time) or a
    # momentary disk hiccup should cost this one log entry, not the rest of
    # a multi-hour batch. Each of the three writes is independent so one
    # failing doesn't block the others.
    try:
        db.insert_run({
            "id": run_id,
            "created_at": time.time(),
            "objective_id": objective_id,
            "stage": stage,
            "prompt_stage": prompt.stage,
            "prompt_name": prompt.name,
            "model": model,
            "temperature": temperature,
            "input_ref": input_ref,
            "input_summary": input_summary,
            "system_prompt": prompt.system_prompt,
            "user_prompt": user_prompt,
            "raw_output": result.output_text,
            "parsed_json": json.dumps(result.parsed_json) if result.parsed_json is not None else None,
            "is_valid_json": int(result.is_valid_json),
            "schema_errors": json.dumps(schema_errors) if schema_errors else None,
            "error": result.error,
            "latency_ms": result.latency_ms,
            "chain_run_id": chain_run_id,
            "chain_sequence": chain_sequence,
        })
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see comment above
        print(f"WARNING: failed to write run {run_id} to lab.db: {exc}", file=sys.stderr)

    try:
        db.save_raw_output(run_id, {
            "run_id": run_id,
            "stage": stage,
            "model": model,
            "prompt": {"stage": prompt.stage, "name": prompt.name},
            "temperature": temperature,
            "response_format": response_format,
            "context_variant": context_variant,
            "input_ref": input_ref,
            "system_prompt": prompt.system_prompt,
            "user_prompt": user_prompt,
            "output_text": result.output_text,
            "parsed_json": result.parsed_json,
            "is_valid_json": result.is_valid_json,
            "schema_errors": schema_errors,
            "error": result.error,
            "latency_ms": result.latency_ms,
        })
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to write run {run_id} blob: {exc}", file=sys.stderr)

    try:
        excerpt = (result.output_text or "")[:300].replace("\n", " ")
        results_log.append({
            "run_id": run_id,
            "stage": stage,
            "prompt_name": prompt.name,
            "model": model,
            "temperature": temperature,
            "response_format": response_format,
            "context_variant": context_variant,
            "input_ref": input_ref,
            "is_valid_json": result.is_valid_json,
            "schema_errors": "; ".join(schema_errors),
            "latency_ms": round(result.latency_ms, 1),
            "output_excerpt": excerpt,
            "chain_run_id": chain_run_id or "",
            "chain_sequence": chain_sequence if chain_sequence is not None else "",
        })
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to append run {run_id} to results_log.csv: {exc}", file=sys.stderr)

    return result
