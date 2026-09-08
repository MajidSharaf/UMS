"""Runs the full (or a deliberately reshaped) pipeline end to end.

This is where "pipeline structure" itself becomes an experiment axis:
every stage can be individually enabled/disabled. Disabling a stage means
its contribution is *omitted*, not stubbed — e.g. turning off Page
Analysis means Evidence Consolidation falls back to raw page text plus
whatever Image Analysis produced, which is exactly the "does this stage
even help" question. Each stage config also carries its own prompt,
model, temperature, response_format and context_variant, so a chain run
mixes as many independent variables as you want in one pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from . import data_sources as ds
from . import db
from .prompts import Prompt
from .runner import RunResult, run_stage

MAX_SLIDE_CONTENT_DEFAULT = 8


@dataclass
class StepConfig:
    stage: str
    enabled: bool
    prompt: Prompt
    model: str
    temperature: float = 0.0
    response_format: str = "json_schema"
    context_variant: str = "full"  # only used by page_analysis


@dataclass
class ChainRunSummary:
    chain_run_id: str
    steps: dict[str, list[RunResult]] = field(default_factory=dict)

    def all_results(self) -> list[RunResult]:
        return [r for results in self.steps.values() for r in results]


def _first_available(*candidates: Optional[Any]) -> tuple[Optional[Any], str]:
    for label, value in candidates:
        if value:
            return value, label
    return None, "raw-data"


def run_chain(
    page_numbers: list[int],
    step_by_stage: dict[str, StepConfig],
    *,
    label: str = "",
    objective_id: Optional[str] = None,
    max_slide_content: int = MAX_SLIDE_CONTENT_DEFAULT,
) -> ChainRunSummary:
    pages = [p for p in ds.list_pages() if p.page_number in page_numbers]
    chain_run_id = db.insert_chain_run(
        objective_id, label or "chain run",
        f"{len(pages)} pages: {sorted(page_numbers)}",
    )
    summary = ChainRunSummary(chain_run_id=chain_run_id)
    seq = 0

    # --- Stage 1: Image Analysis -------------------------------------
    image_outputs: dict[str, dict] = {}
    step = step_by_stage.get("image_analysis")
    if step and step.enabled:
        results = []
        for page in pages:
            for image in ds.images_on_page(page.page_number):
                seq += 1
                context = {"CONTEXT": ds.image_prompt_context(image)}
                result = run_stage(
                    "image_analysis", step.prompt, step.model, context,
                    temperature=step.temperature, response_format=step.response_format,
                    images=[image.image_path], input_ref=image.ref,
                    input_summary=f"page {image.page_number}",
                    context_variant="image_metadata",
                    objective_id=objective_id, chain_run_id=chain_run_id, chain_sequence=seq,
                )
                results.append(result)
                if result.parsed_json:
                    image_outputs[image.filename] = result.parsed_json
        summary.steps["image_analysis"] = results

    # --- Stage 2: Page Analysis ----------------------------------------
    page_outputs: dict[int, dict] = {}
    step = step_by_stage.get("page_analysis")
    if step and step.enabled:
        results = []
        variant_fn = ds.CONTEXT_VARIANTS.get(step.context_variant, ds.page_context_full)
        for page in pages:
            seq += 1
            page_text = variant_fn(page)
            page_images = [
                image_outputs[i.filename] for i in ds.images_on_page(page.page_number)
                if i.filename in image_outputs
            ]
            context = {
                "CONTEXT": page_text,
                "IMAGE_EVIDENCE": json.dumps(page_images) if page_images else "[]",
            }
            result = run_stage(
                "page_analysis", step.prompt, step.model, context,
                temperature=step.temperature, response_format=step.response_format,
                input_ref=page.ref, input_summary=page.heading or f"page {page.page_number}",
                context_variant=step.context_variant,
                objective_id=objective_id, chain_run_id=chain_run_id, chain_sequence=seq,
            )
            results.append(result)
            if result.parsed_json:
                page_outputs[page.page_number] = result.parsed_json
        summary.steps["page_analysis"] = results

    def _raw_pages_text() -> str:
        return "\n\n".join(ds.page_context_full(p) for p in pages)

    # --- Stage 3: Evidence Consolidation --------------------------------
    evidence_output: Optional[dict] = None
    step = step_by_stage.get("evidence_consolidation")
    if step and step.enabled:
        seq += 1
        source, source_label = _first_available(
            ("page_analyses", list(page_outputs.values())),
            ("image_analyses+raw_pages", (list(image_outputs.values()), _raw_pages_text())),
        )
        if source_label == "page_analyses":
            context_str = json.dumps(source)
        elif source_label == "image_analyses+raw_pages":
            images_json, raw_text = source
            context_str = json.dumps({"imageEvidence": images_json, "pageText": raw_text})
        else:
            context_str = _raw_pages_text()
        context = {"CONTEXT": context_str}
        result = run_stage(
            "evidence_consolidation", step.prompt, step.model, context,
            temperature=step.temperature, response_format=step.response_format,
            input_ref=f"pages:{sorted(page_numbers)}", input_summary=f"source={source_label}",
            context_variant=source_label,
            objective_id=objective_id, chain_run_id=chain_run_id, chain_sequence=seq,
        )
        summary.steps["evidence_consolidation"] = [result]
        evidence_output = result.parsed_json

    # --- Stage 4: Project Brief -----------------------------------------
    brief_output: Optional[dict] = None
    step = step_by_stage.get("project_brief")
    if step and step.enabled:
        seq += 1
        source, source_label = _first_available(
            ("evidence", evidence_output),
            ("page_analyses", list(page_outputs.values()) or None),
            ("raw_pages", _raw_pages_text()),
        )
        context_str = json.dumps(source) if source_label != "raw_pages" else source
        context = {"CONTEXT": context_str}
        result = run_stage(
            "project_brief", step.prompt, step.model, context,
            temperature=step.temperature, response_format=step.response_format,
            input_ref=f"pages:{sorted(page_numbers)}", input_summary=f"source={source_label}",
            context_variant=source_label,
            objective_id=objective_id, chain_run_id=chain_run_id, chain_sequence=seq,
        )
        summary.steps["project_brief"] = [result]
        brief_output = result.parsed_json

    # --- Stage 5: Narrative ----------------------------------------------
    narrative_output: Optional[dict] = None
    step = step_by_stage.get("narrative")
    if step and step.enabled:
        seq += 1
        source, source_label = _first_available(
            ("project_brief", brief_output),
            ("evidence", evidence_output),
            ("page_analyses", list(page_outputs.values()) or None),
        )
        context = {"CONTEXT": json.dumps(source) if source else "{}"}
        result = run_stage(
            "narrative", step.prompt, step.model, context,
            temperature=step.temperature, response_format=step.response_format,
            input_ref=f"pages:{sorted(page_numbers)}", input_summary=f"source={source_label}",
            context_variant=source_label,
            objective_id=objective_id, chain_run_id=chain_run_id, chain_sequence=seq,
        )
        summary.steps["narrative"] = [result]
        narrative_output = result.parsed_json

    # --- Stage 6: Slide Plan -----------------------------------------------
    slide_plan_output: Optional[dict] = None
    step = step_by_stage.get("slide_plan")
    if step and step.enabled:
        seq += 1
        source, source_label = _first_available(
            ("narrative", narrative_output),
            ("project_brief", brief_output),
            ("evidence", evidence_output),
        )
        context = {"CONTEXT": json.dumps(source) if source else "{}"}
        result = run_stage(
            "slide_plan", step.prompt, step.model, context,
            temperature=step.temperature, response_format=step.response_format,
            input_ref=f"pages:{sorted(page_numbers)}", input_summary=f"source={source_label}",
            context_variant=source_label,
            objective_id=objective_id, chain_run_id=chain_run_id, chain_sequence=seq,
        )
        summary.steps["slide_plan"] = [result]
        slide_plan_output = result.parsed_json

    # --- Stage 7: Slide Content ------------------------------------------
    step = step_by_stage.get("slide_content")
    if step and step.enabled and slide_plan_output:
        slides = slide_plan_output.get("slides", [])[:max_slide_content]
        results = []
        for idx, slide in enumerate(slides, start=1):
            seq += 1
            context = {
                "CONTEXT": json.dumps({"slide": slide, "narrative": narrative_output or {}}),
            }
            result = run_stage(
                "slide_content", step.prompt, step.model, context,
                temperature=step.temperature, response_format=step.response_format,
                input_ref=f"slide:{idx}", input_summary=slide.get("title", ""),
                context_variant="slide_plan_item",
                objective_id=objective_id, chain_run_id=chain_run_id, chain_sequence=seq,
            )
            results.append(result)
        summary.steps["slide_content"] = results

    return summary
