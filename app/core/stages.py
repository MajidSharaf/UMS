"""Registry of the 7 pipeline stages this lab replicates.

Each stage declares what kind of input it consumes so the UI/runner know
how to build context and whether a vision-capable model is required. See
reference/ums_pipeline_reference.md for how these map onto the real
production pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageDef:
    id: str
    name: str
    order: int
    input_kind: str  # image | page | page_analysis_list | evidence | project_brief | narrative | slide_plan_item
    vision_required: bool
    description: str


STAGES: list[StageDef] = [
    StageDef(
        id="image_analysis",
        name="Image Analysis",
        order=1,
        input_kind="image",
        vision_required=True,
        description="Describe visible evidence in one extracted image (production: brief-image, Qwen3-VL).",
    ),
    StageDef(
        id="page_analysis",
        name="Page Analysis",
        order=2,
        input_kind="page",
        vision_required=False,
        description="Reason over one page's text (+ its image analyses). No 1:1 production stage — experimental.",
    ),
    StageDef(
        id="evidence_consolidation",
        name="Evidence Consolidation",
        order=3,
        input_kind="page_analysis_list",
        vision_required=False,
        description="Merge many page analyses into facts/requirements/constraints/risks (production: deterministic Step 5, here made LLM-driven for experimentation).",
    ),
    StageDef(
        id="project_brief",
        name="Project Brief Generation",
        order=4,
        input_kind="evidence",
        vision_required=False,
        description="Consolidate evidence into an authoritative brief (production: brief-project, Gemma 4 26B).",
    ),
    StageDef(
        id="narrative",
        name="Narrative Generation",
        order=5,
        input_kind="project_brief",
        vision_required=False,
        description="Theme, key messages, storyline from the brief (production: blend of brief-summary + publish-normalize).",
    ),
    StageDef(
        id="slide_plan",
        name="Slide Planning",
        order=6,
        input_kind="narrative",
        vision_required=False,
        description="Exact slide count with topic/objective/key message per slide (production: publish-skeleton).",
    ),
    StageDef(
        id="slide_content",
        name="Slide Content Generation",
        order=7,
        input_kind="slide_plan_item",
        vision_required=False,
        description="Full copy for one planned slide (production: publish-copy).",
    ),
]

STAGES_BY_ID: dict[str, StageDef] = {s.id: s for s in STAGES}


def ordered_stage_ids() -> list[str]:
    return [s.id for s in sorted(STAGES, key=lambda s: s.order)]
