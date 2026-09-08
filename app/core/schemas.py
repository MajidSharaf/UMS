"""Pydantic output contracts for each pipeline stage.

Simplified from the full production shapes documented in
reference/ums_pipeline_reference.md so that small local models
(llama3.1:8b and similar) have a realistic chance of producing valid
structured output. Grow these toward the production shape as bigger
models get added to Ollama.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ImageAnalysisOutput(BaseModel):
    summary: str = ""
    observations: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class PageAnalysisOutput(BaseModel):
    summary: str = ""
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class EvidenceConsolidationOutput(BaseModel):
    facts: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class ProjectBriefOutput(BaseModel):
    project_summary: str = ""
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class NarrativeOutput(BaseModel):
    theme: str = ""
    key_messages: list[str] = Field(default_factory=list)
    storyline: list[str] = Field(default_factory=list)


class SlideSpec(BaseModel):
    title: str = ""
    objective: str = ""
    key_message: str = ""
    content_points: list[str] = Field(default_factory=list)


class SlidePlanOutput(BaseModel):
    slide_count: int = 0
    slides: list[SlideSpec] = Field(default_factory=list)


class SlideContentOutput(BaseModel):
    title: str = ""
    body: list[str] = Field(default_factory=list)
    speaker_notes: Optional[str] = None


STAGE_SCHEMAS: dict[str, type[BaseModel]] = {
    "image_analysis": ImageAnalysisOutput,
    "page_analysis": PageAnalysisOutput,
    "evidence_consolidation": EvidenceConsolidationOutput,
    "project_brief": ProjectBriefOutput,
    "narrative": NarrativeOutput,
    "slide_plan": SlidePlanOutput,
    "slide_content": SlideContentOutput,
}


def validate_stage_output(stage: str, data: dict) -> tuple[bool, list[str]]:
    """Returns (is_valid, error_messages). Never raises."""
    model_cls = STAGE_SCHEMAS.get(stage)
    if model_cls is None:
        return False, [f"Unknown stage '{stage}'"]
    try:
        model_cls.model_validate(data)
        return True, []
    except Exception as exc:  # pydantic ValidationError, but keep it broad/safe
        return False, [str(exc)]


def json_schema_for(stage: str) -> dict:
    """Full JSON Schema for Ollama's structured-output `format` field."""
    model_cls = STAGE_SCHEMAS[stage]
    return model_cls.model_json_schema()
