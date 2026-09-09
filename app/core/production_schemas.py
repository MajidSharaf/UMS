"""Real production JSON Schemas, transcribed from the UMS documentation
export (PIPELINE_PROMPTS / PROMPT_RESPONSE_SCHEMAS in pipelinedocs.js),
not the simplified shapes this lab used in the earlier round.

These get passed as Ollama's `format` field for genuine schema-constrained
decoding, not the loose response_format="json" mode used before - that
loose mode is the most likely reason production's own prompt returned
agentic tool-call JSON instead of a brief in the last sweep: with no
schema locking the shape down, the model drifted into whatever JSON
pattern it had seen most in training.
"""
from __future__ import annotations


def _confidence_schema() -> dict:
    return {
        "type": "object",
        "required": ["detection", "interpretation"],
        "properties": {
            "detection": {"type": "number", "minimum": 0, "maximum": 1},
            "interpretation": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": False,
    }


def _string_array(max_items: int, max_length: int) -> dict:
    return {"type": "array", "maxItems": max_items, "items": {"type": "string", "maxLength": max_length}}


def image_analysis_schema() -> dict:
    """brief-image, normal pass. 1024-token budget."""
    return {
        "type": "object",
        "required": ["summary", "subjects", "style", "environment", "colors", "visibleText", "uncertainties", "confidence"],
        "properties": {
            "summary": {"type": "string", "maxLength": 280},
            "subjects": _string_array(6, 80),
            "style": _string_array(4, 80),
            "environment": _string_array(4, 80),
            "colors": _string_array(5, 40),
            "visibleText": _string_array(5, 160),
            "uncertainties": _string_array(3, 160),
            "confidence": _confidence_schema(),
        },
        "additionalProperties": False,
    }


def compact_image_schema() -> dict:
    """brief-image, length-retry pass. 640-token budget."""
    return {
        "type": "object",
        "required": ["summary", "keywords", "visibleText", "confidence"],
        "properties": {
            "summary": {"type": "string", "maxLength": 240},
            "keywords": _string_array(8, 80),
            "visibleText": _string_array(3, 160),
            "confidence": _confidence_schema(),
        },
        "additionalProperties": False,
    }


# document/project/scope/site/programme/commercial are nested objects;
# executiveSummary/projectBackground/vision are prose strings; everything
# else is an array - exact typing rule from the real schema generator.
BRIEF_KEYS = [
    "document", "project", "executiveSummary", "projectBackground", "vision", "objectives",
    "successCriteria", "scope", "site", "programmeRequirements", "spatialRequirements",
    "functionalRequirements", "designRequirements", "technicalRequirements", "digitalRequirements",
    "bimRequirements", "informationRequirements", "sustainabilityRequirements", "accessibilityRequirements",
    "securityRequirements", "authorityRequirements", "approvals", "stakeholders", "users",
    "deliverables", "milestones", "programme", "commercial", "quantities", "constraints",
    "assumptions", "dependencies", "risks", "decisions", "openQuestions", "conflicts",
    "missingInformation", "recommendedClarifications", "sourceDocuments", "evidenceReferences",
]
_BRIEF_STRING_KEYS = {"executiveSummary", "projectBackground", "vision"}
_BRIEF_OBJECT_KEYS = {"document", "project", "scope", "site", "programme", "commercial"}


def project_brief_schema() -> dict:
    """brief-project. 4096-token budget. The real 39-key AECO shape."""
    def field_type(key: str) -> dict:
        if key in _BRIEF_STRING_KEYS:
            return {"type": "string"}
        if key in _BRIEF_OBJECT_KEYS:
            return {"type": "object"}
        return {"type": "array"}
    return {
        "type": "object",
        "required": BRIEF_KEYS,
        "properties": {key: field_type(key) for key in BRIEF_KEYS},
    }


_SUMMARY_STRING_KEYS = ["documentTitle", "documentType", "purpose", "executiveSummary"]
_SUMMARY_ARRAY_KEYS = ["mainTopics", "keyFindings", "importantFacts", "importantFigures",
                       "decisions", "risks", "openQuestions", "pageHighlights"]


def summary_schema() -> dict:
    """brief-summary. 1500-token budget."""
    props = {k: {"type": "string"} for k in _SUMMARY_STRING_KEYS}
    props.update({k: {"type": "array"} for k in _SUMMARY_ARRAY_KEYS})
    return {
        "type": "object",
        "required": _SUMMARY_STRING_KEYS + _SUMMARY_ARRAY_KEYS,
        "properties": props,
        "additionalProperties": False,
    }


def planned_slide_schema() -> dict:
    """One slide entry inside publish-normalize / publish-skeleton.
    purposeId's real enum comes from the selected layout catalog, which
    this lab doesn't have, so it's left as a free string rather than
    a fabricated enum."""
    return {
        "type": "object",
        "required": ["index", "title", "topic", "communicationObjective", "keyMessage", "purposeId", "rationale", "visualRole"],
        "properties": {
            "index": {"type": "integer", "minimum": 1},
            "title": {"type": "string", "maxLength": 80},
            "topic": {"type": "string", "maxLength": 120},
            "communicationObjective": {"type": "string", "maxLength": 180},
            "keyMessage": {"type": "string", "maxLength": 220},
            "purposeId": {"type": "string"},
            "rationale": {"type": "string", "maxLength": 220},
            "visualRole": {"type": "string", "enum": ["none", "image", "chart", "diagram"]},
        },
        "additionalProperties": False,
    }


def normalized_context_schema() -> dict:
    """publish-normalize."""
    return {
        "type": "object",
        "required": ["title", "objective", "audience", "tone", "language", "requestedSlideCount",
                     "mandatoryTopics", "constraints", "visualDirection", "presentationSummary", "slides"],
        "properties": {
            "title": {"type": "string", "maxLength": 100},
            "objective": {"type": "string", "maxLength": 300},
            "audience": {"type": "string", "maxLength": 160},
            "tone": {"type": "string", "maxLength": 100},
            "language": {"type": "string", "maxLength": 60},
            "requestedSlideCount": {"type": "integer", "minimum": 3, "maximum": 40},
            "mandatoryTopics": _string_array(40, 100),
            "constraints": _string_array(20, 160),
            "visualDirection": {"type": "string", "maxLength": 240},
            "presentationSummary": {"type": "string", "maxLength": 400},
            "slides": {"type": "array", "minItems": 3, "maxItems": 40, "items": planned_slide_schema()},
        },
        "additionalProperties": False,
    }


def slide_plan_schema(requested_slide_count: int | None = None) -> dict:
    """publish-skeleton. 2048-token budget."""
    slides_schema = {"type": "array", "items": planned_slide_schema()}
    if requested_slide_count:
        slides_schema["minItems"] = requested_slide_count
        slides_schema["maxItems"] = requested_slide_count
    return {
        "type": "object",
        "required": ["presentationSummary", "slides"],
        "properties": {
            "presentationSummary": {"type": "string", "maxLength": 400},
            "slides": slides_schema,
        },
        "additionalProperties": False,
    }


def slide_enrichment_schema() -> dict:
    """publish-enrich. 768-token/slide budget."""
    return {
        "type": "object",
        "required": ["contentBrief", "requiredContent", "visualBrief", "researchNeeds"],
        "properties": {
            "contentBrief": {"type": "string", "maxLength": 360},
            "requiredContent": {"type": "array", "minItems": 1, "maxItems": 6, "items": {"type": "string", "maxLength": 160}},
            "visualBrief": {"type": "string", "maxLength": 240},
            "researchNeeds": _string_array(5, 160),
        },
        "additionalProperties": False,
    }


def slide_content_schema() -> dict:
    """publish-copy. Real production generates this from the selected
    layout's content contract, which this lab doesn't have (no real
    layout catalog) - kept as a plain, generic slide-copy shape rather
    than fabricating a fake layout contract."""
    return {
        "type": "object",
        "required": ["title", "body"],
        "properties": {
            "title": {"type": "string", "maxLength": 80},
            "body": {"type": "array", "items": {"type": "string"}},
            "speakerNotes": {"type": "string"},
        },
        "additionalProperties": False,
    }
