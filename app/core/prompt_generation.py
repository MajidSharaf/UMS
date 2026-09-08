"""Asks a local Ollama text model to write new prompt variants for a
pipeline stage — the actual "AI that generates prompts" piece. Each call
targets one specific angle (terseness, anti-hallucination strictness,
chain-of-thought, etc.) so a batch of variants differs on a real axis
instead of being near-duplicate paraphrases of each other.
"""
from __future__ import annotations

import json
from pydantic import BaseModel, Field

from . import ollama_client, schemas
from .prompts import Prompt, get_prompt, list_prompt_names, save_prompt
from .stages import STAGES_BY_ID

# (angle_id, instruction) - each is a genuinely different prompt-design
# strategy, not just a rephrasing, so generated variants actually differ.
ANGLES: list[tuple[str, str]] = [
    ("terse", "Extremely terse and direct. Minimal instructions, no elaboration, "
              "the shortest system prompt that still reliably gets the job done."),
    ("strict-anti-hallucination", "Maximally strict about never inventing information. Repeatedly "
                                   "emphasize verifiability, explicit uncertainty marking, and leaving "
                                   "fields empty rather than guessing."),
    ("chain-of-thought", "Instruct the model to reason through the material step by step "
                          "internally first, then output ONLY the final JSON - no reasoning "
                          "text, no commentary, in the actual response."),
    ("checklist", "Frame the instructions as an explicit numbered checklist the model must "
                  "satisfy item by item before responding."),
    ("persona-authority", "Adopt a strong, specific domain-expert persona (e.g. a named role "
                           "with decades of relevant experience) and see whether that framing "
                           "changes the rigor of the output."),
    ("risk-first", "Prioritize surfacing risks, open questions, and missing/uncertain "
                   "information ahead of routine facts - a skeptical, caution-first read."),
    ("completeness-first", "Prioritize exhaustive completeness over brevity - capture every "
                            "relevant detail even if the output ends up longer."),
]


class GeneratedPromptItem(BaseModel):
    name: str = Field(description="short kebab-case name, unique, no spaces")
    system_prompt: str
    user_template: str = Field(description="must contain the literal token {{CONTEXT}}")


class GeneratedPromptBatch(BaseModel):
    prompts: list[GeneratedPromptItem]


def _build_meta_prompt(stage_id: str, angle_label: str, angle_instruction: str,
                        batch_size: int, existing_names: list[str]) -> tuple[str, str]:
    stage_def = STAGES_BY_ID[stage_id]
    baseline = get_prompt(stage_id, "production")
    schema_summary = json.dumps(schemas.json_schema_for(stage_id))
    needs_image_evidence = stage_id == "page_analysis"

    system = (
        "You are a prompt engineering assistant. You write system+user prompt pairs for one "
        "stage of a document-analysis LLM pipeline. Every user_template you write MUST contain "
        "the literal placeholder token {{CONTEXT}}"
        + (" and also the literal token {{IMAGE_EVIDENCE}}" if needs_image_evidence else "")
        + ". Return only JSON matching the required schema."
    )

    reference = (
        f"Reference prompt for this stage (an example that already works - draw on its ideas, "
        f"don't just copy it):\nSYSTEM: {baseline.system_prompt}\nUSER: {baseline.user_template}\n\n"
        if baseline else ""
    )

    user = (
        f"Stage: {stage_def.name} - {stage_def.description}\n\n"
        f"Target JSON output shape the prompt must instruct the model to return:\n{schema_summary}\n\n"
        f"{reference}"
        f"Names already used for this stage, do not reuse: {existing_names or '(none yet)'}\n\n"
        f"Write exactly {batch_size} NEW, DISTINCT prompt variants for this stage using this "
        f"specific style ({angle_label}):\n{angle_instruction}\n\n"
        f"Each variant needs a short unique kebab-case name, a system_prompt, and a user_template "
        f"containing the literal token {{{{CONTEXT}}}}"
        + (" and the literal token {{IMAGE_EVIDENCE}}" if needs_image_evidence else "") + "."
    )
    return system, user


def _valid_variant(stage_id: str, item: GeneratedPromptItem, existing_names: set[str]) -> str | None:
    """Returns an error string if invalid, else None."""
    if not item.name.strip():
        return "empty name"
    if item.name in existing_names:
        return f"name '{item.name}' already used"
    if "{{CONTEXT}}" not in item.user_template:
        return "user_template missing {{CONTEXT}}"
    if stage_id == "page_analysis" and "{{IMAGE_EVIDENCE}}" not in item.user_template:
        return "user_template missing {{IMAGE_EVIDENCE}}"
    if not item.system_prompt.strip():
        return "empty system_prompt"
    return None


def generate_batch(stage_id: str, model: str, angle_label: str, angle_instruction: str,
                    batch_size: int, existing_names: list[str]) -> tuple[list[Prompt], list[str]]:
    """One Ollama call requesting batch_size variants for one angle. Returns
    (saved_prompts, rejection_reasons) - invalid items are skipped, not
    saved, with a reason logged for each."""
    system, user = _build_meta_prompt(stage_id, angle_label, angle_instruction, batch_size, existing_names)
    gen = ollama_client.generate(
        model=model, system_prompt=system, user_prompt=user,
        temperature=0.7,  # some variety is the point here, unlike the pipeline's temp 0
        response_format=GeneratedPromptBatch.model_json_schema(),
        num_predict=4096,
    )
    try:
        parsed = json.loads(gen.output_text)
        batch = GeneratedPromptBatch.model_validate(parsed)
    except Exception as exc:
        return [], [f"batch response invalid: {exc}"]

    saved, errors = [], []
    seen = set(existing_names)
    for item in batch.prompts:
        error = _valid_variant(stage_id, item, seen)
        if error:
            errors.append(f"'{item.name}': {error}")
            continue
        prompt = save_prompt(
            stage_id, item.name, item.system_prompt, item.user_template,
            notes=f"AI-generated ({angle_label}) by {model}",
        )
        saved.append(prompt)
        seen.add(item.name)
    return saved, errors
