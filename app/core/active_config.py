"""The pipeline runs with exactly one prompt + one model per stage at a
time — the "active" configuration. This is deliberately separate from the
experiment lab: the lab tries many variations and logs the results; a
human looks at those results and, when one wins, promotes it here. The
pipeline itself never exposes a picker — it just runs whatever is active.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import yaml

from . import config
from .prompts import Prompt, get_prompt, list_prompts

ACTIVE_CONFIG_PATH = config.REPO_ROOT / "pipeline_config.yaml"

# The clean pipeline only varies prompt and model, per stage. Temperature and
# decoding mode are fixed pipeline-wide so the run stays deterministic and
# uncluttered; those remain tunable inside the experiment lab.
PIPELINE_TEMPERATURE = 0.0
PIPELINE_RESPONSE_FORMAT = "json_schema"


@dataclass
class ActiveStageConfig:
    prompt_name: str
    model: str


def _load_all() -> dict[str, dict]:
    if not ACTIVE_CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _save_all(data: dict[str, dict]) -> None:
    ACTIVE_CONFIG_PATH.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def get_active(stage_id: str) -> ActiveStageConfig | None:
    data = _load_all().get(stage_id)
    if not data:
        return None
    return ActiveStageConfig(**data)


def get_active_prompt(stage_id: str) -> Prompt | None:
    active = get_active(stage_id)
    if not active:
        return None
    return get_prompt(stage_id, active.prompt_name)


def set_active(stage_id: str, prompt_name: str, model: str) -> None:
    data = _load_all()
    data[stage_id] = asdict(ActiveStageConfig(prompt_name, model))
    _save_all(data)


def ensure_default(stage_id: str, model: str) -> ActiveStageConfig:
    """If nothing is active yet for this stage, default to the seeded
    'production' prompt so the pipeline is runnable immediately."""
    existing = get_active(stage_id)
    if existing:
        return existing
    prod = next((p for p in list_prompts(stage_id) if p.name == "production"), None)
    name = prod.name if prod else "production"
    set_active(stage_id, name, model)
    return get_active(stage_id)
