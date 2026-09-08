"""File-backed prompt library: every prompt is one plain-text file under
prompts_library/<stage>/<name>.txt. No YAML, no version numbers — a name
is a name; editing a prompt overwrites its file, and re-saving under a
new name keeps the old one around unchanged. History, if you want it,
comes from git (or just don't overwrite — save under a new name).

File format:
    NAME: <name>
    PRODUCTION: true|false
    NOTES: <optional one-line note>
    ---SYSTEM---
    <system prompt text, can span multiple lines>
    ---USER---
    <user template text, can span multiple lines, uses {{CONTEXT}} etc.>

Placeholders in user_template use {{DOUBLE_BRACES}} rather than Python
str.format so JSON braces inside the template never collide with
substitution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config

PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")
_SYSTEM_MARKER = "---SYSTEM---"
_USER_MARKER = "---USER---"


@dataclass
class Prompt:
    stage: str
    name: str
    system_prompt: str
    user_template: str
    notes: str = ""
    is_production_baseline: bool = False

    @property
    def display_name(self) -> str:
        return f"{self.name} [production]" if self.is_production_baseline else self.name

    @property
    def file_path(self) -> Path:
        return _stage_dir(self.stage) / f"{_slug(self.name)}.txt"

    def placeholders(self) -> list[str]:
        return sorted(set(PLACEHOLDER_RE.findall(self.user_template)))

    def render_user_prompt(self, **context: str) -> str:
        text = self.user_template
        for key, value in context.items():
            text = text.replace(f"{{{{{key}}}}}", str(value))
        return text


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "prompt"


def _stage_dir(stage: str) -> Path:
    path = config.PROMPTS_LIBRARY_DIR / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def _serialize(p: Prompt) -> str:
    header = [f"NAME: {p.name}", f"PRODUCTION: {'true' if p.is_production_baseline else 'false'}"]
    if p.notes:
        header.append(f"NOTES: {p.notes}")
    return "\n".join(header) + f"\n{_SYSTEM_MARKER}\n{p.system_prompt}\n{_USER_MARKER}\n{p.user_template}\n"


def _parse(stage: str, text: str) -> Prompt:
    sys_idx = text.index(_SYSTEM_MARKER)
    usr_idx = text.index(_USER_MARKER)
    header, system_prompt, user_template = (
        text[:sys_idx], text[sys_idx + len(_SYSTEM_MARKER):usr_idx].strip("\n"),
        text[usr_idx + len(_USER_MARKER):].strip("\n"),
    )
    name, is_production, notes = "", False, ""
    for line in header.splitlines():
        if line.startswith("NAME:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("PRODUCTION:"):
            is_production = line.split(":", 1)[1].strip().lower() == "true"
        elif line.startswith("NOTES:"):
            notes = line.split(":", 1)[1].strip()
    return Prompt(stage=stage, name=name, system_prompt=system_prompt, user_template=user_template,
                  notes=notes, is_production_baseline=is_production)


def list_prompts(stage: str) -> list[Prompt]:
    prompts = []
    for path in sorted(_stage_dir(stage).glob("*.txt")):
        try:
            prompts.append(_parse(stage, path.read_text(encoding="utf-8")))
        except ValueError:
            continue  # malformed file (missing markers) - skip rather than crash the app
    return sorted(prompts, key=lambda p: p.name)


def list_prompt_names(stage: str) -> list[str]:
    return sorted({p.name for p in list_prompts(stage)})


def get_prompt(stage: str, name: str) -> Optional[Prompt]:
    for p in list_prompts(stage):
        if p.name == name:
            return p
    return None


def save_prompt(
    stage: str,
    name: str,
    system_prompt: str,
    user_template: str,
    notes: str = "",
    is_production_baseline: bool = False,
) -> Prompt:
    """Writes (or overwrites) the file for this (stage, name)."""
    prompt = Prompt(stage=stage, name=name, system_prompt=system_prompt, user_template=user_template,
                     notes=notes, is_production_baseline=is_production_baseline)
    prompt.file_path.write_text(_serialize(prompt), encoding="utf-8")
    return prompt


def seed_if_missing(stage: str, name: str, system_prompt: str, user_template: str,
                     notes: str = "", is_production_baseline: bool = False) -> Prompt:
    existing = get_prompt(stage, name)
    if existing:
        return existing
    return save_prompt(stage, name, system_prompt, user_template, notes, is_production_baseline)
