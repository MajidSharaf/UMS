"""Objectives are just named notes describing what an experiment session is
trying to achieve — file-backed like prompts, so they're plain JSON you can
read/edit/diff outside the app too."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from . import config


@dataclass
class Objective:
    id: str
    title: str
    description: str = ""
    created_at: float = field(default_factory=time.time)


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "objective"


def list_objectives() -> list[Objective]:
    items = []
    for path in sorted(config.OBJECTIVES_DIR.glob("*.json")):
        items.append(Objective(**json.loads(path.read_text(encoding="utf-8"))))
    return sorted(items, key=lambda o: o.created_at, reverse=True)


def save_objective(title: str, description: str = "") -> Objective:
    obj_id = f"{_slug(title)}-{int(time.time())}"
    obj = Objective(id=obj_id, title=title, description=description)
    (config.OBJECTIVES_DIR / f"{obj_id}.json").write_text(json.dumps(asdict(obj), indent=2), encoding="utf-8")
    return obj


def get_objective(obj_id: str) -> Objective | None:
    path = config.OBJECTIVES_DIR / f"{obj_id}.json"
    if not path.exists():
        return None
    return Objective(**json.loads(path.read_text(encoding="utf-8")))
