"""Central paths and settings for the experimentation lab.

Everything is resolved relative to the repo root so the app works the same
whether launched with `streamlit run app/Home.py` from the repo root or from
elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
PAGES_DIR = DATA_DIR / "pages"
IMAGES_DIR = DATA_DIR / "images"
EXTRACTION_JSON = DATA_DIR / "json" / "extraction.json"

# Additional documents ingested later (see pdf_ingest.py) live under here,
# one subfolder each, so a new PDF never overwrites the original document
# above. document_id=None everywhere in data_sources.py means "the
# original document" (the paths above), preserving existing behavior.
DOCUMENTS_DIR = DATA_DIR / "documents"


def resolve_document_paths(document_id: str | None = None) -> dict:
    if document_id is None:
        return {"pages_dir": PAGES_DIR, "images_dir": IMAGES_DIR, "extraction_json": EXTRACTION_JSON}
    doc_dir = DOCUMENTS_DIR / document_id
    return {
        "pages_dir": doc_dir / "pages",
        "images_dir": doc_dir / "images",
        "extraction_json": doc_dir / "json" / "extraction.json",
    }


def list_document_ids() -> list[str]:
    """Additional documents only — the original document has no id (None)."""
    if not DOCUMENTS_DIR.exists():
        return []
    return sorted(p.name for p in DOCUMENTS_DIR.iterdir() if p.is_dir())

PROMPTS_LIBRARY_DIR = REPO_ROOT / "prompts_library"
OBJECTIVES_DIR = REPO_ROOT / "objectives"

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
DB_PATH = EXPERIMENTS_DIR / "lab.db"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Heuristic used only as a fallback when /api/show capabilities are
# unavailable (older Ollama). Real capability check lives in ollama_client.
VISION_NAME_HINTS = ("vl", "vision", "llava", "moondream", "bakllava", "gemma3")

for _dir in (RUNS_DIR, OBJECTIVES_DIR, PROMPTS_LIBRARY_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
