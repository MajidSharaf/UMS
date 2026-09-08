"""Loads an extracted document (pages, images, extraction.json) and builds
stage inputs — including several interchangeable "context packaging"
variants so context-structure is itself an experiment axis, not a fixed
choice.

Every function takes an optional document_id. None (the default
everywhere) means the original document this lab shipped with, at
data/pages, data/images, data/json/extraction.json — so every existing
call site and every already-logged run keeps working unchanged. A real
document_id points at data/documents/<id>/ (see pdf_ingest.py), letting
you ingest and work with additional documents without touching the
original.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config

IMAGE_NAME_RE = re.compile(r"img-p(\d{3})-r(\d{3})-\d+\.\w+$")
PAGE_NAME_RE = re.compile(r"page-(\d{4})\.png$")


def _doc_suffix(document_id: Optional[str]) -> str:
    return "" if document_id is None else f"@{document_id}"


def split_ref(ref: str) -> tuple[str, Optional[str]]:
    """Reverses the @document_id suffix ref/get_page/get_image add, e.g.
    'page:5@my-doc' -> ('page:5', 'my-doc'); 'page:5' -> ('page:5', None)."""
    if "@" in ref:
        base, doc_id = ref.rsplit("@", 1)
        return base, doc_id
    return ref, None


@functools.lru_cache(maxsize=None)
def load_extraction(document_id: Optional[str] = None) -> dict:
    path = config.resolve_document_paths(document_id)["extraction_json"]
    if not path.exists():
        return {"pages": [], "tables": [], "smallImages": [], "raw": {}}
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _pages_by_number(document_id: Optional[str] = None) -> dict[int, dict]:
    return {p["pageNumber"]: p for p in load_extraction(document_id).get("pages", [])}


@functools.lru_cache(maxsize=None)
def document_filename(document_id: Optional[str] = None) -> str:
    return load_extraction(document_id).get("raw", {}).get("document", {}).get("filename", "document.pdf")


@dataclass
class PageRef:
    page_number: int
    image_path: Path
    heading: str = ""
    prose_text: str = ""
    warnings: list[str] = field(default_factory=list)
    image_count: int = 0
    table_count: int = 0
    document_id: Optional[str] = None

    @property
    def ref(self) -> str:
        return f"page:{self.page_number}{_doc_suffix(self.document_id)}"


@dataclass
class ImageRef:
    filename: str
    image_path: Path
    page_number: int
    region_id: str
    nearest_heading: str = ""
    nearby_text: str = ""
    document_id: Optional[str] = None

    @property
    def ref(self) -> str:
        return f"image:{self.filename}{_doc_suffix(self.document_id)}"


def _text_blocks_text(page: dict) -> tuple[str, str]:
    """Returns (heading_text, prose_text) joined from the page's text blocks."""
    headings, prose = [], []
    region_types = {r["regionId"]: r.get("type") for r in page.get("regions", [])}
    for block in page.get("textBlocks", []):
        text = (block.get("text") or "").strip()
        if not text:
            continue
        if region_types.get(block.get("regionId")) == "heading":
            headings.append(text)
        else:
            prose.append(text)
    return " | ".join(headings), "\n".join(prose)


@functools.lru_cache(maxsize=None)
def list_pages(document_id: Optional[str] = None) -> tuple[PageRef, ...]:
    pages_dir = config.resolve_document_paths(document_id)["pages_dir"]
    by_number = _pages_by_number(document_id)
    refs = []
    if pages_dir.exists():
        for path in sorted(pages_dir.glob("page-*.png")):
            match = PAGE_NAME_RE.search(path.name)
            if not match:
                continue
            page_number = int(match.group(1))
            page = by_number.get(page_number, {})
            heading, prose = _text_blocks_text(page)
            refs.append(
                PageRef(
                    page_number=page_number, image_path=path, heading=heading, prose_text=prose,
                    warnings=page.get("warnings", []), image_count=page.get("imageCount", 0),
                    table_count=page.get("tableCount", 0), document_id=document_id,
                )
            )
    return tuple(refs)


def get_page(page_number: int, document_id: Optional[str] = None) -> Optional[PageRef]:
    for p in list_pages(document_id):
        if p.page_number == page_number:
            return p
    return None


@functools.lru_cache(maxsize=None)
def list_images(document_id: Optional[str] = None) -> tuple[ImageRef, ...]:
    images_dir = config.resolve_document_paths(document_id)["images_dir"]
    by_number = _pages_by_number(document_id)
    refs = []
    if images_dir.exists():
        for path in sorted(images_dir.glob("img-*")):
            match = IMAGE_NAME_RE.search(path.name)
            if not match:
                continue
            page_number = int(match.group(1))
            region_id = f"p{match.group(1)}-r{match.group(2)}"
            page = by_number.get(page_number, {})
            heading, prose = _text_blocks_text(page)
            # extraction.json does not carry per-image bbox/proximity for
            # "meaningful" images (only for smallImages) — approximate
            # nearestHeading/nearbyText with the page's heading + prose, which
            # is the same evidence the production brief-image prompt uses.
            refs.append(
                ImageRef(
                    filename=path.name, image_path=path, page_number=page_number, region_id=region_id,
                    nearest_heading=heading, nearby_text=prose[:800], document_id=document_id,
                )
            )
    return tuple(refs)


def get_image(filename: str, document_id: Optional[str] = None) -> Optional[ImageRef]:
    for i in list_images(document_id):
        if i.filename == filename:
            return i
    return None


def images_on_page(page_number: int, document_id: Optional[str] = None) -> list[ImageRef]:
    return [i for i in list_images(document_id) if i.page_number == page_number]


def resolve_ref(ref: str):
    """Given any PageRef/ImageRef.ref string (with or without an
    @document_id suffix), returns the matching PageRef or ImageRef, or
    None. Used by UI code that only has the stored ref string (e.g. from
    the run log) and needs the source material back."""
    base, document_id = split_ref(ref)
    if base.startswith("page:"):
        return get_page(int(base.split(":", 1)[1]), document_id)
    if base.startswith("image:"):
        return get_image(base.split(":", 1)[1], document_id)
    return None


# ---------------------------------------------------------------------------
# Context packaging variants — same underlying data, different shape/amount
# sent to the model. Pick one per run to test "context structure" as an
# experiment axis.
# ---------------------------------------------------------------------------

def page_context_full(page: PageRef) -> str:
    """Heading + full prose text + counts. Closest to production Step 5."""
    lines = [f"Document: {document_filename(page.document_id)}", f"Page: {page.page_number}"]
    if page.heading:
        lines.append(f"Heading: {page.heading}")
    lines.append(f"Images on page: {page.image_count}, Tables on page: {page.table_count}")
    if page.prose_text:
        lines.append("Text:\n" + page.prose_text)
    return "\n".join(lines)


def page_context_headings_only(page: PageRef) -> str:
    """Minimal context: just headings, no body text. Tests how much the
    model can infer from structure alone vs needing full text."""
    lines = [f"Document: {document_filename(page.document_id)}", f"Page: {page.page_number}"]
    if page.heading:
        lines.append(f"Heading: {page.heading}")
    return "\n".join(lines)


def page_context_with_neighbors(page: PageRef, window: int = 1) -> str:
    """Full context plus prose from adjacent pages, to test whether extra
    surrounding context reduces missed cross-page requirements."""
    pages = {p.page_number: p for p in list_pages(page.document_id)}
    lines = [page_context_full(page)]
    for offset in range(1, window + 1):
        for neighbor_num in (page.page_number - offset, page.page_number + offset):
            neighbor = pages.get(neighbor_num)
            if neighbor and neighbor.prose_text:
                lines.append(f"\n[Context from page {neighbor_num}]\n{neighbor.prose_text[:500]}")
    return "\n".join(lines)


CONTEXT_VARIANTS = {
    "full": page_context_full,
    "headings_only": page_context_headings_only,
    "with_neighbors": page_context_with_neighbors,
}


def image_prompt_context(image: ImageRef) -> str:
    """Matches the production brief-image user prompt fields exactly:
    filename, page number, nearest heading, nearby text."""
    return (
        f"Document: {document_filename(image.document_id)}\n"
        f"Page: {image.page_number}\n"
        f"Heading: {image.nearest_heading}\n"
        f"Nearby text: {image.nearby_text}"
    )
