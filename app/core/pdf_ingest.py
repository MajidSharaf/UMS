"""Lightweight PDF ingestion: renders pages, crops meaningful images, and
detects headings by font size — a PyMuPDF-only substitute for Docling's
static-extraction step (production Step 2). No layout ML model, no OCR,
no multi-GB install — works well for native-text PDFs (not scanned
documents), which covers this document and most modern briefs/reports.

Produces the same directory shape and extraction.json fields that
data_sources.py already reads, so nothing downstream — prompts, schemas,
runner, chain, the Pipeline/Experiment pages — needs to change.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import pymupdf as fitz

IMAGE_MIN_AREA_FRAC = 0.015  # excludes tiny icons/decoration
IMAGE_MAX_AREA_FRAC = 0.85   # excludes full-page background/template graphics
RENDER_ZOOM = 2.0
HEADING_SIZE_RATIO = 1.15    # a text block is a heading if its font is >=15% larger than the page's body size


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "document"


def _classify_text_blocks(text_dict: dict, page_w: float, page_h: float, page_num: int) -> tuple[list[dict], list[dict]]:
    sizes = [
        round(span["size"], 1)
        for block in text_dict.get("blocks", []) if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    ]
    body_size = statistics.mode(sizes) if sizes else 10.0

    regions, text_blocks = [], []
    region_idx = 0
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_text = " ".join(
            span.get("text", "") for line in block.get("lines", []) for span in line.get("spans", [])
        ).strip()
        if not block_text:
            continue
        max_size = max(
            (span["size"] for line in block.get("lines", []) for span in line.get("spans", [])),
            default=body_size,
        )
        region_idx += 1
        region_id = f"p{page_num:03d}-r{region_idx:03d}"
        bx0, by0, bx1, by1 = block["bbox"]
        bbox = {
            "x": bx0 / page_w, "y": by0 / page_h,
            "width": (bx1 - bx0) / page_w, "height": (by1 - by0) / page_h,
        }
        region_type = "heading" if max_size >= body_size * HEADING_SIZE_RATIO else "prose"
        regions.append({"regionId": region_id, "type": region_type, "bbox": bbox})
        text_blocks.append({"regionId": region_id, "text": block_text, "bbox": bbox})
    return regions, text_blocks


def _extract_meaningful_images(page: "fitz.Page", page_num: int, images_dir: Path) -> int:
    page_w, page_h = page.rect.width, page.rect.height
    image_count = 0
    seen_xrefs = set()
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        for rect in page.get_image_rects(xref):
            area_frac = (rect.width * rect.height) / (page_w * page_h)
            if not (IMAGE_MIN_AREA_FRAC <= area_frac <= IMAGE_MAX_AREA_FRAC):
                continue
            image_count += 1
            clip_pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM), clip=rect)
            filename = f"img-p{page_num:03d}-r{image_count:03d}-01.png"
            clip_pix.save(images_dir / filename)
    return image_count


def ingest_pdf(pdf_path: Path, documents_dir: Path, document_id: str | None = None) -> str:
    """Extracts pdf_path into documents_dir/<document_id>/{pages,images,json}.
    Returns the document_id used (derived from the filename if not given)."""
    pdf_path = Path(pdf_path)
    document_id = document_id or slugify(pdf_path.stem)
    doc_dir = documents_dir / document_id
    pages_dir, images_dir, json_dir = doc_dir / "pages", doc_dir / "images", doc_dir / "json"
    for d in (pages_dir, images_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages_meta = []
    for pno in range(doc.page_count):
        page = doc[pno]
        page_num = pno + 1
        page_w, page_h = page.rect.width, page.rect.height

        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM))
        pix.save(pages_dir / f"page-{page_num:04d}.png")

        regions, text_blocks = _classify_text_blocks(page.get_text("dict"), page_w, page_h, page_num)
        image_count = _extract_meaningful_images(page, page_num, images_dir)

        pages_meta.append({
            "pageNumber": page_num, "width": page_w, "height": page_h,
            "regions": regions, "textBlocks": text_blocks,
            "imageCount": image_count, "tableCount": 0, "warnings": [],
        })

    extraction = {
        "pages": pages_meta,
        "tables": [],
        "smallImages": [],
        "activityLog": [{"action": "pymupdf-ingest", "detail": pdf_path.name, "status": "complete"}],
        "raw": {"document": {
            "filename": pdf_path.name, "pageCount": doc.page_count,
            "selectedPages": list(range(1, doc.page_count + 1)),
        }},
    }
    (json_dir / "extraction.json").write_text(json.dumps(extraction, indent=2), encoding="utf-8")
    doc.close()
    return document_id
