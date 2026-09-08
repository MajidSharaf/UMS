"""Extracts a new PDF into data/documents/<id>/ using the PyMuPDF-based
extractor (pdf_ingest.py) — a lightweight substitute for Docling's static
extraction step. Renders pages, crops meaningful images, classifies
heading vs. body text by font size. The original document this lab
shipped with is never touched by this."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import config, data_sources as ds
from core import pdf_ingest

st.set_page_config(page_title="Ingest Document", layout="wide")
st.title("Ingest Document")
st.caption(
    "Extract a new PDF into pages + meaningful images + heading/body text, the same "
    "shape the original document uses, so it works with every stage unchanged. Uses "
    "PyMuPDF only (no Docling, no multi-GB install) — works well for native-text PDFs; "
    "scanned/image-only PDFs won't get usable text this way."
)

existing = config.list_document_ids()
if existing:
    st.markdown("**Already ingested:** " + ", ".join(f"`{d}`" for d in existing))

uploaded = st.file_uploader("PDF to ingest", type=["pdf"])
document_id = st.text_input(
    "Document id (used as the folder name under data/documents/ — leave blank to derive from the filename)",
    value="",
)

if uploaded and st.button("Ingest", type="primary"):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)
    with st.spinner("Rendering pages, cropping images, classifying text…"):
        doc_id = pdf_ingest.ingest_pdf(tmp_path, config.DOCUMENTS_DIR, document_id.strip() or None)
    tmp_path.unlink(missing_ok=True)
    ds.list_pages.cache_clear()
    ds.list_images.cache_clear()
    ds.load_extraction.cache_clear()
    ds._pages_by_number.cache_clear()
    ds.document_filename.cache_clear()
    st.success(f"Ingested as `{doc_id}`.")
    pages = ds.list_pages(doc_id)
    images = ds.list_images(doc_id)
    st.metric("Pages", len(pages))
    st.metric("Meaningful images", len(images))
    if images:
        st.markdown("**Sample extracted images:**")
        cols = st.columns(6)
        for i, img in enumerate(images[:6]):
            with cols[i]:
                st.image(str(img.image_path), caption=f"p{img.page_number}", use_container_width=True)
    st.info(
        "This document isn't wired into the Pipeline/Experiment pages' document pickers yet — "
        "say the word if you want to actually run stages against it next."
    )
