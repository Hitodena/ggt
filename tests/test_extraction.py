"""Tests for text extraction and chunking (no OCR/LibreOffice)."""

from io import BytesIO

import pymupdf
import pytest
from docx import Document

from app.core.config import Settings
from app.services.chunking import chunk_text
from app.services.extraction import ExtractionError, TextExtractor


def _settings(**overrides) -> Settings:
    base = {
        "database_url": "postgresql+asyncpg://u:p@localhost/db",
        "ocr_languages": "eng",
        "pdf_ocr_min_chars": 20,
        "chunk_size": 80,
        "chunk_overlap": 20,
        "libreoffice_path": "soffice",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_extract_docx() -> None:
    document = Document()
    document.add_paragraph("Первый абзац про уход за кожей.")
    document.add_paragraph("Второй абзац с рекомендациями.")
    buffer = BytesIO()
    document.save(buffer)

    blocks = TextExtractor(_settings()).extract("note.docx", buffer.getvalue())
    text = "\n".join(blocks)
    assert "Первый абзац" in text
    assert "Второй абзац" in text


def test_extract_pdf_text_layer() -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    # Default PDF fonts are Latin-only; keep ASCII for reliable text-layer tests.
    page.insert_text((72, 72), "PDF text layer about peeling protocol")
    data = pdf.tobytes()
    pdf.close()

    blocks = TextExtractor(_settings()).extract("guide.pdf", data)
    assert any("peeling" in block.lower() for block in blocks)


def test_extract_unsupported_extension() -> None:
    with pytest.raises(ExtractionError, match="Unsupported"):
        TextExtractor(_settings()).extract("notes.txt", b"hello")


def test_chunk_text_respects_size_and_overlap() -> None:
    settings = _settings(chunk_size=50, chunk_overlap=10)
    long_text = " ".join([f"word{i}" for i in range(40)])
    chunks = chunk_text([long_text], settings=settings)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_chunk_text_keeps_short_block() -> None:
    chunks = chunk_text(["короткий блок"], settings=_settings())
    assert chunks == ["короткий блок"]
