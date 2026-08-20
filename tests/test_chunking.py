"""Tests for paragraph packing and upload filename normalization."""

from app.core.config import Settings
from app.core.filenames import normalize_upload_filename
from app.services.chunking import chunk_text


def test_chunk_text_keeps_heading_with_following_list() -> None:
    blocks = [
        "Уход за кожей летом",
        "Правила использования солнцезащитных средств",
        "1. Наносите SPF за 20 минут до выхода.",
        "2. Обновляйте каждые 2 часа.",
        "3. Не забывайте зону ушей и шеи.",
        "4. Используйте водостойкие формулы у воды.",
        "5. Наносите достаточный слой.",
        "6. Повторяйте после купания.",
        "7. Храните средство в прохладе.",
    ]
    settings = Settings(
        database_url="postgresql+asyncpg://u:p@localhost/db",
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks = chunk_text(blocks, settings=settings)

    assert len(chunks) == 1
    assert "Правила использования солнцезащитных средств" in chunks[0]
    assert "1. Наносите SPF" in chunks[0]
    assert "7. Храните средство" in chunks[0]


def test_chunk_text_packs_across_docx_paragraph_blocks() -> None:
    # Old behaviour: one short block => one chunk. Now blocks are packed.
    blocks = [f"Абзац номер {i}." for i in range(1, 21)]
    settings = Settings(
        database_url="postgresql+asyncpg://u:p@localhost/db",
        chunk_size=200,
        chunk_overlap=40,
    )
    chunks = chunk_text(blocks, settings=settings)
    assert len(chunks) < len(blocks)
    assert all(len(chunk) <= 200 for chunk in chunks)
    joined = "\n".join(chunks)
    assert "Абзац номер 1." in joined
    assert "Абзац номер 20." in joined


def test_normalize_upload_filename_repairs_mojibake() -> None:
    original = "Уход за кожей летом.docx"
    mojibake = original.encode("utf-8").decode("latin-1")
    assert normalize_upload_filename(mojibake) == original


def test_normalize_upload_filename_strips_path_and_fallback() -> None:
    assert normalize_upload_filename(r"C:\tmp\protocol.pdf") == "protocol.pdf"
    assert normalize_upload_filename(None) == "upload.bin"
    assert normalize_upload_filename("   ") == "upload.bin"
