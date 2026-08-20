"""Tests for paragraph packing, structural chunking, and filename normalization."""

from app.core.config import Settings
from app.core.filenames import normalize_upload_filename
from app.services.chunking import chunk_text, chunk_text_structured


def _settings(**overrides) -> Settings:
    base = {
        "database_url": "postgresql+asyncpg://u:p@localhost/db",
        "chunk_size": 1500,
        "chunk_overlap": 200,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


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
    chunks = chunk_text(blocks, settings=_settings())

    assert len(chunks) == 1
    assert "Правила использования солнцезащитных средств" in chunks[0]
    assert "1. Наносите SPF" in chunks[0]
    assert "7. Храните средство" in chunks[0]


def test_chunk_text_packs_across_docx_paragraph_blocks() -> None:
    blocks = [f"Абзац номер {i}." for i in range(1, 21)]
    chunks = chunk_text(
        blocks,
        settings=_settings(chunk_size=200, chunk_overlap=40),
    )
    assert len(chunks) < len(blocks)
    assert all(len(chunk) <= 400 for chunk in chunks)
    joined = "\n".join(chunks)
    assert "Абзац номер 1." in joined
    assert "Абзац номер 20." in joined


def test_structured_chunking_keeps_section_with_numbered_list() -> None:
    blocks = [
        {"type": "heading", "content": "Уход за кожей летом", "level": 1},
        {
            "type": "heading",
            "content": "Правила использования солнцезащитных средств",
            "level": 2,
        },
        {"type": "list", "content": "1. Наносите SPF за 20 минут до выхода.", "level": 0},
        {"type": "list", "content": "2. Обновляйте каждые 2 часа.", "level": 0},
        {"type": "list", "content": "3. Не забывайте зону ушей и шеи.", "level": 0},
        {"type": "list", "content": "4. Используйте водостойкие формулы у воды.", "level": 0},
        {"type": "list", "content": "5. Наносите достаточный слой.", "level": 0},
        {"type": "list", "content": "6. Повторяйте после купания.", "level": 0},
        {"type": "list", "content": "7. Храните средство в прохладе.", "level": 0},
    ]
    chunks = chunk_text_structured(blocks, settings=_settings(chunk_size=200))

    sunscreen = [
        c
        for c in chunks
        if "Правила использования солнцезащитных средств" in c["content"]
    ]
    assert len(sunscreen) == 1
    content = sunscreen[0]["content"]
    assert "1. Наносите SPF" in content
    assert "7. Храните средство" in content
    assert sunscreen[0]["metadata"]["section_title"] == (
        "Правила использования солнцезащитных средств"
    )
    assert sunscreen[0]["metadata"]["is_heading_only"] is False


def test_post_merge_short_heading_with_numbered_list() -> None:
    blocks = [
        {"type": "heading", "content": "Короткий заголовок", "level": 1},
        {"type": "list", "content": "1. Первый пункт.", "level": 0},
        {"type": "list", "content": "2. Второй пункт.", "level": 0},
    ]
    # Tiny size would otherwise isolate the heading; post-merge must glue it.
    chunks = chunk_text_structured(blocks, settings=_settings(chunk_size=30))
    assert len(chunks) == 1
    assert chunks[0]["content"].startswith("Короткий заголовок")
    assert "1. Первый пункт." in chunks[0]["content"]
    assert "2. Второй пункт." in chunks[0]["content"]


def test_structured_multiple_sections() -> None:
    blocks = [
        {"type": "heading", "content": "Секция A", "level": 1},
        {"type": "text", "content": "Текст секции A.", "level": 0},
        {"type": "heading", "content": "Секция B", "level": 1},
        {"type": "text", "content": "Текст секции B.", "level": 0},
    ]
    chunks = chunk_text_structured(blocks, settings=_settings())
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["section_title"] == "Секция A"
    assert chunks[1]["metadata"]["section_title"] == "Секция B"
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[1]["metadata"]["chunk_index"] == 1


def test_plain_chunk_text_does_not_split_numbered_list() -> None:
    blocks = [
        "Правила",
        "1. Один.",
        "2. Два.",
        "3. Три.",
        "4. Четыре.",
        "5. Пять.",
    ]
    chunks = chunk_text(
        blocks,
        settings=_settings(chunk_size=40, chunk_overlap=5),
    )
    joined_list_chunks = [c for c in chunks if "1. Один." in c]
    assert joined_list_chunks
    assert "Правила" in joined_list_chunks[0]
    assert "5. Пять." in joined_list_chunks[0]


def test_normalize_upload_filename_repairs_mojibake() -> None:
    original = "Уход за кожей летом.docx"
    mojibake = original.encode("utf-8").decode("latin-1")
    assert normalize_upload_filename(mojibake) == original


def test_normalize_upload_filename_strips_path_and_fallback() -> None:
    assert normalize_upload_filename(r"C:\tmp\protocol.pdf") == "protocol.pdf"
    assert normalize_upload_filename(None) == "upload.bin"
    assert normalize_upload_filename("   ") == "upload.bin"
