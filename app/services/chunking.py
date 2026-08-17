"""Split extracted text into overlapping chunks for embeddings."""

from __future__ import annotations

from loguru import logger

from app.core.config import Settings, get_settings


def chunk_text(
    blocks: list[str] | str,
    *,
    settings: Settings | None = None,
) -> list[str]:
    cfg = settings or get_settings()
    size = max(1, cfg.chunk_size)
    overlap = max(0, min(cfg.chunk_overlap, size // 2))

    if isinstance(blocks, str):
        source_blocks = [blocks]
    else:
        source_blocks = blocks

    normalized = [block.strip() for block in source_blocks if block and block.strip()]
    if not normalized:
        return []

    chunks: list[str] = []
    for block in normalized:
        chunks.extend(_chunk_block(block, size=size, overlap=overlap))
    logger.debug(
        "Chunking done | input_blocks={} chunks={} size={} overlap={}",
        len(normalized),
        len(chunks),
        size,
        overlap,
    )
    return chunks


def _chunk_block(text: str, *, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return _chunk_by_chars(text, size=size, overlap=overlap)

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = _overlap_tail(current, overlap)
            candidate = (
                f"{current}\n{paragraph}".strip() if current else paragraph
            )
        if len(candidate) <= size:
            current = candidate
            continue
        # Paragraph itself is larger than size — hard-split it.
        if current:
            chunks.append(current)
            current = ""
        parts = _chunk_by_chars(paragraph, size=size, overlap=overlap)
        chunks.extend(parts[:-1])
        current = parts[-1] if parts else ""

    if current:
        chunks.append(current)
    return [_hard_limit(chunk, size) for chunk in chunks if chunk.strip()]


def _hard_limit(text: str, size: int) -> str:
    if len(text) <= size:
        return text
    return text[:size].rstrip()


def _chunk_by_chars(text: str, *, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            # Prefer sentence / space boundary near the cut.
            window = text[start:end]
            split_at = max(
                window.rfind(". "),
                window.rfind(" "),
                window.rfind("\n"),
            )
            if split_at > size // 3:
                end = start + split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk[:size])
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text
    return text[-overlap:].lstrip()
