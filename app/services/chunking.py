"""Split extracted text into overlapping / structural chunks for embeddings."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.core.config import Settings, get_settings

_NUMBERED_ITEM_RE = re.compile(r"^\d+[.)]\s+")
_HEADING_ONLY_MAX_CHARS = 120


def chunk_text(
    blocks: list[str] | list[dict[str, Any]] | str,
    *,
    settings: Settings | None = None,
) -> list[str]:
    """Pack extracted blocks into chunks.

    Structured DOCX blocks (dicts with ``type``/``content``) use section-aware
    chunking. Plain strings use paragraph packing with overlap.
    """
    cfg = settings or get_settings()
    if _is_structured(blocks):
        structured = chunk_text_structured(blocks, settings=cfg)  # type: ignore[arg-type]
        return [item["content"] for item in structured]

    size = max(1, cfg.chunk_size)
    overlap = max(0, min(cfg.chunk_overlap, size // 2))

    if isinstance(blocks, str):
        source_blocks = [blocks]
    else:
        source_blocks = [b for b in blocks if isinstance(b, str)]  # type: ignore[misc]

    paragraphs = _flatten_paragraphs(source_blocks)
    if not paragraphs:
        return []

    chunks = _pack_paragraphs(paragraphs, size=size, overlap=overlap)
    logger.debug(
        "Chunking done | input_blocks={} paragraphs={} chunks={} "
        "size={} overlap={}",
        len([b for b in source_blocks if b and b.strip()]),
        len(paragraphs),
        len(chunks),
        size,
        overlap,
    )
    return chunks


def chunk_text_structured(
    blocks: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Section-aware chunking for structured DOCX blocks.

    Rules:
    - Heading + body until next equal/higher heading = one section chunk
    - Never split between a heading and a following numbered list
    - Never split a consecutive numbered list mid-sequence
    - Post-merge: short heading-only chunk (<~120 chars) glued to next if
      that next chunk starts with ``^\\d+[.)]``
    """
    cfg = settings or get_settings()
    size = max(1, cfg.chunk_size)

    normalized = [_normalize_block(b) for b in blocks if _normalize_block(b)]
    if not normalized:
        return []

    sections = _group_into_sections(normalized)
    raw_chunks = _sections_to_chunks(sections, size=size)
    merged = _post_merge_heading_lists(raw_chunks)

    result: list[dict[str, Any]] = []
    for index, chunk in enumerate(merged):
        content = chunk["content"].strip()
        if not content:
            continue
        result.append(
            {
                "content": content,
                "metadata": {
                    "chunk_index": index,
                    "section_title": chunk.get("section_title"),
                    "is_heading_only": bool(chunk.get("is_heading_only")),
                },
            }
        )

    logger.debug(
        "Structured chunking done | blocks={} sections={} chunks={}",
        len(normalized),
        len(sections),
        len(result),
    )
    return result


def _is_structured(blocks: list[str] | list[dict[str, Any]] | str) -> bool:
    if isinstance(blocks, str):
        return False
    if not blocks:
        return False
    first = blocks[0]
    return isinstance(first, dict) and "content" in first


def _normalize_block(block: dict[str, Any]) -> dict[str, Any] | None:
    content = str(block.get("content") or "").strip()
    if not content:
        return None
    block_type = str(block.get("type") or "text")
    if block_type not in {"heading", "list", "text", "table"}:
        block_type = "text"
    if block_type == "text" and _NUMBERED_ITEM_RE.match(content):
        block_type = "list"
    level = int(block.get("level") or 0)
    return {"type": block_type, "content": content, "level": level}


def _group_into_sections(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group blocks into sections keyed by heading boundaries."""
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def start_section(
        *,
        title: str | None,
        level: int,
        heading_block: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body: list[dict[str, Any]] = []
        section = {
            "title": title,
            "level": level,
            "heading": heading_block,
            "body": body,
        }
        sections.append(section)
        return section

    for block in blocks:
        if block["type"] == "heading":
            current = start_section(
                title=block["content"],
                level=int(block["level"] or 1),
                heading_block=block,
            )
            continue

        if current is None:
            current = start_section(title=None, level=0, heading_block=None)
        current["body"].append(block)

    return sections


def _sections_to_chunks(
    sections: list[dict[str, Any]],
    *,
    size: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for section in sections:
        chunks.extend(_chunk_section(section, size=size))
    return chunks


def _chunk_section(section: dict[str, Any], *, size: int) -> list[dict[str, Any]]:
    title = section.get("title")
    heading = section.get("heading")
    body: list[dict[str, Any]] = list(section.get("body") or [])
    soft_limit = max(size * 2, size)

    if heading is None and not body:
        return []

    # Atomic units: heading text, then body units. Consecutive list items
    # stay as one inseparable unit so we never cut a list mid-sequence.
    units: list[tuple[str, str]] = []
    if heading is not None:
        units.append(("heading", str(heading["content"])))

    list_run: list[str] = []
    for block in body:
        text = str(block["content"])
        if block["type"] == "list":
            list_run.append(text)
            continue
        if list_run:
            units.append(("list", "\n".join(list_run)))
            list_run = []
        units.append(("text", text))
    if list_run:
        units.append(("list", "\n".join(list_run)))

    if not units:
        return []

    packed: list[str] = []
    current = ""
    current_has_heading = False

    def flush() -> None:
        nonlocal current, current_has_heading
        if current.strip():
            packed.append(current.strip())
        current = ""
        current_has_heading = False

    for kind, unit in units:
        candidate = f"{current}\n{unit}".strip() if current else unit

        if not current:
            current = unit
            current_has_heading = kind == "heading"
            continue

        # Always keep heading with the next unit (esp. numbered list).
        if current_has_heading and len(current) < _HEADING_ONLY_MAX_CHARS:
            current = candidate
            current_has_heading = False
            continue

        # Keep packing while under size.
        if len(candidate) <= size:
            current = candidate
            current_has_heading = False
            continue

        # Soft overflow to avoid splitting a list or heading+list.
        if kind == "list" and len(candidate) <= soft_limit:
            current = candidate
            current_has_heading = False
            continue

        # List units are never character-split; emit whole even if oversized.
        if kind == "list":
            flush()
            packed.append(unit)
            continue

        flush()
        if len(unit) <= size:
            current = unit
            current_has_heading = kind == "heading"
            continue
        # Oversized non-list text: hard-split by characters.
        parts = _chunk_by_chars(unit, size=size, overlap=0)
        packed.extend(parts)

    flush()

    result: list[dict[str, Any]] = []
    heading_text = str(heading["content"]).strip() if heading is not None else None
    for index, content in enumerate(packed):
        is_heading_only = (
            heading_text is not None and content.strip() == heading_text
        )
        result.append(
            {
                "content": content,
                "section_title": title,
                "is_heading_only": is_heading_only,
            }
        )
    return result


def _post_merge_heading_lists(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge short heading-only chunks into the following numbered list."""
    if not chunks:
        return []

    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        content = str(current["content"]).strip()
        next_chunk = chunks[index + 1] if index + 1 < len(chunks) else None

        should_merge = (
            next_chunk is not None
            and (
                current.get("is_heading_only")
                or len(content) < _HEADING_ONLY_MAX_CHARS
            )
            and _starts_with_numbered_item(str(next_chunk["content"]))
            and not _starts_with_numbered_item(content)
        )
        if should_merge and next_chunk is not None:
            combined = f"{content}\n{next_chunk['content']}".strip()
            merged.append(
                {
                    "content": combined,
                    "section_title": current.get("section_title")
                    or next_chunk.get("section_title"),
                    "is_heading_only": False,
                }
            )
            index += 2
            continue

        merged.append(current)
        index += 1
    return merged


def _starts_with_numbered_item(text: str) -> bool:
    first_line = text.strip().split("\n", 1)[0].strip()
    return bool(_NUMBERED_ITEM_RE.match(first_line))


def _flatten_paragraphs(blocks: list[str]) -> list[str]:
    paragraphs: list[str] = []
    for block in blocks:
        if not block or not block.strip():
            continue
        for line in block.split("\n"):
            text = line.strip()
            if text:
                paragraphs.append(text)
    return paragraphs


def _pack_paragraphs(
    paragraphs: list[str],
    *,
    size: int,
    overlap: int,
) -> list[str]:
    chunks: list[str] = []
    current = ""
    soft_limit = size * 2

    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= size:
            current = candidate
            continue

        # Do not cut between a short heading and a numbered list.
        if (
            current
            and len(current) < _HEADING_ONLY_MAX_CHARS
            and _starts_with_numbered_item(paragraph)
            and not _starts_with_numbered_item(current)
            and len(candidate) <= soft_limit
        ):
            current = candidate
            continue

        # Do not split a numbered list in the middle.
        if (
            current
            and _ends_with_numbered_item(current)
            and _starts_with_numbered_item(paragraph)
            and len(candidate) <= soft_limit
        ):
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
        if current:
            chunks.append(current)
            current = ""
        parts = _chunk_by_chars(paragraph, size=size, overlap=overlap)
        chunks.extend(parts[:-1])
        current = parts[-1] if parts else ""

    if current:
        chunks.append(current)

    limited = [
        _hard_limit(chunk, soft_limit) for chunk in chunks if chunk.strip()
    ]
    return _post_merge_plain(limited)

def _post_merge_plain(chunks: list[str]) -> list[str]:
    """Post-merge for plain string chunks (heading + numbered list)."""
    if not chunks:
        return []
    merged: list[str] = []
    index = 0
    while index < len(chunks):
        current = chunks[index].strip()
        nxt = chunks[index + 1].strip() if index + 1 < len(chunks) else None
        if (
            nxt is not None
            and len(current) < _HEADING_ONLY_MAX_CHARS
            and not _starts_with_numbered_item(current)
            and _starts_with_numbered_item(nxt)
        ):
            merged.append(f"{current}\n{nxt}".strip())
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def _ends_with_numbered_item(text: str) -> bool:
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return False
    return bool(_NUMBERED_ITEM_RE.match(lines[-1]))


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
