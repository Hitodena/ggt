#!/usr/bin/env python
"""Reindex an existing KB document by deleting and re-uploading the source file.

Old chunks are not mutated in place — re-upload is required after chunking changes.

Examples::

    python scripts/reindex_document.py \\
        --specialist-id SPEC \\
        --title "Уход за кожей летом" \\
        --file "/path/to/Уход за кожей летом.docx"

    python scripts/reindex_document.py \\
        --specialist-id SPEC \\
        --title "Уход за кожей летом" \\
        --file "./summer.docx" \\
        --tags '{"system":{"labels":["summer"]}}'
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from app.core.config import get_settings
from app.dao.knowledge import KnowledgeDAO
from app.db.session import get_session_factory
from app.services.upload import UploadError, UploadService, parse_upload_tags


async def _find_documents(specialist_id: str, title_substr: str):
    factory = get_session_factory()
    async with factory() as session:
        items, _total = await KnowledgeDAO.list_for_specialist(
            session,
            specialist_id=specialist_id,
            limit=200,
            offset=0,
        )
    needle = title_substr.casefold()
    return [doc for doc in items if needle in (doc.title or "").casefold()]


async def reindex(
    *,
    specialist_id: str,
    title: str,
    file_path: Path,
    tags: dict | list | None,
    dry_run: bool,
) -> int:
    if not file_path.is_file():
        logger.error("File not found: {}", file_path)
        return 1

    matches = await _find_documents(specialist_id, title)
    if not matches:
        logger.warning(
            "No existing document matched title={!r} for specialist_id={}",
            title,
            specialist_id,
        )
    else:
        for doc in matches:
            logger.info(
                "Matched document | id={} title={!r} chunks={}",
                doc.id,
                doc.title,
                len(doc.chunks),
            )

    if dry_run:
        logger.info(
            "Dry-run: would delete {} doc(s) and upload {}",
            len(matches),
            file_path,
        )
        return 0

    data = file_path.read_bytes()
    factory = get_session_factory()
    async with factory() as session:
        for doc in matches:
            deleted = await KnowledgeDAO.delete_document(
                session,
                doc.id,
                specialist_id=specialist_id,
            )
            logger.info("Deleted old document | id={} ok={}", doc.id, deleted)

        service = UploadService(session, settings=get_settings())
        try:
            document, job_id = await service.upload(
                specialist_id=specialist_id,
                filename=file_path.name,
                data=data,
                content_type=None,
                title=title,
                tags=tags,
            )
        except UploadError as exc:
            logger.error("Reindex upload failed: {}", exc)
            return 1

    logger.info(
        "Reindex done | document_id={} job_id={} chunks={} title={!r}",
        document.id,
        job_id,
        len(document.chunks),
        document.title,
    )
    for index, chunk in enumerate(document.chunks):
        preview = chunk.content.replace("\n", " ")[:120]
        logger.info(
            "  chunk[{}] chars={} preview={!r}",
            index,
            len(chunk.content),
            preview,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument(
        "--title",
        default="Уход за кожей летом",
        help="Document title to match / store",
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to the source file to re-upload",
    )
    parser.add_argument(
        "--tags",
        default=None,
        help="Optional JSON tags string for upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list matching documents; do not delete/upload",
    )
    args = parser.parse_args(argv)

    try:
        tags = parse_upload_tags(args.tags)
    except UploadError as exc:
        logger.error("{}", exc)
        return 1

    return asyncio.run(
        reindex(
            specialist_id=args.specialist_id,
            title=args.title,
            file_path=args.file,
            tags=tags,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
