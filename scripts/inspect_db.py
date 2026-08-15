#!/usr/bin/env python
"""Read-only inspection of Neon KB schema."""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings


async def main() -> int:
    settings = get_settings()
    engine = create_async_engine(settings.sqlalchemy_url(), pool_pre_ping=True)
    async with engine.connect() as conn:
        version = (
            await conn.execute(text("select version()"))
        ).scalar_one()
        print("version:", version)

        exts = (
            await conn.execute(
                text(
                    "select extname, extversion from pg_extension "
                    "order by extname"
                )
            )
        ).all()
        print("extensions:")
        for name, ver in exts:
            print(f"  {name} {ver}")

        tables = (
            await conn.execute(
                text(
                    "select table_name from information_schema.tables "
                    "where table_schema = 'public' and table_name like 'kb_%' "
                    "order by table_name"
                )
            )
        ).all()
        print("kb tables:")
        for (name,) in tables:
            print(f"  {name}")

        emb = (
            await conn.execute(
                text(
                    "select format_type(a.atttypid, a.atttypmod) "
                    "from pg_attribute a "
                    "join pg_class c on c.oid = a.attrelid "
                    "join pg_namespace n on n.oid = c.relnamespace "
                    "where n.nspname = 'public' and c.relname = 'kb_chunks' "
                    "and a.attname = 'embedding'"
                )
            )
        ).scalar_one()
        print("kb_chunks.embedding:", emb)

        counts = (
            await conn.execute(
                text(
                    "select "
                    "(select count(*) from kb_documents)::text, "
                    "(select count(*) from kb_chunks)::text"
                )
            )
        ).one()
        print(f"counts: documents={counts[0]} chunks={counts[1]}")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
