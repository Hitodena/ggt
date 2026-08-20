"""Helpers for upload filenames from multipart forms."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


def normalize_upload_filename(filename: str | None, *, fallback: str = "upload.bin") -> str:
    """Return a safe basename with UTF-8 mojibake repaired when possible.

    Browsers/proxies sometimes send non-ASCII names latin-1-decoded or as
    ``filename*=UTF-8''...``. We repair common cases and strip path components.
    """
    raw = (filename or "").strip()
    if not raw:
        return fallback

    repaired = _repair_encoding(raw)
    # RFC 5987 style remnant: UTF-8''%D0%A3...
    if "''" in repaired:
        _, _, encoded = repaired.partition("''")
        if encoded:
            repaired = _repair_encoding(unquote(encoded))

    name = Path(repaired.replace("\\", "/")).name.strip()
    return name or fallback


def _repair_encoding(value: str) -> str:
    # Classic mojibake: UTF-8 bytes interpreted as latin-1/cp1252.
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    try:
        return value.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
