"""Normalize and interpret knowledge chunk tags."""

from __future__ import annotations

from typing import Any


def normalize_knowledge_tags(
    tags: list[str] | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize tags into ``{system, audience}`` shape when possible.

    - ``None`` / empty → ``None``
    - already has ``system`` and/or ``audience`` → kept as-is (plus extras)
    - plain list → stored under ``system.labels``
    - flat dict without reserved keys → treated as ``audience``
    """
    if tags is None:
        return None

    if isinstance(tags, list):
        labels = [str(item) for item in tags if str(item).strip()]
        if not labels:
            return None
        return {"system": {"labels": labels}}

    if not isinstance(tags, dict):
        return None

    if not tags:
        return None

    if "system" in tags or "audience" in tags:
        normalized: dict[str, Any] = {}
        system = tags.get("system")
        audience = tags.get("audience")
        if isinstance(system, dict) and system:
            normalized["system"] = system
        elif system is not None and not isinstance(system, dict):
            normalized["system"] = {"value": system}
        if isinstance(audience, dict) and audience:
            normalized["audience"] = audience
        elif audience is not None and not isinstance(audience, dict):
            normalized["audience"] = {"value": audience}
        for key, value in tags.items():
            if key in {"system", "audience"}:
                continue
            normalized[key] = value
        return normalized or None

    # Flat dict: treat as audience segment criteria.
    return {"audience": dict(tags)}


def extract_audience(
    tags: list[str] | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return audience dict if present and non-empty."""
    if not isinstance(tags, dict):
        return None
    audience = tags.get("audience")
    if isinstance(audience, dict) and audience:
        return audience
    return None


def extract_filter_audience(
    filter_tags: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return audience criteria from a search/answer filter_tags payload."""
    if not isinstance(filter_tags, dict):
        return None
    audience = filter_tags.get("audience")
    if isinstance(audience, dict) and audience:
        return audience
    return None


def chunk_visible_for_filter(
    tags: list[str] | dict[str, Any] | None,
    filter_tags: dict[str, Any] | None,
) -> bool:
    """Pure-Python visibility rules mirroring DAO audience filtering.

    - No audience filter → hide chunks that have ``tags.audience``.
    - With audience filter → show general chunks and chunks whose audience
      JSON contains the filter audience (JSONB ``@>`` semantics).
    """
    chunk_audience = extract_audience(tags)
    filter_audience = extract_filter_audience(filter_tags)

    if filter_audience is None:
        return chunk_audience is None

    if chunk_audience is None:
        return True

    return _json_contains(chunk_audience, filter_audience)


def _json_contains(left: Any, right: Any) -> bool:
    """Subset match equivalent to Postgres JSONB containment ``left @> right``."""
    if right is None:
        return left is None
    if isinstance(right, dict):
        if not isinstance(left, dict):
            return False
        for key, value in right.items():
            if key not in left:
                return False
            if not _json_contains(left[key], value):
                return False
        return True
    if isinstance(right, list):
        if not isinstance(left, list):
            return False
        for item in right:
            if not any(_json_contains(candidate, item) for candidate in left):
                return False
        return True
    return left == right
