"""Normalize and interpret knowledge chunk tags."""

from __future__ import annotations

from typing import Any

_TAG_LIST_KEYS = ("audience", "clinical", "labels")


def _coerce_string_list(value: Any) -> list[str] | None:
    """Coerce a value into a non-empty list of trimmed strings.

    - list → stringified non-empty items
    - legacy dict ``{"gender": "male"}`` → ``["gender:male"]``
    - single scalar → one-element list
    - empty / unusable → ``None``
    """
    if value is None:
        return None

    if isinstance(value, dict):
        items = [
            f"{key}:{item}"
            for key, item in value.items()
            if str(key).strip() and str(item).strip()
        ]
        return items or None

    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None

    text = str(value).strip()
    return [text] if text else None


def normalize_knowledge_tags(
    tags: list[str] | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize tags into ``{audience, clinical, labels, system}`` shape.

    - ``None`` / empty → ``None``
    - plain list → ``{"labels": [...]}``
    - ``audience`` / ``clinical`` / ``labels`` coerced to ``list[str]``
      (legacy dicts become ``["key:value", ...]``; unknown strings kept —
      no whitelist, never raises)
    - ``system`` kept as a dict when present
    - flat dict without reserved keys → treated as legacy audience criteria
    """
    if tags is None:
        return None

    if isinstance(tags, list):
        labels = _coerce_string_list(tags)
        if not labels:
            return None
        return {"labels": labels}

    if not isinstance(tags, dict):
        return None

    if not tags:
        return None

    reserved = {"system", *_TAG_LIST_KEYS}
    has_reserved = any(key in tags for key in reserved)

    if not has_reserved:
        # Flat legacy dict: treat as audience segment criteria.
        audience = _coerce_string_list(tags)
        return {"audience": audience} if audience else None

    normalized: dict[str, Any] = {}

    system = tags.get("system")
    if isinstance(system, dict) and system:
        # Drop deprecated system.labels — labels live at top level now.
        cleaned_system = {
            key: value for key, value in system.items() if key != "labels"
        }
        if cleaned_system:
            normalized["system"] = cleaned_system
    elif system is not None and not isinstance(system, dict):
        normalized["system"] = {"value": system}

    for key in _TAG_LIST_KEYS:
        if key not in tags:
            continue
        coerced = _coerce_string_list(tags.get(key))
        if coerced:
            normalized[key] = coerced

    # Migrate legacy system.labels into top-level labels if not already set.
    if (
        "labels" not in normalized
        and isinstance(system, dict)
        and "labels" in system
    ):
        migrated = _coerce_string_list(system.get("labels"))
        if migrated:
            normalized["labels"] = migrated

    return normalized or None


def extract_tag_list(
    tags: list[str] | dict[str, Any] | None,
    key: str,
) -> list[str] | None:
    """Return a non-empty string list for ``key``, or ``None``."""
    if not isinstance(tags, dict):
        return None
    return _coerce_string_list(tags.get(key))


def extract_audience(
    tags: list[str] | dict[str, Any] | None,
) -> list[str] | None:
    """Return audience string list if present and non-empty."""
    return extract_tag_list(tags, "audience")


def extract_chunk_system_meta(
    tags: list[str] | dict[str, Any] | None,
) -> tuple[int | None, str | None, bool | None]:
    """Return ``(chunk_index, section_title, is_heading_only)`` from tags."""
    if not isinstance(tags, dict):
        return None, None, None
    system = tags.get("system")
    if not isinstance(system, dict):
        return None, None, None
    chunk_index = system.get("chunk_index")
    section_title = system.get("section_title")
    is_heading_only = system.get("is_heading_only")
    return (
        int(chunk_index) if isinstance(chunk_index, int) else None,
        str(section_title) if isinstance(section_title, str) else None,
        bool(is_heading_only) if is_heading_only is not None else None,
    )


def extract_filter_list(
    filter_tags: dict[str, Any] | None,
    key: str,
) -> list[str] | None:
    """Return non-empty string list for a filter category, or ``None``."""
    if not isinstance(filter_tags, dict):
        return None
    return _coerce_string_list(filter_tags.get(key))


def extract_filter_audience(
    filter_tags: dict[str, Any] | None,
) -> list[str] | None:
    """Return audience criteria from a search/answer filter_tags payload."""
    return extract_filter_list(filter_tags, "audience")


def chunk_visible_for_filter(
    tags: list[str] | dict[str, Any] | None,
    filter_tags: dict[str, Any] | None,
) -> bool:
    """Pure-Python visibility rules mirroring DAO tag filtering.

    - No audience filter → hide chunks that have non-empty ``tags.audience``.
    - With audience filter → show general chunks and chunks whose audience
      list contains the filter audience (JSONB ``@>`` / subset semantics).
    - If ``filter_tags`` has ``clinical`` / ``labels``, additionally require
      those lists to contain the filter lists (AND across categories).
    """
    chunk_audience = extract_audience(tags)
    filter_audience = extract_filter_audience(filter_tags)

    if filter_audience is None:
        if chunk_audience is not None:
            return False
    elif chunk_audience is not None and not _json_contains(
        chunk_audience, filter_audience
    ):
        return False

    for key in ("clinical", "labels"):
        filter_list = extract_filter_list(filter_tags, key)
        if filter_list is None:
            continue
        chunk_list = extract_tag_list(tags, key)
        if chunk_list is None or not _json_contains(chunk_list, filter_list):
            return False

    return True


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
