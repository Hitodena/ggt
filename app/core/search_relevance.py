"""Post-filters for semantic search relevance."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")

_SEASON_FORMS = {
    "summer": {
        "лето",
        "летом",
        "летний",
        "летняя",
        "летнее",
        "летние",
        "летнего",
        "летней",
        "летнюю",
        "летних",
    },
    "winter": {
        "зима",
        "зимой",
        "зимний",
        "зимняя",
        "зимнее",
        "зимние",
        "зимнего",
        "зимней",
        "зимнюю",
        "зимних",
    },
}


def search_hit_is_relevant(
    *,
    query: str,
    document_title: str,
    content: str,
    distance: float,
    max_distance: float | None,
) -> bool:
    """Return whether a vector-search hit is safe to expose.

    Vector search always has a nearest neighbor, even for unrelated queries.
    The distance cutoff removes weak nearest neighbors; the season check
    prevents close-but-wrong matches like "winter skin care" hitting a
    summer-only document just because both mention skin care.
    """
    if max_distance is not None and distance > max_distance:
        return False

    query_seasons = _seasons_in_text(query)
    if not query_seasons:
        return True

    hit_text = f"{document_title}\n{content}"
    return bool(query_seasons & _seasons_in_text(hit_text))


def _seasons_in_text(text: str) -> set[str]:
    words = {word.casefold() for word in _WORD_RE.findall(text)}
    return {
        season
        for season, forms in _SEASON_FORMS.items()
        if words & forms
    }
