"""Tests for semantic search post-filtering."""

from app.core.search_relevance import search_hit_is_relevant


def test_rejects_weak_nearest_neighbor() -> None:
    assert (
        search_hit_is_relevant(
            query="Уход за собакой",
            document_title="Уход за кожей летом",
            content="Правила использования солнцезащитных средств",
            distance=0.6921,
            max_distance=0.62,
        )
        is False
    )


def test_rejects_wrong_season_even_when_embedding_is_close() -> None:
    assert (
        search_hit_is_relevant(
            query="Уход за кожей зимой",
            document_title="Уход за кожей летом",
            content="Правила использования солнцезащитных средств",
            distance=0.3895,
            max_distance=0.62,
        )
        is False
    )


def test_keeps_relevant_sunscreen_query() -> None:
    assert (
        search_hit_is_relevant(
            query="правила солнцезащитных",
            document_title="Уход за кожей летом",
            content=(
                "Правила использования солнцезащитных средств\n"
                "1. Наносите SPF..."
            ),
            distance=0.4607,
            max_distance=0.62,
        )
        is True
    )


def test_keeps_same_season_match() -> None:
    assert (
        search_hit_is_relevant(
            query="Уход за кожей зимой",
            document_title="Зимний уход за кожей",
            content="Зимой коже нужны защита от холода и восстановление.",
            distance=0.31,
            max_distance=0.62,
        )
        is True
    )
