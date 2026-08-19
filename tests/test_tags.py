"""Unit tests for tag normalization and audience visibility rules."""

from app.core.tags import (
    chunk_visible_for_filter,
    extract_audience,
    normalize_knowledge_tags,
)


def test_normalize_none_and_empty() -> None:
    assert normalize_knowledge_tags(None) is None
    assert normalize_knowledge_tags([]) is None
    assert normalize_knowledge_tags({}) is None


def test_normalize_list_goes_to_system_labels() -> None:
    assert normalize_knowledge_tags(["a", "b"]) == {
        "system": {"labels": ["a", "b"]}
    }


def test_normalize_structured_system_audience() -> None:
    tags = {
        "system": {"filename": "x.pdf"},
        "audience": {"gender": "male", "age_min": 40},
    }
    assert normalize_knowledge_tags(tags) == tags


def test_normalize_flat_dict_becomes_audience() -> None:
    assert normalize_knowledge_tags({"gender": "male", "age_min": 40}) == {
        "audience": {"gender": "male", "age_min": 40}
    }


def test_system_only_tags_are_not_audience() -> None:
    tags = {"system": {"filename": "protocol.pdf", "content_type": "application/pdf"}}
    assert extract_audience(tags) is None
    assert chunk_visible_for_filter(tags, None) is True


def test_without_filter_hides_audience_chunks() -> None:
    audience_tags = {"audience": {"gender": "male", "age_min": 40}}
    general_tags = {"system": {"filename": "x.pdf"}}

    assert chunk_visible_for_filter(audience_tags, None) is False
    assert chunk_visible_for_filter(general_tags, None) is True
    assert chunk_visible_for_filter(None, None) is True
    assert chunk_visible_for_filter(audience_tags, {}) is False


def test_with_audience_filter_keeps_general_and_matching() -> None:
    general = {"system": {"filename": "x.pdf"}}
    male_40 = {"audience": {"gender": "male", "age_min": 40}}
    female = {"audience": {"gender": "female"}}
    filter_tags = {"audience": {"gender": "male"}}

    assert chunk_visible_for_filter(general, filter_tags) is True
    assert chunk_visible_for_filter(male_40, filter_tags) is True
    assert chunk_visible_for_filter(female, filter_tags) is False
    assert chunk_visible_for_filter(
        male_40,
        {"audience": {"gender": "male", "age_min": 40}},
    ) is True
    assert chunk_visible_for_filter(
        male_40,
        {"audience": {"gender": "male", "age_min": 50}},
    ) is False
