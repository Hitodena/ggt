"""Unit tests for tag normalization and audience visibility rules."""

from app.core.tags import (
    chunk_visible_for_filter,
    extract_audience,
    extract_chunk_system_meta,
    normalize_knowledge_tags,
)
from app.schemas import KnowledgeSearchRequest
from app.services.upload import UploadService


def test_normalize_none_and_empty() -> None:
    assert normalize_knowledge_tags(None) is None
    assert normalize_knowledge_tags([]) is None
    assert normalize_knowledge_tags({}) is None


def test_normalize_list_goes_to_labels() -> None:
    assert normalize_knowledge_tags(["a", "b"]) == {"labels": ["a", "b"]}


def test_normalize_flat_string_lists() -> None:
    tags = {
        "audience": ["sex:female", "age_bucket:26_35"],
        "clinical": ["procedure:rf_face"],
        "labels": ["manual:spf_лето"],
    }
    assert normalize_knowledge_tags(tags) == tags


def test_normalize_keeps_system_and_migrates_system_labels() -> None:
    tags = {
        "system": {
            "filename": "x.pdf",
            "labels": ["legacy"],
        },
        "audience": ["sex:male"],
    }
    assert normalize_knowledge_tags(tags) == {
        "system": {"filename": "x.pdf"},
        "audience": ["sex:male"],
        "labels": ["legacy"],
    }


def test_normalize_legacy_audience_dict() -> None:
    assert normalize_knowledge_tags(
        {"audience": {"gender": "male", "age_min": 40}}
    ) == {"audience": ["gender:male", "age_min:40"]}


def test_normalize_flat_dict_becomes_audience_strings() -> None:
    assert normalize_knowledge_tags({"gender": "male", "age_min": 40}) == {
        "audience": ["gender:male", "age_min:40"]
    }


def test_normalize_accepts_unknown_label_strings() -> None:
    tags = {
        "audience": ["sex:unknown_value"],
        "labels": ["totally:made_up"],
    }
    assert normalize_knowledge_tags(tags) == tags


def test_system_only_tags_are_not_audience() -> None:
    tags = {
        "system": {
            "filename": "protocol.pdf",
            "content_type": "application/pdf",
        }
    }
    assert extract_audience(tags) is None
    assert chunk_visible_for_filter(tags, None) is True


def test_without_filter_hides_audience_chunks() -> None:
    audience_tags = {"audience": ["sex:male", "age_bucket:40_plus"]}
    general_tags = {"system": {"filename": "x.pdf"}}

    assert chunk_visible_for_filter(audience_tags, None) is False
    assert chunk_visible_for_filter(general_tags, None) is True
    assert chunk_visible_for_filter(None, None) is True
    assert chunk_visible_for_filter(audience_tags, {}) is False
    assert chunk_visible_for_filter({"audience": []}, None) is True


def test_with_audience_filter_keeps_general_and_matching() -> None:
    general = {"system": {"filename": "x.pdf"}}
    male_40 = {"audience": ["sex:male", "age_bucket:40_plus"]}
    female = {"audience": ["sex:female"]}
    filter_tags = {"audience": ["sex:male"]}

    assert chunk_visible_for_filter(general, filter_tags) is True
    assert chunk_visible_for_filter(male_40, filter_tags) is True
    assert chunk_visible_for_filter(female, filter_tags) is False
    assert (
        chunk_visible_for_filter(
            male_40,
            {"audience": ["sex:male", "age_bucket:40_plus"]},
        )
        is True
    )
    assert (
        chunk_visible_for_filter(
            male_40,
            {"audience": ["sex:male", "age_bucket:50_plus"]},
        )
        is False
    )


def test_clinical_and_labels_filter_and() -> None:
    tags = {
        "audience": ["sex:female"],
        "clinical": ["procedure:rf_face"],
        "labels": ["manual:spf_лето"],
    }
    assert (
        chunk_visible_for_filter(
            tags,
            {
                "audience": ["sex:female"],
                "clinical": ["procedure:rf_face"],
            },
        )
        is True
    )
    assert (
        chunk_visible_for_filter(
            tags,
            {
                "audience": ["sex:female"],
                "clinical": ["procedure:other"],
            },
        )
        is False
    )
    assert (
        chunk_visible_for_filter(
            {"audience": ["sex:female"]},
            {"audience": ["sex:female"], "labels": ["manual:spf_лето"]},
        )
        is False
    )


def test_extract_chunk_system_meta() -> None:
    tags = {
        "system": {
            "chunk_index": 2,
            "section_title": "Правила",
            "is_heading_only": False,
            "filename": "x.docx",
        }
    }
    assert extract_chunk_system_meta(tags) == (2, "Правила", False)
    assert extract_chunk_system_meta(None) == (None, None, None)


def test_merge_upload_tags_flat_lists() -> None:
    merged = UploadService._merge_upload_tags(
        {
            "audience": ["sex:female"],
            "clinical": ["procedure:rf_face"],
            "labels": ["manual:spf_лето"],
        },
        filename="note.txt",
        content_type="text/plain",
    )
    assert merged["audience"] == ["sex:female"]
    assert merged["clinical"] == ["procedure:rf_face"]
    assert merged["labels"] == ["manual:spf_лето"]
    assert merged["system"]["filename"] == "note.txt"
    assert merged["system"]["content_type"] == "text/plain"
    assert "labels" not in merged["system"]


def test_merge_upload_tags_plain_list_becomes_labels() -> None:
    merged = UploadService._merge_upload_tags(
        ["manual:a", "manual:b"],
        filename="a.txt",
        content_type=None,
    )
    assert merged["labels"] == ["manual:a", "manual:b"]
    assert merged["system"]["filename"] == "a.txt"


def test_search_request_ignores_extra_fields() -> None:
    payload = KnowledgeSearchRequest.model_validate(
        {
            "specialist_id": "spec-1",
            "query": "spf",
            "limit": 3,
            "external_id": "should-be-ignored",
            "foo": 123,
        }
    )
    assert payload.specialist_id == "spec-1"
    assert payload.query == "spf"
    assert not hasattr(payload, "external_id")
