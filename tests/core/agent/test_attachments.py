"""Tests for the shared file_info → chip-shape projector."""

from xagent.core.agent.attachments import (
    build_image_context_references,
    project_file_info_to_chip,
)


def test_project_keeps_chip_fields_and_strips_paths():
    """Only chip-relevant fields persist; absolute paths must not leak
    (the field reaches the browser via both the attachments column and
    the user_message trace event payload)."""
    raw = [
        {
            "file_id": "uuid-1",
            "name": "normalized.xlsx",
            "original_name": "Q1 Report.xlsx",
            "size": 12345,
            "type": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            "path": "/abs/leak/should/be/stripped.xlsx",
        }
    ]
    assert project_file_info_to_chip(raw) == [
        {
            "file_id": "uuid-1",
            "name": "Q1 Report.xlsx",  # original_name preferred over name
            "size": 12345,
            "type": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        }
    ]


def test_project_drops_entries_without_file_id():
    out = project_file_info_to_chip(
        [
            {"name": "no-id.txt", "size": 1},
            {"file_id": "keep", "name": "keep.txt"},
        ]
    )
    assert out == [{"file_id": "keep", "name": "keep.txt", "size": None, "type": None}]


def test_project_falls_back_to_name_when_original_name_missing():
    assert project_file_info_to_chip([{"file_id": "fid", "name": "x.txt"}]) == [
        {"file_id": "fid", "name": "x.txt", "size": None, "type": None}
    ]


def test_project_falls_back_to_placeholder_when_no_name_at_all():
    assert project_file_info_to_chip([{"file_id": "fid"}]) == [
        {"file_id": "fid", "name": "uploaded file", "size": None, "type": None}
    ]


def test_project_tolerates_garbage_input():
    """Defensive — caller may pass None, a non-list, or list of non-dicts;
    the projector should return [] rather than raise."""
    assert project_file_info_to_chip(None) == []
    assert project_file_info_to_chip("not a list") == []  # type: ignore[arg-type]
    assert project_file_info_to_chip([None, "garbage", 42]) == []  # type: ignore[list-item]
    assert project_file_info_to_chip([]) == []


def test_build_image_context_references_keeps_only_direct_provider_formats():
    references = build_image_context_references(
        [
            {
                "file_id": "png-id",
                "original_name": "diagram.png",
                "type": "image/png",
                "size": 123,
                "path": "/private/uploads/diagram.png",
            },
            {
                "file_id": "svg-id",
                "name": "drawing.svg",
                "type": "image/svg+xml",
            },
            {
                "file_id": "pdf-id",
                "name": "notes.pdf",
                "type": "application/pdf",
            },
        ]
    )

    assert len(references) == 1
    reference = references[0]
    assert reference.file_id == "png-id"
    assert reference.safe_file_ref["filename"] == "diagram.png"
    assert reference.safe_file_ref["mime_type"] == "image/png"
    assert "path" not in reference.durable_dict()["file_ref"]
    assert reference.metadata == {"source": "user_upload"}


def test_build_image_context_references_infers_mime_and_deduplicates_file_ids():
    references = build_image_context_references(
        [
            {"file_id": "image-id", "name": "photo.webp", "type": None},
            {"file_id": "image-id", "name": "duplicate.png", "type": "image/png"},
        ]
    )

    assert [reference.file_id for reference in references] == ["image-id"]
    assert references[0].safe_file_ref["mime_type"] == "image/webp"


def test_build_image_context_references_normalizes_optional_upload_metadata():
    references = build_image_context_references(
        [
            {
                "file_id": "parameterized-mime",
                "name": "diagram.png",
                "type": " Image/PNG; charset=binary ",
                "size": "12kb",
            },
            {
                "file_id": "blank-name",
                "original_name": "   ",
                "name": "  ",
                "type": "image/jpeg",
                "size": -1,
            },
        ]
    )

    assert [reference.file_id for reference in references] == [
        "parameterized-mime",
        "blank-name",
    ]
    assert references[0].safe_file_ref["mime_type"] == "image/png"
    assert "size" not in references[0].safe_file_ref
    assert references[1].safe_file_ref["filename"] == "uploaded image"
    assert "size" not in references[1].safe_file_ref
