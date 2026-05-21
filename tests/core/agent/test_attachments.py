"""Tests for the shared file_info → chip-shape projector."""

from xagent.core.agent.attachments import project_file_info_to_chip


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
