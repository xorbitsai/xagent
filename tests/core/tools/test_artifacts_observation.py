from pathlib import Path

from xagent.core.tools.artifacts import (
    format_tool_result_for_observation,
    snapshot_generated_artifact_files,
)


def test_format_tool_result_for_observation_hides_image_path_when_artifact_exists():
    observation = format_tool_result_for_observation(
        "generate_image",
        {
            "success": True,
            "image_path": "/Users/example/uploads/generated_image.png",
            "file_id": "582e7b79-4de9-4905-b73b-7d5a70ad64fe",
            "artifacts": [
                {
                    "type": "image",
                    "file_id": "582e7b79-4de9-4905-b73b-7d5a70ad64fe",
                    "filename": "generated_image.png",
                    "mime_type": "image/png",
                    "display": "inline",
                }
            ],
        },
    )

    assert "/Users/example/uploads/generated_image.png" not in observation
    assert (
        "![generated_image.png](file:582e7b79-4de9-4905-b73b-7d5a70ad64fe)"
        in observation
    )
    assert "file preview service" in observation
    assert "/api/files/public/preview/" not in observation


def test_format_tool_result_for_observation_returns_plain_string_without_artifacts():
    result = {"success": True, "output": "done"}

    assert format_tool_result_for_observation("tool", result) == str(result)


def test_format_tool_result_for_observation_strips_file_ref_paths():
    observation = format_tool_result_for_observation(
        "execute_python_code",
        {
            "success": True,
            "generated_files": ["report.docx"],
            "file_refs": [
                {
                    "file_id": "doc-file-id",
                    "filename": "report.docx",
                    "file_path": "/tmp/xagent/output/report.docx",
                    "relative_path": "report.docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                }
            ],
            "artifacts": [
                {
                    "type": "document",
                    "file_id": "doc-file-id",
                    "filename": "report.docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "display": "inline",
                }
            ],
        },
    )

    assert "/tmp/xagent/output/report.docx" not in observation
    assert "file_path" not in observation
    assert "relative_path" in observation
    assert "[report.docx](file:doc-file-id)" in observation


def test_format_tool_result_for_observation_strips_singular_file_ref_paths():
    observation = format_tool_result_for_observation(
        "pptx_tool",
        {
            "success": True,
            "file_ref": {
                "file_id": "deck-file-id",
                "filename": "deck.pptx",
                "file_path": "/tmp/xagent/output/deck.pptx",
                "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            },
            "artifacts": [
                {
                    "type": "presentation",
                    "file_id": "deck-file-id",
                    "filename": "deck.pptx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "display": "inline",
                }
            ],
        },
    )

    assert "/tmp/xagent/output/deck.pptx" not in observation
    assert "file_path" not in observation
    assert "[deck.pptx](file:deck-file-id)" in observation


def test_format_tool_result_for_observation_mentions_office_artifact_links():
    observation = format_tool_result_for_observation(
        "execute_python_code",
        {
            "success": True,
            "artifacts": [
                {
                    "type": "document",
                    "file_id": "doc-file-id",
                    "filename": "report.docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "display": "inline",
                },
                {
                    "type": "spreadsheet",
                    "file_id": "sheet-file-id",
                    "filename": "data.xlsx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "display": "inline",
                },
                {
                    "type": "presentation",
                    "file_id": "slides-file-id",
                    "filename": "deck.pptx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "display": "inline",
                },
            ],
        },
    )

    assert "[report.docx](file:doc-file-id)" in observation
    assert "[data.xlsx](file:sheet-file-id)" in observation
    assert "[deck.pptx](file:slides-file-id)" in observation
    assert "Markdown/chat file" in observation


def test_format_tool_result_for_observation_normalizes_artifact_type_case():
    observation = format_tool_result_for_observation(
        "generate_image",
        {
            "success": True,
            "artifacts": [
                {
                    "type": "Image",
                    "file_id": "image-file-id",
                    "filename": "plot.png",
                    "mime_type": "image/png",
                    "display": "inline",
                }
            ],
        },
    )

    assert "![plot.png](file:image-file-id)" in observation
    assert "Markdown/chat image" in observation


def test_snapshot_generated_artifact_files_skips_files_deleted_before_stat(
    tmp_path, monkeypatch
):
    deleted_before_stat = tmp_path / "deleted.pdf"
    deleted_before_stat.write_bytes(b"pdf")
    original_stat = Path.stat

    def stat_with_deleted_file(self, *args, **kwargs):
        if self == deleted_before_stat:
            raise FileNotFoundError
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_with_deleted_file)

    assert snapshot_generated_artifact_files(tmp_path) == {}


def test_snapshot_generated_artifact_files_allows_hidden_root_ancestor(tmp_path):
    hidden_root = tmp_path / ".xagent-hidden-review" / "workspace" / "output"
    hidden_root.mkdir(parents=True)
    report = hidden_root / "report.docx"
    report.write_bytes(b"docx")
    hidden_descendant = hidden_root / ".cache" / "ignored.docx"
    hidden_descendant.parent.mkdir()
    hidden_descendant.write_bytes(b"docx")

    snapshot = snapshot_generated_artifact_files(hidden_root)

    assert report in snapshot
    assert hidden_descendant not in snapshot


def test_artifact_type_for_filename_pptx_vs_ppt_boundary():
    """Only OOXML .pptx is emitted as ``presentation``; legacy binary
    .ppt must fall through to ``file`` so it doesn't reach the frontend
    ``PptxPreviewRenderer`` (pptxviewjs supports only .pptx).
    """
    from xagent.core.tools.artifacts import artifact_type_for_filename

    assert artifact_type_for_filename("deck.pptx") == "presentation"
    assert artifact_type_for_filename("DECK.PPTX") == "presentation"  # case-insensitive
    assert artifact_type_for_filename("legacy.ppt") == "file"
    assert artifact_type_for_filename("LEGACY.PPT") == "file"
