from pathlib import Path

import pytest

from xagent.core import file_ref
from xagent.core.file_ref import (
    FILE_REF_OUTPUT_INSTRUCTIONS,
    build_file_id_ref,
    build_file_ref,
    build_workspace_file_ref,
    parse_file_id_ref,
    sanitize_file_ref_for_context,
)
from xagent.core.workspace import TaskWorkspace

FINAL_DELIVERABLE_INSTRUCTION_MARKER = "## FINAL DELIVERABLE FILE REFERENCES"


def test_final_deliverable_instruction_scopes_workspace_lookup_by_capability() -> None:
    assert file_ref.WORKSPACE_OUTPUT_FILES_TOOL_NAME == "get_workspace_output_files"
    lookup_instruction = file_ref.final_deliverable_file_reference_instructions(
        can_lookup=True
    )
    forced_instruction = file_ref.final_deliverable_file_reference_instructions(
        can_lookup=False
    )
    inline_instruction = file_ref.final_deliverable_file_reference_instructions(
        can_lookup=False,
        include_heading=False,
    )

    for instruction in (lookup_instruction, forced_instruction, inline_instruction):
        assert "exact markdown_link" in instruction
        assert "tool result produced a file" in instruction
        assert "trusted non-internal FileRef references one" in instruction
        assert "FileRef produced a file" not in instruction
        assert "trusted non-internal FileRef" in instruction
        assert "trusted public FileRef" not in instruction
        assert "file UUID" not in instruction
        assert "inline_markdown for screenshots" in instruction
        assert "inline image Markdown" in instruction
        assert "image intended for inline display" in instruction
        assert "different exact user-facing rendering" in instruction
        assert "verbatim" in instruction
        assert "filename and extension" in instruction
        assert "Preserve every returned file_id exactly" in instruction
        assert "Do not claim that a file was delivered" not in instruction
        assert "Never invent, guess, shorten" not in instruction
        assert "intermediate" in instruction
        assert "Never include internal FileRefs" in instruction

    assert FINAL_DELIVERABLE_INSTRUCTION_MARKER not in inline_instruction
    assert FINAL_DELIVERABLE_INSTRUCTION_MARKER in lookup_instruction
    assert FINAL_DELIVERABLE_INSTRUCTION_MARKER in forced_instruction
    assert "get_workspace_output_files" in lookup_instruction
    assert "once before finalizing" in lookup_instruction
    assert "no trusted file_id remains" in lookup_instruction
    assert "returned non-null file_id" in lookup_instruction
    assert "file_id or markdown_link" not in lookup_instruction
    assert "render [filename](file:file_id)" in lookup_instruction
    assert "matches the markdown_link form" in lookup_instruction
    assert "canonical markdown_link form" not in lookup_instruction
    assert "do not repeat the lookup" in lookup_instruction
    assert "get_workspace_output_files" not in forced_instruction
    assert "lookup is unavailable" in forced_instruction
    assert "do not reconstruct one or claim delivery" in forced_instruction
    assert "deliverable link is unavailable" in forced_instruction
    assert "lookup is unavailable" not in lookup_instruction
    assert forced_instruction not in FILE_REF_OUTPUT_INSTRUCTIONS
    assert FINAL_DELIVERABLE_INSTRUCTION_MARKER not in FILE_REF_OUTPUT_INSTRUCTIONS
    assert "get_workspace_output_files" in FILE_REF_OUTPUT_INSTRUCTIONS
    assert (
        "trusted provenance for rendering a file link." in FILE_REF_OUTPUT_INSTRUCTIONS
    )
    assert "under the final-deliverable rules" not in FILE_REF_OUTPUT_INSTRUCTIONS
    assert "Do not call final_answer claiming" in FILE_REF_OUTPUT_INSTRUCTIONS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("file:", None),
        ("file://", None),
        (
            "file:355f1fee-48e4-4cb6-afd3-71654e2f5c7e",
            "355f1fee-48e4-4cb6-afd3-71654e2f5c7e",
        ),
        (
            "file://355f1fee-48e4-4cb6-afd3-71654e2f5c7e",
            "355f1fee-48e4-4cb6-afd3-71654e2f5c7e",
        ),
        ("file://legacy%20id", "legacy id"),
        ("355f1fee-48e4-4cb6-afd3-71654e2f5c7e", None),
        ("file:///Users/example/photo.jpg", None),
        ("file://server/share/photo.jpg", None),
        ("file:output/photo.jpg", None),
        ("file://[invalid", None),
        ("https://example.com/photo.jpg", None),
    ],
)
def test_parse_file_id_ref(value: str | None, expected: str | None) -> None:
    assert parse_file_id_ref(value) == expected


def test_build_file_id_ref_uses_canonical_form() -> None:
    result = build_file_id_ref("legacy id")

    assert result == "file:legacy%20id"
    assert parse_file_id_ref(result) == "legacy id"


def test_build_file_ref_uses_canonical_file_id_ref() -> None:
    result = build_file_ref(file_id="file-id", filename="report.txt")

    assert result["markdown_link"] == "[report.txt](file:file-id)"


def test_internal_file_ref_omits_public_links_after_sanitization() -> None:
    result = sanitize_file_ref_for_context(
        build_file_ref(
            file_id="internal-frame",
            filename="frame.png",
            mime_type="image/png",
            internal=True,
        )
    )

    assert result["internal"] is True
    assert result["preview_url"] is None
    assert result["download_url"] is None
    assert result["markdown_link"] is None


def test_internal_file_ref_omits_relative_path_after_sanitization() -> None:
    result = sanitize_file_ref_for_context(
        {
            **build_file_ref(
                file_id="internal-frame",
                filename="frame.png",
                mime_type="image/png",
                internal=True,
            ),
            "relative_path": (
                "temp/.xagent-internal/computer_observations/session/digest.png"
            ),
        }
    )

    assert "relative_path" not in result


def test_internal_workspace_file_ref_omits_relative_path_at_construction(
    tmp_path: Path,
) -> None:
    workspace = TaskWorkspace("task_file_ref", base_dir=str(tmp_path))
    try:
        screenshot = workspace.temp_dir / "frame.png"
        screenshot.write_bytes(b"image")

        result = build_workspace_file_ref(
            workspace=workspace,
            file_path=screenshot,
            mime_type="image/png",
            internal=True,
        )

        assert "relative_path" not in result
    finally:
        workspace.cleanup()


@pytest.mark.parametrize(
    "file_id",
    [
        "",
        ".",
        "..",
        "nested/file-id",
        "nested\\file-id",
        "nested%2Ffile-id",
        "nested%5Cfile-id",
        "%2E%2E",
    ],
)
def test_build_file_id_ref_rejects_path_like_values(file_id: str) -> None:
    with pytest.raises(ValueError):
        build_file_id_ref(file_id)


@pytest.mark.parametrize(
    "relative_path",
    [
        r"C:\Users\name\file.png",
        r"..\secret.png",
        r"images\..\secret.png",
        r"\\server\share\file.png",
        r"\rooted\file.png",
    ],
)
def test_context_file_ref_rejects_windows_shaped_paths(relative_path: str) -> None:
    result = sanitize_file_ref_for_context(
        {
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
            "relative_path": relative_path,
        }
    )

    assert "relative_path" not in result


def test_context_file_ref_normalizes_safe_windows_separators() -> None:
    result = sanitize_file_ref_for_context(
        {
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
            "relative_path": r"images\frame.png",
        }
    )

    assert result["relative_path"] == "images/frame.png"


class DurableOnlyWorkspace:
    def __init__(self, root: Path) -> None:
        self.workspace_dir = root
        self.register_calls = 0

    def get_file_id_from_path(self, path: str) -> None:
        return None

    def register_file(self, path: str) -> str:
        self.register_calls += 1
        return "durable-file"


def test_internal_workspace_file_ref_fails_closed(tmp_path: Path) -> None:
    workspace = DurableOnlyWorkspace(tmp_path)
    screenshot = tmp_path / "frame.png"
    screenshot.write_bytes(b"image")

    with pytest.raises(TypeError, match="internal file registration"):
        build_workspace_file_ref(
            workspace=workspace,
            file_path=screenshot,
            mime_type="image/png",
            internal=True,
        )

    assert workspace.register_calls == 0
