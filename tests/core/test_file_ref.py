import pytest

from xagent.core.file_ref import (
    build_file_id_ref,
    build_file_ref,
    parse_file_id_ref,
    sanitize_file_ref_for_context,
)


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
