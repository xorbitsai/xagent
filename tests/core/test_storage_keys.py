"""Characterization tests for the canonical storage-key builders.

The builders in ``xagent.core.file_storage.keys`` replace two near-duplicate
implementations (``workspace._build_workspace_storage_key`` and
``managed_file_ref.build_task_output_storage_key``). For every well-formed
input the canonical builder must produce the exact key the old builders
agreed on; the degenerate-input fallbacks are pinned here as the single
documented behavior.
"""

import pytest

from xagent.core.file_storage import keys as storage_keys
from xagent.core.file_storage import normalize_storage_key
from xagent.core.file_storage.keys import (
    build_task_output_storage_key,
    build_upload_storage_key,
    safe_storage_filename,
)


@pytest.mark.parametrize(
    ("relative_path", "expected_suffix"),
    [
        # Well-formed inputs: both legacy builders produced exactly this.
        ("report.txt", "report.txt"),
        ("nested/dir/report.txt", "nested/dir/report.txt"),
        ("output/data.csv", "output/data.csv"),
        # Collapsible noise: both legacy builders normalized to the same key.
        ("nested/./report.txt", "nested/report.txt"),
        ("nested//report.txt", "nested/report.txt"),
        ("nested/report.txt/", "nested/report.txt"),
    ],
)
def test_task_output_key_matches_legacy_builders_for_well_formed_paths(
    relative_path, expected_suffix
):
    key = build_task_output_storage_key(7, 42, "file-id", relative_path)
    assert key == f"users/7/tasks/42/outputs/file-id/{expected_suffix}"


@pytest.mark.parametrize(
    ("relative_path", "expected_suffix"),
    [
        # Traversal component: fall back to a safe basename (the
        # managed_file_ref behavior; the workspace builder used to drop the
        # ".." components instead — no call site produces such input).
        ("a/../b/report.txt", "report.txt"),
        ("../report.txt", "report.txt"),
        # Nothing usable left: fall back to "file" (the workspace behavior;
        # the managed_file_ref builder used to degenerate to ".").
        ("", "file"),
        (".", "file"),
        ("//", "file"),
        # Traversal-only path: basename would be ".." — degrade to "file".
        ("a/..", "file"),
        # Leading slash: strip, keep structure (both legacy builders differed
        # only in degenerate ways here).
        ("/abs/report.txt", "abs/report.txt"),
        # Forbidden characters are sanitized so strict writes accept the key.
        ("dir/back\\slash.txt", "dir/back_slash.txt"),
        ("dir/ctrl\x00char.txt", "dir/ctrl_char.txt"),
    ],
)
def test_task_output_key_degenerate_input_fallbacks(relative_path, expected_suffix):
    key = build_task_output_storage_key(7, 42, "file-id", relative_path)
    assert key == f"users/7/tasks/42/outputs/file-id/{expected_suffix}"


@pytest.mark.parametrize(
    ("filename", "expected_name"),
    [
        ("report.txt", "report.txt"),
        ("dir/report.txt", "report.txt"),
        ("  spaced.txt  ", "spaced.txt"),
        ("", "file"),
        (".", "file"),
        ("..", "file"),
        ("back\\slash.txt", "back_slash.txt"),
        ("ctrl\x1fchar.txt", "ctrl_char.txt"),
    ],
)
def test_upload_key_uses_safe_filename(filename, expected_name):
    assert safe_storage_filename(filename) == expected_name
    assert (
        build_upload_storage_key(7, "file-id", filename)
        == f"users/7/uploads/file-id/{expected_name}"
    )


def test_upload_generation_key_preserves_scope_and_sanitizes_filename():
    assert hasattr(storage_keys, "build_upload_generation_storage_key")
    key = storage_keys.build_upload_generation_storage_key(
        7,
        "legacy-file-id",
        "dir/report.txt",
        generation="12345678123456781234567812345678",
        scope_segments=("workforces", "9"),
    )

    assert key == (
        "users/7/workforces/9/uploads/legacy-file-id/_versions/"
        "12345678123456781234567812345678/report.txt"
    )
    assert normalize_storage_key(key) == key


@pytest.mark.parametrize(
    ("builder", "args"),
    [
        (build_upload_storage_key, (7, "file-id", "back\\slash\x00.txt")),
        (build_task_output_storage_key, (7, 42, "file-id", "a/../\\weird\x1b//")),
        (build_task_output_storage_key, (7, 42, "file-id", "")),
    ],
)
def test_builder_output_always_passes_strict_normalization(builder, args):
    key = builder(*args)
    assert normalize_storage_key(key) == key
