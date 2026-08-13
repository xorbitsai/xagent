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


def test_format_tool_result_for_observation_drops_redundant_image_metadata():
    observation = format_tool_result_for_observation(
        "generate_image",
        {
            "success": True,
            "image_path": "/tmp/xagent/output/creative.png",
            "file_id": "image-file-id",
            "artifacts": [
                {
                    "type": "image",
                    "file_id": "image-file-id",
                    "filename": "creative.png",
                    "mime_type": "image/png",
                    "display": "inline",
                }
            ],
            "file_ref": {
                "file_id": "image-file-id",
                "filename": "creative.png",
                "relative_path": "creative.png",
            },
            "generated_files": ["creative.png"],
            "usage": {"prompt_tokens": 12, "total_tokens": 34},
            "task_metric": {"latency_ms": 8123},
            "request_id": "req-abc123",
            "model_used": "gemini-2.5-flash-image",
            "saved_to_workspace": True,
        },
    )

    assert "![creative.png](file:image-file-id)" in observation
    assert "gemini-2.5-flash-image" in observation
    assert "relative_path" in observation
    for dropped in (
        "'artifacts'",
        "generated_files",
        "prompt_tokens",
        "latency_ms",
        "req-abc123",
        "saved_to_workspace",
    ):
        assert dropped not in observation


def test_format_tool_result_for_observation_drops_raw_video_provider_payload():
    observation = format_tool_result_for_observation(
        "generate_video",
        {
            "success": True,
            "video_url": "https://provider.example/signed/clip.mp4?token=secret",
            "last_frame_url": "https://frames.example/last-frame.png",
            "video_path": "/tmp/xagent/output/clip.mp4",
            "file_id": "video-file-id",
            "artifacts": [
                {
                    "type": "video",
                    "file_id": "video-file-id",
                    "filename": "clip.mp4",
                    "mime_type": "video/mp4",
                    "display": "inline",
                }
            ],
            "duration": 5,
            "resolution": "1080p",
            "raw_response": {"data": {"binary": "x" * 4096}},
        },
    )

    assert "[clip.mp4](file:video-file-id)" in observation
    assert "1080p" in observation
    assert "xxxx" not in observation
    assert "provider.example" not in observation
    # last_frame_url is an input to the next generate_video call, not telemetry.
    assert "https://frames.example/last-frame.png" in observation


def test_format_tool_result_for_observation_redacts_the_download_failure_shape():
    # A failed video download reports artifacts=[], which used to skip redaction
    # entirely and leak the raw payload plus the signed URL verbatim.
    observation = format_tool_result_for_observation(
        "generate_video",
        {
            "success": True,
            "video_url": "https://provider.example/signed/clip.mp4?token=secret",
            "video_path": None,
            "file_id": None,
            "artifacts": [],
            "file_ref": {
                "file_id": "vid",
                "filename": "clip.mp4",
                "file_path": "/tmp/xagent/output/clip.mp4",
            },
            "raw_response": {"data": {"binary": "x" * 4096}},
        },
    )

    assert "xxxx" not in observation
    assert "/tmp/xagent/output/clip.mp4" not in observation
    # Without artifact lines the URL is the model's only handle on the result.
    assert "https://provider.example/signed/clip.mp4?token=secret" in observation


def test_media_paths_are_redacted_without_a_file_ref_to_register_them():
    # build_workspace_file_ref failing leaves file_ref None, so nothing else
    # carries the path — the media *_path keys have to stand on their own.
    for key, filename in (
        ("video_path", "clip.mp4"),
        ("audio_path", "voice.mp3"),
        ("transcription_path", "transcript.json"),
    ):
        observation = format_tool_result_for_observation(
            "media_tool",
            {
                "success": True,
                key: f"/Users/someone/.xagent/workspaces/w1/output/{filename}",
                "file_id": "media-file-id",
                "file_ref": None,
            },
        )

        assert "/Users/someone" not in observation, key
        assert key not in observation, key


def test_observation_metadata_drops_exactly_the_excluded_keys():
    from xagent.core.tools.artifacts import (
        _OBSERVATION_EXCLUDED_KEYS,
        _UNBOUNDED_PAYLOAD_KEYS,
        _observation_metadata,
    )

    payload = {
        "success": True,
        "model_used": "gemini",
        "usage": {},
        "task_metric": {},
        "request_id": "r",
        "saved_to_workspace": True,
        "artifacts": [],
        "generated_files": [],
        "video_url": "u",
        "raw_response": {},
        "last_frame_url": "f",
    }

    assert set(_observation_metadata(payload, _OBSERVATION_EXCLUDED_KEYS)) == {
        "success",
        "model_used",
        "last_frame_url",
    }
    assert set(_observation_metadata(payload, _UNBOUNDED_PAYLOAD_KEYS)) == set(
        payload
    ) - {"raw_response"}


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
