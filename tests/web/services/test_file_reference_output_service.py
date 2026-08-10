from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.artifacts import artifact_type_for_filename
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.file_reference_output_service import (
    _AUDIO_EXTENSIONS,
    reconcile_assistant_file_references,
)


def _create_context():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    user = User(username="owner", password_hash="hashed", is_admin=False)
    db.add(user)
    db.flush()
    task = Task(
        user_id=int(user.id),
        title="FileRef task",
        description="Generate a video",
        status=TaskStatus.COMPLETED,
    )
    db.add(task)
    db.flush()
    return db, user, task


def _add_file(
    db, user, task, *, file_id: str, filename: str, mime_type: str = "video/mp4"
):
    record = UploadedFile(
        file_id=file_id,
        user_id=int(user.id),
        task_id=int(task.id) if task is not None else None,
        filename=filename,
        storage_path=f"/tmp/{file_id}/{filename}",
        mime_type=mime_type,
        file_size=123,
    )
    db.add(record)
    db.flush()
    return record


def test_reconcile_keeps_valid_file_reference():
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_video.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[generated_video.mp4](file:real-id)",
        )

        assert content == "[generated_video.mp4](file:real-id)"
    finally:
        db.close()


def test_reconcile_repairs_invented_id_from_unique_filename():
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="c6553861-5bdd-4628-9b15-1310e34fe499",
            filename="generated_video_253a6da9.mp4",
        )
        _add_file(
            db,
            user,
            None,
            file_id="older-unbound-id",
            filename="generated_video_253a6da9.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=(
                "下载：[generated_video_253a6da9.mp4]"
                "(file:253a6da9-76e1-4b16-b26e-2eba2d8b0583)"
            ),
        )

        assert content == (
            "下载：[generated_video_253a6da9.mp4]"
            "(file:c6553861-5bdd-4628-9b15-1310e34fe499)"
        )
    finally:
        db.close()


def test_reconcile_unlinks_unknown_or_ambiguous_file_reference():
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="first-id", filename="report.mp4")
        _add_file(db, user, task, file_id="second-id", filename="report.mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=(
                "[missing.mp4](file:invented-id) and "
                "[report.mp4](file:another-invented-id)"
            ),
        )

        assert content == "missing.mp4 and report.mp4"
        assert "file:" not in content
    finally:
        db.close()


def test_reconcile_unlinks_record_with_invalid_file_id():
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="invalid/id",
            filename="generated_video.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[generated_video.mp4](file:invented-id)",
        )

        assert content == "generated_video.mp4"
    finally:
        db.close()


def test_reconcile_adds_title_and_keeps_label_when_extension_is_missing():
    # The primary mechanism as of #1202: the model's own (localized) label
    # survives untouched in the persisted transcript; the real filename
    # rides in the link *title* instead, which the frontend already prefers
    # for type detection over the label.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_video.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载视频（MP4）](file:real-id)",
        )

        assert content == '[下载视频（MP4）](file:real-id "generated_video.mp4")'
    finally:
        db.close()


def test_reconcile_adds_title_for_audio_when_extension_is_missing():
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_podcast.mp3",
            mime_type="audio/mpeg",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载音频（MP3）](file:real-id)",
        )

        assert content == '[下载音频（MP3）](file:real-id "generated_podcast.mp3")'
    finally:
        db.close()


def test_reconcile_adds_title_for_m4v_extension():
    # Regression test: the frontend's video regex
    # (inline-file-preview-utils.ts) recognizes .m4v, and
    # artifact_type_for_filename must agree or this silently no-ops for
    # .m4v the same way it used to for all audio extensions.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_video.m4v",
            mime_type="video/mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载视频（M4V）](file:real-id)",
        )

        assert content == '[下载视频（M4V）](file:real-id "generated_video.m4v")'
    finally:
        db.close()


def test_reconcile_adds_title_case_insensitively():
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="CLIP.MP4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载视频](file:real-id)",
        )

        assert content == '[下载视频](file:real-id "CLIP.MP4")'
    finally:
        db.close()


def test_reconcile_self_heals_stale_title_to_current_filename():
    # The title is always overwritten (when the label doesn't already
    # reveal the type), so a stale title from before a rename can never
    # linger and point at the wrong file.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="renamed_video.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content='[下载视频（MP4）](file:real-id "old_video_name.mp4")',
        )

        assert content == '[下载视频（MP4）](file:real-id "renamed_video.mp4")'
    finally:
        db.close()


def test_reconcile_keeps_custom_label_for_plain_download_file():
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="notes.txt",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载笔记](file:real-id)",
        )

        assert content == "[下载笔记](file:real-id)"
    finally:
        db.close()


def test_reconcile_keeps_custom_label_for_office_document():
    # Office types are deliberately excluded from label-restore (rewriting
    # them would flip existing compact links into heavy inline preview
    # boxes). Use .docx so this fails loudly if "document" is ever added
    # to the inline-preview media set by mistake.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="quarterly_report.docx",
            mime_type=(
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ),
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载报告](file:real-id)",
        )

        assert content == "[下载报告](file:real-id)"
    finally:
        db.close()


def test_reconcile_keeps_mismatched_label_that_already_has_media_extension():
    # A label that already ends in a valid media suffix is left alone even
    # when it names a different file than the record's real filename —
    # there's no signal that the mismatch is wrong rather than intentional.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_video_a1b2c3.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[my_clip.mp4](file:real-id)",
        )

        assert content == "[my_clip.mp4](file:real-id)"
    finally:
        db.close()


def test_reconcile_uses_title_for_filename_with_bracket_characters():
    # A "]" or "[" has no special meaning inside a quoted title (unlike a
    # label, which the "]" would terminate early), so a bracket-bearing
    # filename now takes the title path instead of degrading to a plain
    # download link the way it did before titles existed.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="clip ].mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载视频（MP4）](file:real-id)",
        )

        assert content == '[下载视频（MP4）](file:real-id "clip ].mp4")'
    finally:
        db.close()


def test_reconcile_falls_back_to_label_rewrite_for_filename_with_quote():
    # A literal double quote can't safely become a title (it would
    # terminate the title clause early), but has no special meaning in a
    # label -- falls back to the pre-title label-rewrite mechanism.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename='clip".mp4',
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载视频（MP4）](file:real-id)",
        )

        assert content == '[clip".mp4](file:real-id)'
    finally:
        db.close()


def test_reconcile_skips_rewrite_for_filename_unsafe_for_both_title_and_label():
    # A filename containing a blank line would split the paragraph before
    # CommonMark ever parses the inline link, whether substituted into a
    # label or a title -- leave the reference untouched in that case.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="a\n\nb.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载视频（MP4）](file:real-id)",
        )

        assert content == "[下载视频（MP4）](file:real-id)"
    finally:
        db.close()


def test_reconcile_adds_title_for_image_syntax_video_reference():
    # A model may wrap a video in image syntax. The frontend's image
    # renderer resolves the preview kind from title/alt, so adding the
    # title turns a would-be broken <img> into a video player while the
    # model's alt text stays untouched.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_video.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="![预览视频](file:real-id)",
        )

        assert content == '![预览视频](file:real-id "generated_video.mp4")'
    finally:
        db.close()


def test_reconcile_keeps_alt_text_for_image_syntax_image_reference():
    # Genuine image references keep their (possibly descriptive) alt text —
    # only video/audio labels are restored.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="chart.png",
            mime_type="image/png",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="![季度收入图表](file:real-id)",
        )

        assert content == "![季度收入图表](file:real-id)"
    finally:
        db.close()


def test_reconcile_adds_title_for_empty_label_media_reference():
    # The frontend falls back to the title when the label is empty
    # (fileName = title || linkText || fileNameFromPath), so an empty label
    # still gets a working preview once the title carries the filename.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_video.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[](file:real-id)",
        )

        assert content == '[](file:real-id "generated_video.mp4")'
    finally:
        db.close()


def test_reconcile_title_injection_is_idempotent():
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_video.mp4",
        )

        once = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载视频（MP4）](file:real-id)",
        )
        twice = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=once,
        )

        assert once == '[下载视频（MP4）](file:real-id "generated_video.mp4")'
        assert twice == once
    finally:
        db.close()


def test_reconcile_round_trips_single_quote_title_for_non_media_reference():
    # Title syntax also allows single quotes. Non-media references don't
    # go through the title-injection logic at all, so this exercises pure
    # input parsing: the parsed title is preserved and re-serialized with
    # the double-quote delimiter this function always writes.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="report.pdf",
            mime_type="application/pdf",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[report](file:real-id 'annual_report_2024.pdf')",
        )

        assert content == '[report](file:real-id "annual_report_2024.pdf")'
    finally:
        db.close()


def test_video_extension_detection_matches_frontend_set():
    # Pins the backend's video-extension coverage to the frontend's regex
    # in inline-file-preview-utils.ts (`/\.(mp4|m4v|mov|webm|mpeg|mpg)$/`).
    # A mismatch here (like the missing .m4v this test was added to catch)
    # makes the label-restore rewrite silently no-op for that extension.
    frontend_video_extensions = {".mp4", ".m4v", ".mov", ".webm", ".mpeg", ".mpg"}
    for extension in frontend_video_extensions:
        assert artifact_type_for_filename(f"clip{extension}") == "video", extension


def test_audio_extension_detection_matches_frontend_set():
    # Same parity guard for audio, pinning _AUDIO_EXTENSIONS against the
    # frontend regex (`/\.(mp3|wav|ogg|opus|flac|m4a|aac)$/`). Audio is the
    # category that actually drifted historically: the initial label-restore
    # fix silently no-oped for every audio extension.
    frontend_audio_extensions = {
        ".mp3",
        ".wav",
        ".ogg",
        ".opus",
        ".flac",
        ".m4a",
        ".aac",
    }
    assert frontend_audio_extensions == _AUDIO_EXTENSIONS


def test_reconcile_reuses_prefetched_records_without_querying():
    db, user, task = _create_context()
    try:
        record = _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_video.mp4",
        )

        with patch.object(
            db,
            "query",
            side_effect=AssertionError("prefetched reconciliation must not query"),
        ):
            content = reconcile_assistant_file_references(
                db,
                task_id=int(task.id),
                user_id=int(user.id),
                content="[generated_video.mp4](file:invented-id)",
                records=[record],
            )

        assert content == "[generated_video.mp4](file:real-id)"
    finally:
        db.close()
