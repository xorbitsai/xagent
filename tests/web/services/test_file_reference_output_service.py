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


def test_reconcile_rewrites_label_to_filename_when_extension_is_missing():
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

        assert content == "[generated_video.mp4](file:real-id)"
    finally:
        db.close()


def test_reconcile_rewrites_label_to_filename_for_audio_when_extension_is_missing():
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

        assert content == "[generated_podcast.mp3](file:real-id)"
    finally:
        db.close()


def test_reconcile_rewrites_label_for_m4v_extension():
    # Regression test: the frontend's video regex
    # (inline-file-preview-utils.ts) recognizes .m4v, and
    # artifact_type_for_filename must agree or this rewrite silently
    # no-ops for .m4v the same way it used to for all audio extensions.
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

        assert content == "[generated_video.m4v](file:real-id)"
    finally:
        db.close()


def test_reconcile_rewrites_label_case_insensitively():
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

        assert content == "[CLIP.MP4](file:real-id)"
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


def test_reconcile_skips_label_rewrite_for_filename_with_unsafe_chars():
    # A real filename containing "]" must never be substituted into the
    # label unescaped -- it would terminate the markdown link early and
    # drop the file reference entirely, which is worse than leaving the
    # model's prose label (and its plain-download fallback) in place.
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

        assert content == "[下载视频（MP4）](file:real-id)"
    finally:
        db.close()


def test_reconcile_skips_label_rewrite_for_filename_with_control_chars():
    # A filename containing a blank line would split the paragraph before
    # CommonMark ever parses the inline link -- same failure mode as "]",
    # through a different character class.
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


def test_reconcile_rewrites_empty_label_to_filename_for_media():
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

        assert content == "[generated_video.mp4](file:real-id)"
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
