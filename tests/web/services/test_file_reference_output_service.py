import time
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


def test_reconcile_uses_title_for_filename_with_closing_bracket():
    # "]" has no special meaning inside a quoted title (unlike a label,
    # which it would terminate early), so a filename with a closing bracket
    # but no opening one takes the title path instead of degrading to a
    # plain download link the way it did before titles existed.
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


def test_reconcile_skips_rewrite_for_filename_with_opening_bracket():
    # Unlike "]", an opening "[" IS unsafe for a title: the title-parsing
    # regex excludes it so a malformed title can never swallow a
    # subsequent, well-formed [label](file:...) reference whole. A
    # filename with "[" is therefore unsafe for both title and label (the
    # label guard already excluded "[") and the reference is left
    # untouched rather than getting a half-safe rewrite.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="clip [draft.mp4",
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
    # (displayFilename = visibleText || title || fileNameFromPath), so an
    # empty label still gets a working preview once the title carries the
    # filename.
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


def test_reconcile_repairs_invented_id_from_title_when_label_is_prose():
    # The repair heuristic must consult the title, not just the label: once
    # a reference has already been through one title-injection pass, a
    # later pass over content whose original id has since gone invalid can
    # only find the real filename in the title -- the label is untouched
    # model prose that reveals nothing about it.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="c6553861-5bdd-4628-9b15-1310e34fe499",
            filename="generated_video_253a6da9.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=(
                '[下载视频（MP4）](file:invented-id "generated_video_253a6da9.mp4")'
            ),
        )

        assert content == (
            "[下载视频（MP4）](file:c6553861-5bdd-4628-9b15-1310e34fe499 "
            '"generated_video_253a6da9.mp4")'
        )
    finally:
        db.close()


def test_reconcile_repairs_from_title_after_original_record_is_removed():
    # Reproduces the repair-heuristic gap end to end: reconcile once (a
    # title gets injected), delete the underlying record, register a
    # replacement under a new id with the same filename, then reconcile the
    # already-titled content again. Before the title was consulted for
    # repair, this second pass would have dropped the reference to plain
    # text -- the label is untouched prose, so filename-based repair had
    # nothing to match on even though the correct filename was sitting
    # right there in the title.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="original-id",
            filename="generated_video.mp4",
        )

        once = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[下载视频（MP4）](file:original-id)",
        )
        assert once == '[下载视频（MP4）](file:original-id "generated_video.mp4")'

        stale_record = (
            db.query(UploadedFile).filter(UploadedFile.file_id == "original-id").one()
        )
        db.delete(stale_record)
        db.flush()
        _add_file(
            db,
            user,
            task,
            file_id="replacement-id",
            filename="generated_video.mp4",
        )

        twice = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=once,
        )

        assert twice == ('[下载视频（MP4）](file:replacement-id "generated_video.mp4")')
    finally:
        db.close()


def test_reconcile_unlinks_titled_reference_with_ambiguous_filename():
    # The title-aware repair heuristic is still subject to the same
    # ambiguity guard as label-based repair: a filename match that isn't
    # unique can't be trusted to pick the right record.
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="first-id", filename="report.mp4")
        _add_file(db, user, task, file_id="second-id", filename="report.mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content='[下载报告](file:invented-id "report.mp4")',
        )

        # The title survives as plain text: it's this function's own
        # channel for the real filename, and dropping it here would
        # destroy the one clue naming which file was meant.
        assert content == "下载报告 (report.mp4)"
        assert "file:" not in content
    finally:
        db.close()


def test_reconcile_unlinks_titled_reference_when_stored_file_id_is_invalid():
    # Title-based repair can still land on a record whose OWN stored id is
    # malformed; that must still be caught and unlinked (dropping the
    # brackets/link structure), same as the label-based case -- but the
    # title itself survives as plain text rather than being destroyed.
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
            content='[下载视频](file:invented-id "generated_video.mp4")',
        )

        assert content == "下载视频 (generated_video.mp4)"
    finally:
        db.close()


def test_reconcile_keeps_pre_existing_title_when_label_already_reveals_type():
    # When the label alone is already classifiable, the backend must not
    # touch a pre-existing title -- there's no signal that an
    # author-supplied title is wrong just because it differs from the
    # record's current filename (mirrors
    # test_reconcile_keeps_mismatched_label_that_already_has_media_extension
    # for the title case).
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
            content='[my_clip.mp4](file:real-id "custom title.mp4")',
        )

        assert content == '[my_clip.mp4](file:real-id "custom title.mp4")'
    finally:
        db.close()


def test_reconcile_handles_multiple_valid_links_in_one_message():
    # The substitution is a single global regex pass over the whole
    # message; each match must be reconciled independently rather than
    # leaking state (e.g. a stale title) from one match into the next.
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="video-id", filename="generated_video.mp4")
        _add_file(
            db,
            user,
            task,
            file_id="audio-id",
            filename="generated_podcast.mp3",
            mime_type="audio/mpeg",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=(
                "[下载视频（MP4）](file:video-id) and [下载音频（MP3）](file:audio-id)"
            ),
        )

        assert content == (
            '[下载视频（MP4）](file:video-id "generated_video.mp4") and '
            '[下载音频（MP3）](file:audio-id "generated_podcast.mp3")'
        )
    finally:
        db.close()


def test_reconcile_drops_unsafe_pass_through_title_without_touching_label():
    # A non-media reference's title is pure pass-through (no injection
    # logic runs for it). Single-quote syntax lets an input title
    # legitimately contain a literal double quote, but this function
    # always re-emits titles with the double-quote delimiter, so that
    # title can't be reproduced safely -- drop it rather than rewrite the
    # label, which has nothing to do with title safety for non-media types.
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
            content="[report](file:real-id 'annual \"2024\" report.pdf')",
        )

        assert content == "[report](file:real-id)"
    finally:
        db.close()


def test_reconcile_unlinks_reference_with_escaped_quote_in_invented_titles_id():
    # A backslash-escaped quote inside the title must not make the whole
    # [label](file:id "...") construct fail to match -- that would let an
    # invented id skip validation entirely (the pre-title-support regex
    # bypassed this kind of link completely; this asserts the id is
    # actually evaluated and correctly unlinked, not silently ignored). The
    # title itself survives as plain text (raw, backslash included -- same
    # as the label, this function never unescapes it for display).
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="real-id", filename="clip.mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content='[x](file:invented-id "we\\"ird.mp4")',
        )

        assert content == 'x (we\\"ird.mp4)'
    finally:
        db.close()


def test_reconcile_validates_id_despite_unsafe_escaped_title():
    # Complements the unlink case above: here the id is directly valid, so
    # escape-aware title parsing must let the reference survive
    # reconciliation. The raw escaped title itself is still unsafe to
    # re-emit verbatim (see _UNSAFE_TITLE_RE) and is dropped rather than
    # corrupting the output.
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
            content='[report](file:real-id "we\\"ird.pdf")',
        )

        assert content == "[report](file:real-id)"
    finally:
        db.close()


def test_reconcile_validates_id_for_paren_delimited_title():
    # CommonMark also allows a (title) delimiter form. It must be
    # recognized so id validation runs instead of silently skipping the
    # whole reference the way the pre-escape-aware regex did.
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
            content="[report](file:real-id (annual report))",
        )

        assert content == '[report](file:real-id "annual report")'
    finally:
        db.close()


def test_reconcile_does_not_merge_adjacent_links_across_a_malformed_title():
    # A malformed title in one reference must not swallow a second,
    # well-formed reference whole. Before excluding "[" from the
    # title/junk character classes, this exact input collapsed into one
    # match spanning both links and silently dropped id2's reference from
    # the output entirely.
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="id2", filename="clip.mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content='[a](file:id1 "x [b](file:id2 ")',
        )

        # id2's reference is recovered as its own independent, validated
        # match; the malformed leading fragment is left as inert literal
        # text rather than being silently merged away.
        assert '[b](file:id2 "clip.mp4")' in content
        assert content == '[a](file:id1 "x [b](file:id2 "clip.mp4")'
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


def test_reconcile_does_not_strand_content_across_a_malformed_parenthetical():
    # Regression for a real data-corruption bug: the "junk" fallback for a
    # malformed title used to allow "(" through, so it could match only
    # through an inner, unrelated ")" and strand everything after it (up to
    # the link's own closing paren) as literal text outside the link. With
    # "(" excluded, this exact input fails to match at all -- identical to
    # this regex's behavior before title support existed at all, i.e. the
    # correct, non-destructive "leave it untouched" outcome.
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="real-id", filename="report.mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[x](file:real-id (c) 2024) tail",
        )

        assert content == "[x](file:real-id (c) 2024) tail"
    finally:
        db.close()


def test_reconcile_normalizes_unparsable_trailing_junk():
    # Pins the deliberate side effect of the junk fallback: an input like
    # this is inert literal text to CommonMark (invalid destination/title
    # syntax), but the model clearly meant a file link, so reconciliation
    # normalizes it into a live, validated reference and drops the junk.
    # Contrast with the parenthetical case above, which cannot be safely
    # bounded and is left completely untouched instead.
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="real-id", filename="report.mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[report.mp4](file:real-id not a title)",
        )

        assert content == "[report.mp4](file:real-id)"
    finally:
        db.close()


def test_reconcile_validates_id_for_title_with_trailing_whitespace():
    # A title clause is allowed trailing whitespace before the link's
    # closing paren. Without this, the title-form match would fail and fall
    # through to the junk alternative, which has no notion of title syntax
    # and would silently discard the title text -- with no re-injection for
    # a non-media file like this one.
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
            content='[report](file:real-id  "annual report"  )',
        )

        assert content == '[report](file:real-id "annual report")'
    finally:
        db.close()


def test_reconcile_repairs_invented_id_from_label_when_title_names_a_different_file():
    # Mirror-image case of test_reconcile_repairs_invented_id_from_title_...:
    # the title takes priority, but only when it actually resolves to
    # exactly one record. Here the title is unresolvable prose while the
    # label alone is the exact, resolvable filename -- an unconditional
    # title-over-label pick would incorrectly unlink this instead of
    # repairing it. The pre-existing title survives untouched afterwards:
    # the label already reveals the media type ("report.mp4" ends in
    # .mp4), so the title-injection step below has no reason to intervene.
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="report.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content='[report.mp4](file:invented-id "My yearly report")',
        )

        assert content == '[report.mp4](file:real-id "My yearly report")'
    finally:
        db.close()


def test_reconcile_keeps_junk_untouched_when_id_does_not_resolve():
    # Regression: the junk fallback used to be applied unconditionally --
    # even when the id it was attached to didn't resolve to any record, the
    # match still collapsed to the bare label, permanently deleting the
    # junk text (ordinary trailing prose the model wrote) from the
    # persisted transcript. No record is registered here at all, so the id
    # can never resolve or be repaired; the whole match must survive as-is.
    db, user, task = _create_context()
    try:
        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="Check [the docs](file:nonexistent for more details) later",
        )

        assert content == "Check [the docs](file:nonexistent for more details) later"
    finally:
        db.close()


def test_reconcile_keeps_junk_untouched_when_stored_file_id_is_invalid():
    # Same guard as above, at the second (build_file_id_ref) failure point.
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="invalid/id", filename="report.mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[report.mp4](file:invalid/id has more text) tail",
        )

        assert content == "[report.mp4](file:invalid/id has more text) tail"
    finally:
        db.close()


def test_reconcile_still_drops_recoverable_junk_when_id_resolves():
    # Contrast with the two tests above: when the id *does* resolve, junk
    # attached to it is still ordinary discardable trailing text, not
    # something this fix should start preserving. Uses a non-media
    # extension so the assertion isn't entangled with title injection.
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
            content="Check [the docs](file:real-id for more details) later",
        )

        assert content == "Check [the docs](file:real-id) later"
    finally:
        db.close()


def test_reconcile_does_not_let_a_title_span_a_blank_line():
    # A title clause separated from the target by a blank line renders as
    # two independent, inert literal paragraphs to any real CommonMark
    # parser. \s+/\s* around the title forms would match across that blank
    # line and silently merge the two paragraphs into one live link,
    # discarding the second paragraph's text. This function never itself
    # emits a title separated from the target by anything but a single
    # space, so the whole construct must simply fail to match -- identical
    # to this regex's pre-title behavior for any other unparsable input.
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="real-id", filename="stale.mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content='[a](file:real-id\n\n"stale.mp4")',
        )

        assert content == '[a](file:real-id\n\n"stale.mp4")'
    finally:
        db.close()


def test_reconcile_leaves_single_newline_titled_reference_untouched():
    # CommonMark permits one line ending between destination and title
    # (still a single link when rendered), but the horizontal-only
    # whitespace that keeps a title clause from spanning a blank line (see
    # the blank-line test above) declines this shape too. Pinned: the
    # reference is left byte-for-byte untouched -- not validated, but not
    # mangled either -- matching the regex's fallback for every other
    # shape it can't safely parse.
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="real-id", filename="clip.mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content='[a](file:real-id\n"clip.mp4")',
        )

        assert content == '[a](file:real-id\n"clip.mp4")'
    finally:
        db.close()


def test_reconcile_falls_back_to_label_rewrite_for_paren_in_filename():
    # A real filename containing ")" can't safely become this function's
    # own title syntax (see the _UNSAFE_TITLE_RE comment for why that's an
    # implementation constraint, not a CommonMark one) -- it must still
    # fall back to the older label-rewrite mechanism rather than leaving
    # the reference undetectable.
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="real-id", filename="video (1).mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[clip](file:real-id)",
        )

        assert content == "[video (1).mp4](file:real-id)"
    finally:
        db.close()


def test_reconcile_stays_fast_on_a_large_unresolvable_reference():
    # Regression guard for quadratic backtracking between the atomic-group
    # target and the junk alternative when a huge target has no closing
    # paren anywhere: this used to take several seconds per call at ~32KB
    # and scale quadratically, on a function that reruns on every read of
    # attacker-influenceable chat history. Generous bound (real fix runs in
    # well under a second even at far larger sizes) to avoid environment
    # flakiness while still catching a real regression.
    db, user, task = _create_context()
    try:
        content = "[x](file:" + "a" * 100_000
        start = time.monotonic()
        result = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=content,
        )
        elapsed = time.monotonic() - start

        assert result == content
        assert elapsed < 2.0
    finally:
        db.close()


def test_reconcile_does_not_repair_via_whitespace_only_title_or_label():
    # Path("   ".strip()).name is "" -- without an explicit guard, a
    # whitespace-only title/label would look up the empty-string filename
    # key (records_by_filename keys are the raw, unstripped
    # str(filename).casefold()) and could match a record whose own filename
    # is literally empty, silently repairing an invented id to the wrong
    # file. Internal record construction doesn't validate filenames the way
    # the HTTP upload endpoints do, so an empty-filename record is a real
    # (if unusual) possibility, not a hypothetical. Exercises both
    # candidates from the shared (parsed_title, label) repair loop -- a
    # title clause is present here, not just a whitespace-only label, so
    # the title branch of that loop is actually covered.
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="empty-name-id", filename="")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content='[  ](file:invented-id "   ")',
        )

        assert "file:" not in content
    finally:
        db.close()


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
