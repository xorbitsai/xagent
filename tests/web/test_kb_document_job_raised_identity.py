"""The document job rolls back a raised last attempt under its real identity (#2662)."""

from __future__ import annotations

import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.web.api import test_kb_dir as kb_dir
from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    update_collection_stats_sync,
)
from xagent.core.tools.core.RAG_tools.utils.string_utils import (
    generate_deterministic_doc_id,
)
from xagent.web.api import kb as kb_module
from xagent.web.jobs.exceptions import BackgroundJobHandlerError
from xagent.web.jobs.kb_tasks import handle_kb_ingest_document
from xagent.web.models.background_job import BackgroundJob
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.kb_ingest_targets import admit_kb_ingest_target

test_env = kb_dir.test_env
temp_uploads = kb_dir.temp_uploads

_list_refs = kb_module._list_document_refs_for_uploaded_file
_INGEST = "xagent.web.jobs.kb_tasks.run_document_ingestion"


def _register(seen: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    update_collection_stats_sync(collection_name=kwargs["collection"])
    registered = register_document(
        collection=kwargs["collection"],
        source_path=kwargs["source_path"],
        user_id=kwargs["user_id"],
        file_id=kwargs["file_id"],
        metadata_source_path=kwargs["metadata_source_path"],
    )
    seen.update(file_id=kwargs["file_id"], **registered)
    return registered


def _register_then_raise(seen: dict[str, Any], on_register: Any = None) -> Any:
    def _ingest(**kwargs: Any) -> IngestionResult:
        _register(seen, kwargs)
        if on_register is not None:
            on_register(kwargs)
        raise RuntimeError("binding write failed")

    return _ingest


def _register_and_succeed(seen: dict[str, Any]) -> Any:
    def _ingest(**kwargs: Any) -> IngestionResult:
        registered = _register(seen, kwargs)
        return IngestionResult(
            status="success",
            doc_id=registered["doc_id"],
            completed_steps=[{"name": "register_document", "metadata": registered}],
            message="ok",
        )

    return _ingest


def _raise_before_registration(**_kwargs: Any) -> IngestionResult:
    raise RuntimeError("before registration")


def _submit(
    test_env: Any, content: bytes, *patches: Any, filename: str = "doc.txt"
) -> str:
    app, headers, _user, _ = test_env
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "xagent.web.api.kb._ensure_background_job_queue_available_async",
                new=AsyncMock(),
            )
        )
        stack.enter_context(
            patch(
                "xagent.web.api.kb._enqueue_background_job_or_503_async",
                side_effect=lambda _db, job: job,
            )
        )
        for p in patches:
            stack.enter_context(p)
        response = TestClient(app).post(
            "/api/kb/ingest/jobs",
            files={"file": (filename, content, "text/plain")},
            data={"collection": "coll"},
            headers=headers,
        )
    assert response.status_code == 202, response.text
    return str(response.json()["id"])


def _run(
    test_env: Any, job_id: str, ingest: Any, *, attempts: int = 3, payload: Any = None
) -> Any:
    db = test_env[3]()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).one()
        job.attempts = attempts
        job.max_attempts = 3
        if payload is not None:
            job.payload = payload(dict(job.payload))
        db.commit()
        with patch(_INGEST, side_effect=ingest):
            try:
                return handle_kb_ingest_document(db, job)
            except Exception as exc:  # noqa: BLE001
                return exc
    finally:
        db.close()


def _payload(test_env: Any, job_id: str) -> dict[str, Any]:
    db = test_env[3]()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).one()
        return dict(job.payload)
    finally:
        db.close()


def _file_ids(test_env: Any) -> list[str]:
    db = test_env[3]()
    try:
        return [str(row.file_id) for row in db.query(UploadedFile).all()]
    finally:
        db.close()


def _collection_exists(name: str) -> bool:
    try:
        kb_module.get_collection_sync(name)
    except ValueError:
        return False
    return True


def _assert_original_error(outcome: Any, message: str = "binding write failed") -> None:
    assert isinstance(outcome, RuntimeError)
    assert not isinstance(outcome, BackgroundJobHandlerError)
    assert str(outcome) == message


def _assert_rebuilt_result(outcome: Any, payload: dict[str, Any]) -> None:
    doc_id = generate_deterministic_doc_id("coll", payload["file_id"])
    assert outcome.result["doc_id"] == doc_id
    assert outcome.result["file_id"] == payload["file_id"]
    assert outcome.result["completed_steps"] == [
        {"name": "register_document", "metadata": {"doc_id": doc_id, "created": True}}
    ]


def _publish_first(test_env: Any, filename: str = "doc.txt") -> dict[str, Any]:
    first: dict[str, Any] = {}
    job_id = _submit(test_env, b"first", filename=filename)
    result = _run(test_env, job_id, _register_and_succeed(first), attempts=1)
    assert result["status"] == "success"
    assert first["file_id"] in _file_ids(test_env)
    return first


def test_last_attempt_raise_removes_the_new_document(test_env, temp_uploads) -> None:
    seen: dict[str, Any] = {}
    job_id = _submit(test_env, b"new")
    payload = _payload(test_env, job_id)

    outcome = _run(test_env, job_id, _register_then_raise(seen))

    _assert_original_error(outcome)
    assert seen["created"] is True
    assert _list_refs(seen["file_id"]) == []
    assert _file_ids(test_env) == []
    assert not _collection_exists("coll")
    assert not Path(payload["source_path"]).exists()
    assert payload["document_existed_before"] is False


def test_last_attempt_raise_keeps_other_documents_of_the_collection(
    test_env, temp_uploads
) -> None:
    other = _publish_first(test_env, filename="other.txt")
    seen: dict[str, Any] = {}
    job_id = _submit(test_env, b"new")

    outcome = _run(test_env, job_id, _register_then_raise(seen))

    _assert_original_error(outcome)
    assert _payload(test_env, job_id)["collection_existed_before"] is True
    assert _list_refs(seen["file_id"]) == []
    assert _list_refs(other["file_id"]) == [("coll", other["doc_id"])]
    assert _file_ids(test_env) == [other["file_id"]]
    assert _collection_exists("coll")


@pytest.mark.parametrize("registers_again", [True, False])
def test_earlier_attempts_keep_the_document_for_the_retry(
    test_env, temp_uploads, registers_again
) -> None:
    first_attempt: dict[str, Any] = {}
    last_attempt: dict[str, Any] = {}
    job_id = _submit(test_env, b"new")
    payload = _payload(test_env, job_id)

    outcome = _run(test_env, job_id, _register_then_raise(first_attempt), attempts=1)

    _assert_original_error(outcome)
    assert _list_refs(first_attempt["file_id"]) == [("coll", first_attempt["doc_id"])]
    assert Path(payload["source_path"]).exists()

    if registers_again:
        outcome = _run(test_env, job_id, _register_then_raise(last_attempt))
        _assert_original_error(outcome)
        assert last_attempt["doc_id"] == first_attempt["doc_id"]
        assert last_attempt["created"] is False
    else:
        outcome = _run(test_env, job_id, _raise_before_registration)
        _assert_original_error(outcome, "before registration")

    assert _list_refs(first_attempt["file_id"]) == []
    assert not Path(payload["source_path"]).exists()


def test_last_attempt_raise_keeps_a_document_that_existed_before(
    test_env, temp_uploads
) -> None:
    first = _publish_first(test_env)
    job_id = _submit(test_env, b"second")
    payload = _payload(test_env, job_id)

    outcome = _run(test_env, job_id, _register_then_raise({}))

    _assert_original_error(outcome)
    assert payload["file_id"] == first["file_id"]
    assert _list_refs(first["file_id"]) == [("coll", first["doc_id"])]
    assert _file_ids(test_env) == [first["file_id"]]
    assert Path(payload["target_path"]).read_bytes() == b"first"
    assert payload["document_existed_before"] is True


def test_last_attempt_raise_keeps_a_document_whose_row_is_gone(
    test_env, temp_uploads
) -> None:
    first = _publish_first(test_env)
    db = test_env[3]()
    try:
        db.query(UploadedFile).delete()
        db.commit()
    finally:
        db.close()
    job_id = _submit(test_env, b"second")
    payload = _payload(test_env, job_id)

    outcome = _run(test_env, job_id, _register_then_raise({}))

    _assert_original_error(outcome)
    assert payload["file_id"] == first["file_id"]
    assert _list_refs(first["file_id"]) == [("coll", first["doc_id"])]
    assert payload["document_existed_before"] is True


def test_failed_pre_check_keeps_the_document(test_env, temp_uploads) -> None:
    seen: dict[str, Any] = {}
    job_id = _submit(
        test_env,
        b"new",
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            side_effect=RuntimeError("refs down"),
        ),
    )

    outcome = _run(test_env, job_id, _register_then_raise(seen))

    _assert_original_error(outcome)
    assert _list_refs(seen["file_id"]) == [("coll", seen["doc_id"])]
    assert _payload(test_env, job_id)["document_existed_before"] is True


def test_failed_post_check_still_removes_the_new_document(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}
    job_id = _submit(test_env, b"new")
    real_refs = kb_module._list_document_refs_for_uploaded_file

    def _post_check_down(file_id: str) -> Any:
        raise RuntimeError("refs down")

    with patch(
        "xagent.web.api.kb._list_document_refs_for_uploaded_file",
        side_effect=_post_check_down,
    ):
        outcome = _run(test_env, job_id, _register_then_raise(seen))

    _assert_original_error(outcome)
    assert real_refs(seen["file_id"]) == []


def test_failed_post_check_before_registration_reports_an_incomplete_rollback(
    test_env, temp_uploads
) -> None:
    job_id = _submit(test_env, b"new")
    payload = _payload(test_env, job_id)

    with patch(
        "xagent.web.api.kb._list_document_refs_for_uploaded_file",
        side_effect=RuntimeError("refs down"),
    ):
        outcome = _run(test_env, job_id, _raise_before_registration)

    assert isinstance(outcome, BackgroundJobHandlerError)
    assert outcome.retryable is False
    assert str(outcome).startswith(
        "Failed to fully roll back cloud ingest for coll/doc.txt: delete document "
    )
    assert str(outcome).endswith("Original ingestion error: before registration")
    _assert_rebuilt_result(outcome, payload)
    assert not Path(payload["source_path"]).exists()


def test_missing_user_fails_the_job_and_keeps_the_document(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}
    job_id = _submit(test_env, b"new")
    payload = _payload(test_env, job_id)
    db = test_env[3]()
    try:
        db.query(User).delete()
        db.commit()
    finally:
        db.close()

    outcome = _run(test_env, job_id, _register_then_raise(seen))

    assert isinstance(outcome, BackgroundJobHandlerError)
    assert outcome.retryable is False
    assert str(outcome) == (
        f"Cannot roll back KB ingestion for missing user {payload['user_id']}"
    )
    _assert_rebuilt_result(outcome, payload)
    assert _list_refs(seen["file_id"]) == [("coll", seen["doc_id"])]
    assert not Path(payload["source_path"]).exists()


@pytest.mark.parametrize("table", ["kb_ingest_targets", "users"])
def test_failed_read_before_the_rollback_keeps_the_original_error(
    test_env, temp_uploads, table
) -> None:
    job_id = _submit(test_env, b"new")
    payload = _payload(test_env, job_id)

    def _drop_table(_kwargs: dict[str, Any]) -> None:
        db = test_env[3]()
        try:
            Base.metadata.tables[table].drop(db.get_bind())
        finally:
            db.close()

    outcome = _run(test_env, job_id, _register_then_raise({}, _drop_table))

    _assert_original_error(outcome)
    assert not Path(payload["source_path"]).exists()


def test_raise_before_registration_rolls_back_cleanly(test_env, temp_uploads) -> None:
    job_id = _submit(test_env, b"new")
    payload = _payload(test_env, job_id)

    outcome = _run(test_env, job_id, _raise_before_registration)

    _assert_original_error(outcome, "before registration")
    assert _list_refs(payload["file_id"]) == []
    assert not Path(payload["source_path"]).exists()


def test_superseded_job_leaves_the_document_to_the_newer_upload(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}
    job_id = _submit(test_env, b"new")
    payload = _payload(test_env, job_id)

    def _newer_upload(_kwargs: dict[str, Any]) -> None:
        db = test_env[3]()
        try:
            admit_kb_ingest_target(
                db,
                user_id=int(payload["user_id"]),
                collection=payload["collection"],
                target_path=payload["target_path"],
                file_id=payload["file_id"],
                generation_id=str(uuid.uuid4()),
                job_id="newer",
                file_sha256="0" * 64,
            )
        finally:
            db.close()

    with patch(
        "xagent.web.api.kb._raised_ingestion_rollback_result",
        wraps=kb_module._raised_ingestion_rollback_result,
    ) as rebuild:
        outcome = _run(test_env, job_id, _register_then_raise(seen, _newer_upload))

    _assert_original_error(outcome)
    rebuild.assert_not_awaited()
    assert _list_refs(seen["file_id"]) == [("coll", seen["doc_id"])]
    assert not Path(payload["source_path"]).exists()


def test_job_queued_without_the_stamp_keeps_the_document(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}
    job_id = _submit(test_env, b"new")

    def _unstamped(payload: dict[str, Any]) -> dict[str, Any]:
        payload.pop("document_existed_before", None)
        return payload

    outcome = _run(test_env, job_id, _register_then_raise(seen), payload=_unstamped)

    _assert_original_error(outcome)
    assert _list_refs(seen["file_id"]) == [("coll", seen["doc_id"])]


@pytest.mark.parametrize("attempts", [1, 3])
def test_job_without_file_id_reports_the_original_error(
    test_env, temp_uploads, attempts
) -> None:
    job_id = _submit(test_env, b"new")
    payload = _payload(test_env, job_id)

    def _no_file_id(payload: dict[str, Any]) -> dict[str, Any]:
        payload.pop("file_id")
        return payload

    # Fails the post-raise check, which would count an identity built from None
    # as registered and fail deleting it.
    with patch(
        "xagent.web.api.kb._list_document_refs_for_uploaded_file",
        side_effect=RuntimeError("refs down"),
    ):
        outcome = _run(
            test_env,
            job_id,
            _register_then_raise({}),
            attempts=attempts,
            payload=_no_file_id,
        )

    _assert_original_error(outcome)
    assert Path(payload["source_path"]).exists() is (attempts == 1)


def test_duplicate_submit_skips_the_existence_check(test_env, temp_uploads) -> None:
    check = AsyncMock(return_value=False)
    with patch(
        "xagent.web.api.kb._document_existed_before_ingest_for_file_id", new=check
    ):
        first = _submit(test_env, b"same")
        second = _submit(test_env, b"same")

    assert second == first
    check.assert_awaited_once()


def _embed_then(outcome: Any) -> Any:
    """Ingest that bills one embedding call, then succeeds or raises."""
    from xagent.core.model.chat.token_context import MediaCallType, add_media_usage

    def _ingest(**kwargs: Any) -> Any:
        add_media_usage(call_type=MediaCallType.EMBEDDING, quantity=4, model_id="e")
        return outcome(**kwargs)

    return _ingest


@pytest.mark.parametrize("raises", [False, True], ids=["success", "raised"])
def test_document_job_bills_its_submitter_once(test_env, temp_uploads, raises):
    from xagent.web.services import quota_hooks

    calls: list = []
    quota_hooks.set_standalone_usage_hook(
        lambda user_id, details, source: calls.append((user_id, details, source))
    )
    try:
        job_id = _submit(test_env, b"billed")
        db = test_env[3]()
        try:
            job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).one()
            submitter = int(job.user_id)
            job.attempts, job.max_attempts = 1, 3
            db.commit()
            outcome = (
                _raise_before_registration if raises else _register_and_succeed({})
            )
            with patch(_INGEST, side_effect=_embed_then(outcome)):
                try:
                    handle_kb_ingest_document(db, job)
                except Exception:  # noqa: BLE001
                    assert raises
        finally:
            db.close()
    finally:
        quota_hooks.set_standalone_usage_hook(None)

    [(user_id, [row], source)] = calls
    assert (user_id, source, row["quantity"]) == (submitter, "kb_ingest_job", 4.0)


def test_web_job_bills_its_submitter_not_the_storage_tenant():
    from xagent.core.model.chat.token_context import MediaCallType, add_media_usage
    from xagent.web.jobs import kb_tasks
    from xagent.web.services import quota_hooks

    async def _crawl_bills_then_fails(**kwargs: Any) -> Any:
        assert kwargs["user_id"] == 999  # storage stays with the tenant
        add_media_usage(call_type=MediaCallType.EMBEDDING, quantity=2, model_id="e")
        raise RuntimeError("crawl failed")

    job = BackgroundJob(
        user_id=101,
        payload={
            "user_id": 999,
            "collection": "web_coll",
            "crawl_config": {"start_url": "https://example.com"},
            "ingestion_config": {},
        },
    )
    calls: list = []
    quota_hooks.set_standalone_usage_hook(
        lambda user_id, details, source: calls.append((user_id, details, source))
    )
    try:
        with (
            patch.object(kb_tasks, "update_job_progress"),
            patch.object(kb_tasks, "_cleanup_failed_web_collection_metadata_if_new"),
            patch.object(
                kb_tasks, "run_web_ingestion", side_effect=_crawl_bills_then_fails
            ),
            pytest.raises(RuntimeError, match="crawl failed"),
        ):
            kb_tasks.handle_kb_ingest_web(MagicMock(), job)
    finally:
        quota_hooks.set_standalone_usage_hook(None)

    [(user_id, [row], source)] = calls
    assert (user_id, source, row["quantity"]) == (101, "kb_ingest_web_job", 2.0)


def test_document_job_refused_by_the_gate_rolls_back_and_is_not_retried(
    test_env, temp_uploads
):
    from xagent.web.jobs import kb_tasks
    from xagent.web.services import quota_hooks

    job_id = _submit(test_env, b"refused")
    db = test_env[3]()
    quota_hooks.set_standalone_gate_hook(lambda _user_id, _source: "Out of AI credits")
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).one()
        # A first attempt of three: the refusal must not wait for the last one
        # to clean up, because it is not retried.
        job.attempts, job.max_attempts = 1, 3
        db.commit()
        rollback = kb_tasks._rollback_raised_staged_document_ingestion
        with (
            patch(_INGEST) as ingest,
            patch.object(
                kb_tasks,
                "_rollback_raised_staged_document_ingestion",
                wraps=rollback,
            ) as rolled_back,
            pytest.raises(BackgroundJobHandlerError) as failed,
        ):
            handle_kb_ingest_document(db, job)
    finally:
        quota_hooks.set_standalone_gate_hook(None)
        db.close()

    assert str(failed.value) == "Out of AI credits"
    assert failed.value.retryable is False
    ingest.assert_not_called()
    rolled_back.assert_called_once()


def test_web_job_refused_by_the_gate_fails_without_a_retry():
    from xagent.web.jobs import kb_tasks
    from xagent.web.services import quota_hooks

    job = BackgroundJob(
        user_id=101,
        payload={
            "user_id": 999,
            "collection": "web_coll",
            "crawl_config": {"start_url": "https://example.com"},
            "ingestion_config": {},
        },
    )
    gate_calls: list = []

    def gate(user_id: int, source: str, model_ids: tuple[str, ...]) -> str:
        gate_calls.append((user_id, source, model_ids))
        return "Out of AI credits"

    quota_hooks.set_standalone_gate_hook(gate)
    try:
        with (
            patch.object(
                kb_tasks,
                "resolve_ingestion_model_ids",
                return_value=("platform/text-embedding-v4",),
            ) as resolve,
            patch.object(kb_tasks, "update_job_progress"),
            patch.object(
                kb_tasks, "_cleanup_failed_web_collection_metadata_if_new"
            ) as cleanup,
            patch.object(kb_tasks, "run_web_ingestion") as crawl,
            pytest.raises(BackgroundJobHandlerError) as failed,
        ):
            kb_tasks.handle_kb_ingest_web(MagicMock(), job)
    finally:
        quota_hooks.set_standalone_gate_hook(None)

    # Resolved for the storage tenant, as the crawl resolves; gated for the
    # submitter, who is billed.
    assert resolve.call_args.kwargs["user_id"] == 999
    assert gate_calls == [(101, "kb_ingest_web_job", ("platform/text-embedding-v4",))]
    assert str(failed.value) == "Out of AI credits"
    assert failed.value.retryable is False
    crawl.assert_not_called()
    cleanup.assert_called_once()


def _active_pin() -> Any:
    from xagent.core.tools.core.RAG_tools.utils import model_resolution_pin

    return model_resolution_pin._active_pin.get()


def test_jobs_ingest_under_the_resolution_they_were_gated_on(test_env, temp_uploads):
    from xagent.web.jobs import kb_tasks
    from xagent.web.services import quota_hooks

    pins: list = []

    def resolve(*_args: Any, **_kwargs: Any) -> tuple[str, ...]:
        pins.append(_active_pin())
        return ()

    def ingest(**_kwargs: Any) -> Any:
        pins.append(_active_pin())
        raise RuntimeError("stop")

    async def crawl(**_kwargs: Any) -> Any:
        pins.append(_active_pin())
        raise RuntimeError("stop")

    web_job = BackgroundJob(
        user_id=101,
        payload={
            "user_id": 999,
            "collection": "web_coll",
            "crawl_config": {"start_url": "https://example.com"},
            "ingestion_config": {},
        },
    )
    db = test_env[3]()
    # Resolution is for the gate: with none registered there is none.
    quota_hooks.set_standalone_gate_hook(lambda user_id, source, model_ids: None)
    try:
        job_id = _submit(test_env, b"pinned")
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).one()
        with (
            patch.object(kb_tasks, "resolve_ingestion_model_ids", side_effect=resolve),
            patch(_INGEST, side_effect=ingest),
            patch.object(kb_tasks, "run_web_ingestion", side_effect=crawl),
            patch.object(kb_tasks, "update_job_progress"),
            patch.object(kb_tasks, "_cleanup_failed_web_collection_metadata_if_new"),
        ):
            with pytest.raises(RuntimeError, match="stop"):
                handle_kb_ingest_document(db, job)
            with pytest.raises(RuntimeError, match="stop"):
                kb_tasks.handle_kb_ingest_web(MagicMock(), web_job)
    finally:
        quota_hooks.set_standalone_gate_hook(None)
        db.close()

    document_pins, web_pins = pins[:2], pins[2:]
    assert document_pins[0] is not None and document_pins[0] is document_pins[1]
    assert web_pins[0] is not None and web_pins[0] is web_pins[1]


def test_jobs_resolve_nothing_for_a_gate_that_is_not_there(test_env, temp_uploads):
    from xagent.web.jobs import kb_tasks

    pins: list = []

    def stop(**_kwargs: Any) -> Any:
        pins.append(_active_pin())
        raise RuntimeError("stop")

    async def crawl(**_kwargs: Any) -> Any:
        return stop()

    web_job = BackgroundJob(
        user_id=101,
        payload={
            "user_id": 999,
            "collection": "web_coll",
            "crawl_config": {"start_url": "https://example.com"},
            "ingestion_config": {},
        },
    )
    db = test_env[3]()
    try:
        job_id = _submit(test_env, b"ungated")
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).one()
        with (
            patch.object(kb_tasks, "resolve_ingestion_model_ids") as resolve,
            patch(_INGEST, side_effect=stop),
            patch.object(kb_tasks, "run_web_ingestion", side_effect=crawl),
            patch.object(kb_tasks, "update_job_progress"),
            patch.object(kb_tasks, "_cleanup_failed_web_collection_metadata_if_new"),
        ):
            with pytest.raises(RuntimeError, match="stop"):
                handle_kb_ingest_document(db, job)
            with pytest.raises(RuntimeError, match="stop"):
                kb_tasks.handle_kb_ingest_web(MagicMock(), web_job)
    finally:
        db.close()

    resolve.assert_not_called()
    # And nothing is pinned: the work resolves as it did before gating.
    assert pins == [None, None]
