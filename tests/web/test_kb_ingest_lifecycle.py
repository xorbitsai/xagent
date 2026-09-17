from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    IngestionResult,
    IngestionStepResult,
    WebIngestionResult,
)
from xagent.core.tools.core.RAG_tools.kb import KBApiOperationResult
from xagent.web.api import kb as kb_module
from xagent.web.models.user import User


class _Facade:
    def __init__(self) -> None:
        self.rollback_complete_inputs: list[tuple[KBApiOperationResult[Any], bool]] = []
        self.single_cleanup_inputs: list[
            tuple[KBApiOperationResult[Any], int | None]
        ] = []
        self.batch_cleanup_inputs: list[
            tuple[list[KBApiOperationResult[Any]], int | None]
        ] = []

    def with_rollback_complete(
        self,
        api_result: KBApiOperationResult[Any],
        rollback_complete: bool,
    ) -> KBApiOperationResult[Any]:
        self.rollback_complete_inputs.append((api_result, rollback_complete))
        return KBApiOperationResult(
            result=api_result.result,
            operation_outcome=api_result.operation_outcome,
            rollback_complete=rollback_complete,
        )

    def failed_ingest_cleanup_decision(
        self,
        api_result: KBApiOperationResult[Any],
        *,
        successful_documents: int | None = None,
    ) -> Any:
        self.single_cleanup_inputs.append((api_result, successful_documents))
        return type(
            "Decision",
            (),
            {
                "successful_documents": 3,
                "side_effects_may_remain": api_result.rollback_complete is False,
            },
        )()

    def failed_batch_ingest_cleanup_decision(
        self,
        api_results: list[KBApiOperationResult[Any]],
        *,
        successful_documents: int | None = None,
    ) -> Any:
        self.batch_cleanup_inputs.append((api_results, successful_documents))
        return type(
            "Decision",
            (),
            {
                "successful_documents": 5,
                "side_effects_may_remain": True,
            },
        )()


@pytest.mark.asyncio
async def test_api_failed_ingest_config_cleanup_uses_api_outcome_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = _Facade()
    cleanup_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(kb_module, "_get_api_compatibility_facade", lambda: facade)

    async def fake_cleanup(**kwargs: Any) -> None:
        cleanup_calls.append(kwargs)

    monkeypatch.setattr(
        kb_module,
        "_cleanup_collection_metadata_after_failed_ingest",
        fake_cleanup,
    )
    api_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="failed")
    )
    user = User()
    user.id = 7

    await kb_module._cleanup_collection_metadata_after_failed_api_ingest(
        api_result=api_result,
        collection_existed_before=True,
        collection_name="demo",
        user=user,
        context="ingest",
        successful_documents=1,
        rollback_complete=False,
    )

    assert facade.rollback_complete_inputs == [(api_result, False)]
    # The decision reads the result carrying the rollback outcome, not the input.
    assert len(facade.single_cleanup_inputs) == 1
    decided_on, decided_documents = facade.single_cleanup_inputs[0]
    assert decided_on.rollback_complete is False
    assert decided_documents == 1
    assert cleanup_calls == [
        {
            "collection_existed_before": True,
            "collection_name": "demo",
            "user": user,
            "context": "ingest",
            "successful_documents": 3,
            "side_effects_may_remain": True,
        }
    ]


@pytest.mark.asyncio
async def test_api_failed_batch_ingest_config_cleanup_uses_api_outcome_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = _Facade()
    cleanup_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(kb_module, "_get_api_compatibility_facade", lambda: facade)

    async def fake_cleanup(**kwargs: Any) -> None:
        cleanup_calls.append(kwargs)

    monkeypatch.setattr(
        kb_module,
        "_cleanup_collection_metadata_after_failed_ingest",
        fake_cleanup,
    )
    api_results = [
        KBApiOperationResult(result=IngestionResult(status="success", message="ok")),
        KBApiOperationResult(result=IngestionResult(status="error", message="failed")),
    ]
    user = User()
    user.id = 9

    await kb_module._cleanup_collection_metadata_after_failed_batch_api_ingest(
        api_results=api_results,
        collection_existed_before=False,
        collection_name="demo",
        user=user,
        context="ingest_cloud",
        successful_documents=1,
    )

    assert facade.batch_cleanup_inputs == [(api_results, 1)]
    assert cleanup_calls == [
        {
            "collection_existed_before": False,
            "collection_name": "demo",
            "user": user,
            "context": "ingest_cloud",
            "successful_documents": 5,
            "side_effects_may_remain": True,
        }
    ]


def test_background_failed_ingest_config_cleanup_reuses_api_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.jobs import kb_tasks

    api_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="failed")
    )
    user = User()
    user.id = 11
    db = object()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(kb_tasks, "_get_job_user", lambda *args, **kwargs: user)

    async def fake_api_helper(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        kb_module,
        "_cleanup_collection_metadata_after_failed_api_ingest",
        fake_api_helper,
    )

    kb_tasks._cleanup_failed_job_collection_metadata_after_api_ingest(
        db,  # type: ignore[arg-type]
        {
            "collection": "job-kb",
            "collection_existed_before": False,
        },
        api_result=api_result,
        context="background document ingest",
        successful_documents=2,
        rollback_complete=True,
    )

    assert calls == [
        {
            "api_result": api_result,
            "collection_existed_before": False,
            "collection_name": "job-kb",
            "user": user,
            "context": "background document ingest",
            "successful_documents": 2,
            "rollback_complete": True,
        }
    ]


def test_background_web_cleanup_keeps_early_exception_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.jobs import kb_tasks

    fallback = MagicMock()
    api_helper = MagicMock()
    db = object()
    payload = {"collection": "job-kb"}
    monkeypatch.setattr(
        kb_tasks,
        "_cleanup_failed_job_collection_metadata",
        fallback,
    )
    monkeypatch.setattr(
        kb_tasks,
        "_cleanup_failed_job_collection_metadata_after_api_ingest",
        api_helper,
    )

    kb_tasks._cleanup_failed_web_collection_metadata_if_new(
        db,  # type: ignore[arg-type]
        payload,
    )

    fallback.assert_called_once_with(
        db,
        payload,
        context="background web ingest",
        successful_documents=0,
    )
    api_helper.assert_not_called()


def _records_lookup(records: Any):
    """Stub for kb.list_document_records driving failure-time decisions."""

    def _lookup(**kwargs: Any) -> Any:
        if isinstance(records, Exception):
            raise records
        return records

    return _lookup


@pytest.mark.parametrize(
    ("records", "on_error", "expected"),
    [
        ([{"file_id": "sibling"}], True, True),
        ([], True, False),
        # A store read that fails must not authorize a delete...
        (RuntimeError("vector store down"), True, True),
        # ...nor make a possibly empty knowledge base visible.
        (RuntimeError("vector store down"), False, False),
    ],
)
def test_collection_holds_documents_reads_at_decision_time(
    monkeypatch: pytest.MonkeyPatch,
    records: Any,
    on_error: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(kb_module, "list_document_records", _records_lookup(records))

    assert (
        kb_module._collection_holds_documents(
            collection_name="shared-kb",
            user_id=3,
            context="test",
            on_error=on_error,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("stored_config", "expected"),
    [
        ('{"chunk_size": 512}', True),
        (None, False),
        (RuntimeError("config store down"), True),
    ],
)
@pytest.mark.asyncio
async def test_collection_config_exists_reads_at_decision_time(
    monkeypatch: pytest.MonkeyPatch,
    stored_config: Any,
    expected: bool,
) -> None:
    async def _get_collection_config(**kwargs: Any) -> Any:
        if isinstance(stored_config, Exception):
            raise stored_config
        return stored_config

    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: type(
            "F", (), {"get_collection_config": staticmethod(_get_collection_config)}
        )(),
    )

    assert (
        await kb_module._collection_config_exists(
            collection_name="shared-kb",
            user_id=3,
            context="test",
        )
        is expected
    )


@pytest.mark.asyncio
async def test_rollback_keeps_collection_holding_a_sibling_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale collection_existed_before must not delete a sibling job's work."""
    from pathlib import Path

    delete_collection = MagicMock()
    store = MagicMock()
    # A sibling document: same file_id (both ingests upserted the same path),
    # different doc_id.
    store.list_document_records.return_value = [
        {"file_id": "file-1", "doc_id": "sibling-doc"}
    ]
    monkeypatch.setattr(kb_module, "get_vector_index_store", lambda: store)
    monkeypatch.setattr(kb_module, "delete_collection", delete_collection)
    monkeypatch.setattr(kb_module, "_restore_ingest_file_backup", MagicMock())
    monkeypatch.setattr(kb_module, "clear_ingestion_status", MagicMock())

    user = User()
    user.id = 5
    user.is_admin = False
    file_record = MagicMock()
    file_record.file_id = "file-1"

    await kb_module._rollback_failed_ingestion(
        db=MagicMock(),
        user=user,
        collection_name="shared-kb",
        result=IngestionResult(status="error", message="failed", doc_id="my-doc"),
        file_path=Path("/tmp/does-not-matter.pdf"),
        file_record=file_record,
        collection_existed_before=False,
        uploaded_file_existed_before=True,
        file_backup_path=None,
        had_existing_file=False,
    )

    delete_collection.assert_not_called()


@pytest.mark.asyncio
async def test_rollback_keeps_collection_whose_config_a_sibling_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ingests of the same path share a doc_id, so config is the second guard."""
    from pathlib import Path

    delete_collection = MagicMock()
    store = MagicMock()
    store.list_document_records.return_value = [
        {"file_id": "file-1", "doc_id": "my-doc"}
    ]
    monkeypatch.setattr(kb_module, "get_vector_index_store", lambda: store)
    monkeypatch.setattr(kb_module, "delete_collection", delete_collection)
    monkeypatch.setattr(kb_module, "_restore_ingest_file_backup", MagicMock())
    monkeypatch.setattr(kb_module, "clear_ingestion_status", MagicMock())

    async def _published(**kwargs: Any) -> str:
        return '{"chunk_size": 512}'

    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: type("F", (), {"get_collection_config": staticmethod(_published)})(),
    )

    user = User()
    user.id = 5
    user.is_admin = False
    file_record = MagicMock()
    file_record.file_id = "file-1"

    await kb_module._rollback_failed_ingestion(
        db=MagicMock(),
        user=user,
        collection_name="shared-kb",
        result=IngestionResult(status="error", message="failed", doc_id="my-doc"),
        file_path=Path("/tmp/does-not-matter.pdf"),
        file_record=file_record,
        collection_existed_before=False,
        uploaded_file_existed_before=True,
        file_backup_path=None,
        had_existing_file=False,
    )

    delete_collection.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_keeps_metadata_when_the_collection_holds_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stamped flag is stale; a config-only ghost holds no documents."""
    delete_metadata = AsyncMock()
    monkeypatch.setattr(
        kb_module, "list_document_records", _records_lookup([{"file_id": "sibling"}])
    )
    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: type(
            "F", (), {"delete_collection_metadata": staticmethod(delete_metadata)}
        )(),
    )

    user = User()
    user.id = 5
    user.is_admin = False

    await kb_module._cleanup_failed_new_collection_metadata(
        collection_name="shared-kb",
        user=user,
    )

    delete_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_removes_a_config_only_ghost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collection with no documents is the ghost this cleanup exists for."""
    delete_metadata = AsyncMock(return_value={"config_rows": 1})
    monkeypatch.setattr(kb_module, "list_document_records", _records_lookup([]))
    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: type(
            "F", (), {"delete_collection_metadata": staticmethod(delete_metadata)}
        )(),
    )

    user = User()
    user.id = 5
    user.is_admin = False

    await kb_module._cleanup_failed_new_collection_metadata(
        collection_name="ghost-kb",
        user=user,
    )

    delete_metadata.assert_awaited_once()


def _web_result(documents_created: int) -> WebIngestionResult:
    return WebIngestionResult(
        status="success",
        collection="web-kb",
        total_urls_found=documents_created,
        pages_crawled=documents_created,
        pages_failed=0,
        documents_created=documents_created,
        chunks_created=documents_created,
        embeddings_created=documents_created,
        message="crawl completed",
        elapsed_time_ms=0,
    )


@pytest.mark.parametrize(
    ("collection_existed_before", "expected_outcome"),
    [(False, "the knowledge base was not created"), (True, "nothing was added")],
)
def test_empty_ingest_is_demoted_to_error(
    monkeypatch: pytest.MonkeyPatch,
    collection_existed_before: bool,
    expected_outcome: str,
) -> None:
    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: type(
            "F",
            (),
            {
                "with_result": staticmethod(
                    lambda api_result, result: KBApiOperationResult(
                        result=result,
                        operation_outcome=api_result.operation_outcome,
                        rollback_complete=api_result.rollback_complete,
                    )
                )
            },
        )(),
    )
    api_result = KBApiOperationResult(result=_web_result(0))

    api_result = KBApiOperationResult(result=api_result.result, rollback_complete=False)
    demoted_result = kb_module._demote_empty_crawl_to_error(
        api_result,
        collection_existed_before=collection_existed_before,
    )
    demoted = demoted_result.result

    assert demoted.status == "error"
    assert expected_outcome in demoted.message
    # The cleanup decision reads rollback_complete, so demotion must not drop it.
    assert demoted_result.rollback_complete is False


def test_successful_ingest_is_not_demoted() -> None:
    api_result = KBApiOperationResult(result=_web_result(2))

    assert (
        kb_module._demote_empty_crawl_to_error(
            api_result, collection_existed_before=False
        )
        is api_result
    )


def test_job_config_save_failure_is_not_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying after the documents landed would delete the collection."""
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
    from xagent.web.jobs import kb_tasks
    from xagent.web.jobs.exceptions import BackgroundJobHandlerError

    user = User()
    user.id = 13
    monkeypatch.setattr(kb_tasks, "_get_job_user", lambda *args, **kwargs: user)

    async def fail_save(**kwargs: Any) -> None:
        raise kb_module.CollectionConfigSaveError("config store down")

    monkeypatch.setattr(kb_module, "_save_collection_config_after_ingest", fail_save)

    with pytest.raises(BackgroundJobHandlerError) as excinfo:
        kb_tasks._save_job_collection_config_after_ingest(
            MagicMock(),
            {"collection": "job-kb", "user_id": 13},
            IngestionConfig(),
            context="background document ingest",
            documents_created=1,
        )

    assert excinfo.value.retryable is False


@pytest.mark.parametrize(
    ("existing", "expected_extras"),
    [
        (
            '{"chunk_size": 111, "rerank_model_id": "bge-reranker"}',
            {"rerank_model_id": "bge-reranker"},
        ),
        ('{"chunk_size": 111}', {}),
        (None, {}),
        ("not json", {}),
    ],
)
@pytest.mark.asyncio
async def test_config_merge_keeps_settings_this_ingest_does_not_own(
    monkeypatch: pytest.MonkeyPatch,
    existing: Any,
    expected_extras: dict[str, Any],
) -> None:
    """A rerank binding saved while the ingest ran must survive the config write."""

    async def _get_collection_config(**kwargs: Any) -> Any:
        return existing

    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: type(
            "F", (), {"get_collection_config": staticmethod(_get_collection_config)}
        )(),
    )

    merged = json.loads(
        await kb_module._config_json_preserving_extras(
            collection="kb",
            config_json='{"chunk_size": 2048}',
            user_id=1,
            context="test",
            collection_existed_before=True,
        )
    )

    assert merged["chunk_size"] == 2048
    for key, value in expected_extras.items():
        assert merged[key] == value
    if not expected_extras:
        assert set(merged) == {"chunk_size"}


@pytest.mark.asyncio
async def test_config_save_failure_does_not_claim_the_import_must_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documents are listed, so re-importing would only duplicate them."""

    async def _failing_save(**kwargs: Any) -> None:
        raise RuntimeError("config store down")

    async def _no_existing_config(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: type(
            "F",
            (),
            {
                "save_collection_config": staticmethod(_failing_save),
                "get_collection_config": staticmethod(_no_existing_config),
            },
        )(),
    )

    user = User()
    user.id = 5
    user.is_admin = False

    with pytest.raises(kb_module.CollectionConfigSaveError) as excinfo:
        await kb_module._save_collection_config_after_ingest(
            collection="q3",
            config_json='{"chunk_size": 2048}',
            user=user,
            context="ingest",
            documents_created=1,
        )

    message = str(excinfo.value)
    assert "Do not re-import" in message
    assert "stays hidden" not in message


@pytest.mark.parametrize(
    ("collection_existed_before", "should_publish"),
    [(False, True), (True, False)],
)
@pytest.mark.asyncio
async def test_documents_left_by_an_incomplete_rollback_are_published(
    monkeypatch: pytest.MonkeyPatch,
    collection_existed_before: bool,
    should_publish: bool,
) -> None:
    """Documents with no config row are invisible and block the name.

    A pre-existing collection is exempt: republishing would overwrite settings
    its previous import saved.
    """
    save = AsyncMock()
    monkeypatch.setattr(
        kb_module, "list_document_records", _records_lookup([{"file_id": "leftover"}])
    )

    async def _no_existing_config(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: type(
            "F",
            (),
            {
                "save_collection_config": staticmethod(save),
                "get_collection_config": staticmethod(_no_existing_config),
            },
        )(),
    )

    user = User()
    user.id = 5
    user.is_admin = False

    await kb_module._save_collection_config_after_ingest(
        collection="kb",
        config_json='{"chunk_size": 2048}',
        user=user,
        context="ingest_web",
        documents_created=0,
        collection_existed_before=collection_existed_before,
    )

    assert save.await_count == (1 if should_publish else 0)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [({}, True), ({"collection_existed_before": False}, False)],
)
def test_missing_flag_defaults_to_pre_existing(
    payload: dict[str, Any], expected: bool
) -> None:
    """An absent flag must not authorize cleanup of someone else's collection."""
    from xagent.web.jobs import kb_tasks

    assert kb_tasks._collection_existed_before(payload) is expected


@pytest.mark.parametrize(
    ("status", "registered", "expected"),
    [
        # A registered document is listed and deletable even with no chunks.
        ("success", True, 1),
        ("partial", True, 1),
        # Nothing was registered, so nothing landed.
        ("success", False, 0),
        ("error", True, 0),
    ],
)
def test_produced_documents_is_the_shared_publish_predicate(
    status: str, registered: bool, expected: int
) -> None:
    result = IngestionResult(
        status=status,
        message="m",
        completed_steps=[IngestionStepResult(name="register_document", metadata={})]
        if registered
        else [],
    )

    assert result.produced_documents == expected


def _facade_with(**attrs: Any) -> Any:
    return type("F", (), {k: staticmethod(v) for k, v in attrs.items()})()


@pytest.mark.asyncio
async def test_metadata_cleanup_deletes_with_the_scope_it_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admin-scoped delete would wipe a row another tenant owns by name."""
    delete_metadata = AsyncMock(return_value={"config_rows": 1})
    monkeypatch.setattr(kb_module, "list_document_records", _records_lookup([]))
    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: _facade_with(delete_collection_metadata=delete_metadata),
    )

    user = User()
    user.id = 5
    user.is_admin = True

    await kb_module._cleanup_failed_new_collection_metadata(
        collection_name="shared-kb",
        user=user,
    )

    assert delete_metadata.await_args.kwargs["is_admin"] is False


@pytest.mark.asyncio
async def test_publish_does_not_guess_when_the_store_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable store must not make a possibly empty KB visible."""
    save = AsyncMock()
    monkeypatch.setattr(
        kb_module,
        "list_document_records",
        _records_lookup(RuntimeError("vector store down")),
    )
    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: _facade_with(save_collection_config=save),
    )

    user = User()
    user.id = 5
    user.is_admin = False

    await kb_module._save_collection_config_after_ingest(
        collection="kb",
        config_json="{}",
        user=user,
        context="ingest_web",
        documents_created=0,
        collection_existed_before=False,
    )

    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_does_not_delete_when_the_store_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opposite direction: an unreadable store must not authorize a delete."""
    delete_metadata = AsyncMock()
    monkeypatch.setattr(
        kb_module,
        "list_document_records",
        _records_lookup(RuntimeError("vector store down")),
    )
    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: _facade_with(delete_collection_metadata=delete_metadata),
    )

    user = User()
    user.id = 5
    user.is_admin = False

    await kb_module._cleanup_failed_new_collection_metadata(
        collection_name="kb",
        user=user,
    )

    delete_metadata.assert_not_awaited()


@pytest.mark.parametrize(
    ("existed_before", "other_document", "config", "may_delete"),
    [
        (False, False, None, True),
        # a sibling landed a document, or published the config
        (False, True, None, False),
        (False, False, "{}", False),
        (True, False, None, False),
    ],
)
@pytest.mark.asyncio
async def test_rollback_may_delete_collection_reads_live_state(
    monkeypatch: pytest.MonkeyPatch,
    existed_before: bool,
    other_document: bool,
    config: Any,
    may_delete: bool,
) -> None:
    async def _get_collection_config(**kwargs: Any) -> Any:
        return config

    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: _facade_with(get_collection_config=_get_collection_config),
    )

    assert (
        await kb_module._rollback_may_delete_collection(
            collection_name="kb",
            user_id=5,
            collection_existed_before=existed_before,
            other_document_present=other_document,
            context="test",
        )
        is may_delete
    )


@pytest.mark.asyncio
async def test_rollback_keeps_the_collection_when_the_config_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown state must not authorize deleting a whole collection."""

    async def _get_collection_config(**kwargs: Any) -> Any:
        raise RuntimeError("metadata store down")

    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: _facade_with(get_collection_config=_get_collection_config),
    )

    assert (
        await kb_module._rollback_may_delete_collection(
            collection_name="kb",
            user_id=5,
            collection_existed_before=False,
            other_document_present=False,
            context="test",
        )
        is False
    )


def test_job_config_save_failure_keeps_the_result_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FAILED job with no payload is indistinguishable from a real failure."""
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
    from xagent.web.jobs import kb_tasks
    from xagent.web.jobs.exceptions import BackgroundJobHandlerError

    user = User()
    user.id = 13
    monkeypatch.setattr(kb_tasks, "_get_job_user", lambda *args, **kwargs: user)

    async def fail_save(**kwargs: Any) -> None:
        raise kb_module.CollectionConfigSaveError("config store down")

    monkeypatch.setattr(kb_module, "_save_collection_config_after_ingest", fail_save)

    with pytest.raises(BackgroundJobHandlerError) as excinfo:
        kb_tasks._save_job_collection_config_after_ingest(
            MagicMock(),
            {"collection": "job-kb", "user_id": 13},
            IngestionConfig(),
            context="background document ingest",
            documents_created=1,
            result_payload={"file_id": "file-1", "status": "success"},
        )

    assert excinfo.value.retryable is False
    assert excinfo.value.result == {"file_id": "file-1", "status": "success"}


def test_job_with_a_missing_user_fails_instead_of_succeeding_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning here would mark the job SUCCEEDED with nothing published."""
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
    from xagent.web.jobs import kb_tasks
    from xagent.web.jobs.exceptions import BackgroundJobHandlerError

    monkeypatch.setattr(kb_tasks, "_get_job_user", lambda *args, **kwargs: None)

    with pytest.raises(BackgroundJobHandlerError) as excinfo:
        kb_tasks._save_job_collection_config_after_ingest(
            MagicMock(),
            {"collection": "job-kb", "user_id": 13},
            IngestionConfig(),
            context="background document ingest",
            documents_created=1,
            result_payload={"file_id": "file-1"},
        )

    assert excinfo.value.retryable is False
    assert excinfo.value.result == {"file_id": "file-1"}


@pytest.mark.asyncio
async def test_unreadable_config_does_not_overwrite_an_existing_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The write replaces the row wholesale, so writing blind would drop settings."""
    save = AsyncMock()

    async def _unreadable(**kwargs: Any) -> Any:
        raise RuntimeError("config store down")

    monkeypatch.setattr(kb_module, "list_document_records", _records_lookup([{"d": 1}]))
    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: _facade_with(
            save_collection_config=save, get_collection_config=_unreadable
        ),
    )

    user = User()
    user.id = 5
    user.is_admin = False

    await kb_module._save_collection_config_after_ingest(
        collection="kb",
        config_json='{"chunk_size": 2048}',
        user=user,
        context="ingest",
        documents_created=1,
        collection_existed_before=True,
    )

    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_unreadable_config_still_publishes_a_new_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brand-new collection has no settings to lose."""
    save = AsyncMock()

    async def _unreadable(**kwargs: Any) -> Any:
        raise RuntimeError("config store down")

    monkeypatch.setattr(kb_module, "list_document_records", _records_lookup([{"d": 1}]))
    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: _facade_with(
            save_collection_config=save, get_collection_config=_unreadable
        ),
    )

    user = User()
    user.id = 5
    user.is_admin = False

    await kb_module._save_collection_config_after_ingest(
        collection="kb",
        config_json='{"chunk_size": 2048}',
        user=user,
        context="ingest",
        documents_created=1,
        collection_existed_before=False,
    )

    save.assert_awaited_once()


def test_missing_user_is_ignored_when_there_is_nothing_to_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raising here would skip the caller's failure cleanup."""
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
    from xagent.web.jobs import kb_tasks

    get_user = MagicMock(return_value=None)
    monkeypatch.setattr(kb_tasks, "_get_job_user", get_user)

    kb_tasks._save_job_collection_config_after_ingest(
        MagicMock(),
        {"collection": "job-kb", "user_id": 13, "collection_existed_before": True},
        IngestionConfig(),
        context="background web ingest",
        documents_created=0,
    )

    get_user.assert_not_called()


def test_missing_user_on_a_failed_new_collection_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller's metadata cleanup runs after this; raising would orphan the row."""
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
    from xagent.web.jobs import kb_tasks

    monkeypatch.setattr(kb_tasks, "_get_job_user", lambda *args, **kwargs: None)

    kb_tasks._save_job_collection_config_after_ingest(
        MagicMock(),
        {"collection": "job-kb", "user_id": 13, "collection_existed_before": False},
        IngestionConfig(),
        context="background web ingest",
        documents_created=0,
    )


@pytest.mark.parametrize(
    ("existed_before", "expected"),
    [(True, "knowledge base settings"), (False, "not listed yet")],
)
@pytest.mark.asyncio
async def test_config_save_failure_advice_matches_what_the_user_can_do(
    monkeypatch: pytest.MonkeyPatch,
    existed_before: bool,
    expected: str,
) -> None:
    """A collection that was never published has no settings page to visit."""

    async def _failing_save(**kwargs: Any) -> None:
        raise RuntimeError("config store down")

    async def _no_existing_config(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(kb_module, "list_document_records", _records_lookup([{"d": 1}]))
    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: _facade_with(
            save_collection_config=_failing_save,
            get_collection_config=_no_existing_config,
        ),
    )

    user = User()
    user.id = 5
    user.is_admin = False

    with pytest.raises(kb_module.CollectionConfigSaveError) as excinfo:
        await kb_module._save_collection_config_after_ingest(
            collection="q3",
            config_json='{"chunk_size": 2048}',
            user=user,
            context="ingest",
            documents_created=1,
            collection_existed_before=existed_before,
        )

    assert expected in str(excinfo.value)
