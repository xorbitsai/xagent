"""Unit tests for KB API exception decorator.

These tests cover the mapping behavior of `handle_kb_exceptions` in `xagent.web.api.kb`.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from xagent.core.tools.core.RAG_tools.core.exceptions import CollectionMigratingError
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionOperationResult
from xagent.core.tools.core.RAG_tools.kb.api_compatibility import KBApiOperationResult
from xagent.core.tools.core.RAG_tools.utils.user_scope import get_user_scope
from xagent.web.api.kb import handle_kb_exceptions, with_kb_user_scope


@pytest.mark.asyncio
async def test_handle_kb_exceptions_passthrough_http_exception() -> None:
    """HTTPException should be re-raised without being wrapped."""

    @handle_kb_exceptions
    async def _fn() -> None:
        raise HTTPException(status_code=418, detail="teapot")

    with pytest.raises(HTTPException) as exc_info:
        await _fn()

    assert exc_info.value.status_code == 418
    assert exc_info.value.detail == "teapot"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "expected_status", "expected_prefix"),
    [
        (ValueError("bad"), 400, "Data format error:"),
        (KeyError("missing"), 400, "Data format error:"),
        (TypeError("wrong type"), 400, "Data format error:"),
    ],
)
async def test_handle_kb_exceptions_maps_data_errors_to_400(
    exc: Exception, expected_status: int, expected_prefix: str
) -> None:
    """ValueError/KeyError/TypeError should map to 400 with a data error message."""

    @handle_kb_exceptions
    async def _fn() -> None:
        raise exc

    with pytest.raises(HTTPException) as exc_info:
        await _fn()

    assert exc_info.value.status_code == expected_status
    assert isinstance(exc_info.value.detail, str)
    assert exc_info.value.detail.startswith(expected_prefix)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "expected_status", "expected_prefix"),
    [
        (PermissionError("nope"), 403, "File system error:"),
        (OSError("io"), 403, "File system error:"),
    ],
)
async def test_handle_kb_exceptions_maps_fs_errors_to_403(
    exc: Exception, expected_status: int, expected_prefix: str
) -> None:
    """PermissionError/OSError should map to 403 with a file system error message."""

    @handle_kb_exceptions
    async def _fn() -> None:
        raise exc

    with pytest.raises(HTTPException) as exc_info:
        await _fn()

    assert exc_info.value.status_code == expected_status
    assert isinstance(exc_info.value.detail, str)
    assert exc_info.value.detail.startswith(expected_prefix)


@pytest.mark.asyncio
async def test_handle_kb_exceptions_maps_unknown_errors_to_500() -> None:
    """Other exceptions should map to 500 with an internal error message."""

    class _Boom(RuntimeError):
        pass

    @handle_kb_exceptions
    async def _fn() -> None:
        raise _Boom("boom")

    with pytest.raises(HTTPException) as exc_info:
        await _fn()

    assert exc_info.value.status_code == 500
    assert isinstance(exc_info.value.detail, str)
    assert exc_info.value.detail.startswith("服务器内部错误:")


# --- delete_collection_api metadata delegation Tests ---

_FACTORY_PATH = "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store"


@pytest.mark.asyncio
async def test_delete_collection_api_does_not_call_metadata_store_on_success() -> None:
    """API layer should delegate metadata cleanup to management layer."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from xagent.web.api.kb import delete_collection_api
    from xagent.web.models.user import User

    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = True
    mock_db = MagicMock()

    mock_metadata_store = MagicMock()
    mock_metadata_store.delete_collection = AsyncMock()

    _fake_delete_result = MagicMock()
    _fake_delete_result.status = "success"
    _fake_delete_result.affected_documents = 1
    _fake_delete_result.deleted_counts = {"documents": 1}

    with (
        patch("xagent.web.api.kb.delete_collection_physical_dir"),
        patch(
            "xagent.web.api.kb.get_vector_index_store",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_collection",
            return_value=_fake_delete_result,
        ),
        patch(
            "xagent.web.api.kb.delete_collection_uploaded_files",
            return_value=None,
        ),
        patch(
            _FACTORY_PATH,
            return_value=mock_metadata_store,
        ),
    ):
        await delete_collection_api(
            collection_name="test_collection_abc",
            _user=mock_user,
            db=mock_db,
        )

    mock_metadata_store.delete_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_collection_api_success_not_affected_by_metadata_store_mock() -> (
    None
):
    """Even if metadata store is mocked to fail, API success path should not touch it."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from xagent.web.api.kb import delete_collection_api
    from xagent.web.models.user import User

    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = True
    mock_db = MagicMock()

    mock_metadata_store = MagicMock()
    mock_metadata_store.delete_collection = AsyncMock(
        side_effect=RuntimeError("metadata store unreachable")
    )

    _fake_delete_result = MagicMock()
    _fake_delete_result.status = "success"
    _fake_delete_result.affected_documents = 1
    _fake_delete_result.deleted_counts = {"documents": 1}

    with (
        patch("xagent.web.api.kb.delete_collection_physical_dir"),
        patch(
            "xagent.web.api.kb.get_vector_index_store",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_collection",
            return_value=_fake_delete_result,
        ),
        patch(
            "xagent.web.api.kb.delete_collection_uploaded_files",
            return_value=None,
        ),
        patch(
            _FACTORY_PATH,
            return_value=mock_metadata_store,
        ),
    ):
        result = await delete_collection_api(
            collection_name="test_collection",
            _user=mock_user,
            db=mock_db,
        )

    # Should not raise — error is logged but swallowed
    assert result.status == "success"
    mock_metadata_store.delete_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_collection_api_failed_no_metadata_cleanup() -> None:
    """If deletion fails (not success/partial_success), metadata should NOT be cleaned."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from xagent.web.api.kb import delete_collection_api
    from xagent.web.models.user import User

    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = True
    mock_db = MagicMock()

    mock_metadata_store = MagicMock()
    mock_metadata_store.delete_collection = AsyncMock()

    class _FakeErrorResult:
        status = "error"
        affected_documents = 0
        deleted_counts = {}
        physical_cleanup = {}
        uploaded_file_cleanup = None
        uploads_cleanup = {}

    with (
        patch(
            "xagent.web.api.kb._perform_kb_collection_delete",
            return_value=_FakeErrorResult(),
        ),
        patch(
            _FACTORY_PATH,
            return_value=mock_metadata_store,
        ),
    ):
        await delete_collection_api(
            collection_name="test_collection",
            _user=mock_user,
            db=mock_db,
        )

    mock_metadata_store.delete_collection.assert_not_awaited()
    mock_db.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_delete_collection_hook_failure_prevents_physical_delete() -> None:
    from unittest.mock import MagicMock, patch

    from xagent.web.api.kb import delete_collection_api
    from xagent.web.models.user import User

    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False
    mock_db = MagicMock()

    with (
        patch(
            "xagent.web.api.kb.notify_knowledge_base_deleted",
            side_effect=RuntimeError("hook failed"),
        ),
        patch("xagent.web.api.kb._perform_kb_collection_delete") as physical_delete,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_collection_api(
                collection_name="team-kb", _user=mock_user, db=mock_db
            )

    assert exc_info.value.status_code == 500
    physical_delete.assert_not_called()


@pytest.mark.asyncio
async def test_batch_delete_continues_after_hook_failure_without_double_counting() -> (
    None
):
    from unittest.mock import MagicMock, patch

    from xagent.web.api.kb import (
        BatchDeleteCollectionsRequest,
        batch_delete_collections_api,
    )
    from xagent.web.models.user import User

    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False
    mock_db = MagicMock()
    success = MagicMock(status="success")

    with (
        patch(
            "xagent.web.api.kb.notify_knowledge_base_deleted",
            side_effect=[RuntimeError("hook failed"), None],
        ),
        patch(
            "xagent.web.api.kb._perform_kb_collection_delete",
            return_value=success,
        ) as physical_delete,
    ):
        result = await batch_delete_collections_api(
            body=BatchDeleteCollectionsRequest(collection_names=["first", "second"]),
            _user=mock_user,
            db=mock_db,
        )

    assert result.deleted == ["second"]
    assert [item.name for item in result.failed] == ["first"]
    physical_delete.assert_called_once()
    assert physical_delete.call_args.args[0] == "second"


def test_effective_kb_user_enforces_hook_permissions() -> None:
    from unittest.mock import MagicMock, patch

    from xagent.web.api.kb import _effective_knowledge_base_user
    from xagent.web.models.user import User
    from xagent.web.services.knowledge_base_team_scope import KnowledgeBaseAccess

    actor = MagicMock(spec=User)
    actor.id = 1
    db = MagicMock()

    with patch(
        "xagent.web.api.kb.resolve_knowledge_base_access",
        return_value=KnowledgeBaseAccess(
            name="team-kb",
            storage_user_id=2,
            team_owned=True,
            can_edit=False,
            can_delete=False,
        ),
    ):
        with pytest.raises(HTTPException) as edit_exc:
            _effective_knowledge_base_user(db, actor, "team-kb", action="edit")
        with pytest.raises(HTTPException) as delete_exc:
            _effective_knowledge_base_user(db, actor, "team-kb", action="delete")

    assert edit_exc.value.status_code == 403
    assert delete_exc.value.status_code == 403


@pytest.mark.asyncio
async def test_with_kb_user_scope_ignores_positional_duck_typed_objects() -> None:
    """Decorator should not infer user scope from arbitrary positional objects."""

    class _Imposter:
        id = 99
        is_admin = True

    @with_kb_user_scope
    async def _fn(_arg: object) -> tuple[int | None, bool]:
        scope = get_user_scope()
        return scope.user_id, scope.is_admin

    user_id, is_admin = await _fn(_Imposter())

    assert user_id is None
    assert is_admin is False


# --- Embedding-migration freeze mapping ---

_MIGRATION_DETAIL = {
    "code": "kb_collection_migrating",
    "message": "frozen",
}


def _migration_error() -> CollectionMigratingError:
    return CollectionMigratingError("frozen")


def _freeze_state(collection: str = "kb") -> None:
    from xagent.core.tools.core.RAG_tools.management.migration_state import (
        FROZEN,
        CollectionMigrationState,
        write_migration_state,
    )

    write_migration_state(collection, CollectionMigrationState(state=FROZEN))


def _migration_mocks() -> tuple[object, object, object]:
    from unittest.mock import MagicMock

    from xagent.web.models.user import User

    user = MagicMock(spec=User)
    user.id = 1
    user.is_admin = True
    db = MagicMock()
    access = MagicMock()
    access.storage_user_id = 1
    return user, db, access


@pytest.mark.asyncio
async def test_handle_kb_exceptions_maps_migration_freeze_to_409() -> None:
    exc = _migration_error()

    @handle_kb_exceptions
    async def _fn() -> None:
        raise exc

    with pytest.raises(HTTPException) as exc_info:
        await _fn()

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == _MIGRATION_DETAIL


@pytest.mark.asyncio
async def test_delete_collection_route_maps_migration_freeze_to_409() -> None:
    from unittest.mock import AsyncMock, patch

    from xagent.web.api.kb import delete_collection_api

    user, db, access = _migration_mocks()
    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.migration_state.ensure_collection_writable_async",
            new=AsyncMock(side_effect=_migration_error()),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_collection_api(collection_name="kb", _user=user, db=db)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == _MIGRATION_DETAIL


@pytest.mark.asyncio
async def test_batch_delete_frozen_collection_is_per_item_failure_and_continues() -> (
    None
):
    from unittest.mock import MagicMock, patch

    from xagent.web.api.kb import (
        BatchDeleteCollectionsRequest,
        batch_delete_collections_api,
    )

    user, db, access = _migration_mocks()

    async def _guard(name: str) -> None:
        if name == "frozen":
            raise _migration_error()

    success = MagicMock(status="success", message="")

    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.migration_state.ensure_collection_writable_async",
            new=_guard,
        ),
        patch(
            "xagent.web.api.kb._perform_kb_collection_delete",
            return_value=success,
        ) as delete_one,
    ):
        result = await batch_delete_collections_api(
            body=BatchDeleteCollectionsRequest(collection_names=["frozen", "ok"]),
            _user=user,
            db=db,
        )

    assert result.deleted == ["ok"]
    assert [item.name for item in result.failed] == ["frozen"]
    assert result.failed[0].error == "frozen"
    delete_one.assert_called_once()
    assert delete_one.call_args.args[0] == "ok"


@pytest.mark.asyncio
async def test_rename_route_refuses_before_physical_rename() -> None:
    from unittest.mock import AsyncMock, patch

    from xagent.web.api.kb import rename_collection_api

    user, db, access = _migration_mocks()
    _freeze_state("kb")
    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new=AsyncMock()),
        patch("xagent.web.api.kb.rename_collection_storage") as physical_rename,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await rename_collection_api(
                collection_name="kb",
                new_name="kb-new",
                _user=user,
                db=db,
            )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "kb_collection_migrating"
    assert "being upgraded" in exc_info.value.detail["message"]
    physical_rename.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_route_maps_migration_freeze_to_409() -> None:
    import io
    from unittest.mock import AsyncMock, patch

    from fastapi import UploadFile

    from xagent.web.api.kb import ingest

    user, db, access = _migration_mocks()
    upload = UploadFile(filename="doc.pdf", file=io.BytesIO(b"content"))
    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new=AsyncMock()),
        patch(
            "xagent.core.tools.core.RAG_tools.management.migration_state.ensure_collection_writable_async",
            new=AsyncMock(side_effect=_migration_error()),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await ingest(
                collection="kb",
                file=upload,
                parse_method=None,
                _user=user,
                db=db,
            )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == _MIGRATION_DETAIL


@pytest.mark.asyncio
async def test_ingest_cloud_route_maps_migration_freeze_to_409() -> None:
    from unittest.mock import AsyncMock, patch

    from xagent.web.api.kb import CloudFile, CloudIngestRequest, ingest_cloud

    user, db, access = _migration_mocks()
    request = CloudIngestRequest(
        files=[CloudFile(provider="google_drive", fileId="f1", fileName="a.pdf")],
        collection="kb",
    )
    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch("xagent.web.api.kb._enforce_storage_gate"),
        patch("xagent.web.api.kb._ensure_collection_access", new=AsyncMock()),
        patch(
            "xagent.core.tools.core.RAG_tools.management.migration_state.ensure_collection_writable_async",
            new=AsyncMock(side_effect=_migration_error()),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await ingest_cloud(request=request, _user=user, db=db)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == _MIGRATION_DETAIL


@pytest.mark.asyncio
async def test_ingest_web_route_maps_migration_freeze_to_409() -> None:
    from unittest.mock import AsyncMock, patch

    from xagent.web.api.kb import ingest_web

    user, db, access = _migration_mocks()
    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new=AsyncMock()),
        patch(
            "xagent.core.tools.core.RAG_tools.management.migration_state.ensure_collection_writable_async",
            new=AsyncMock(side_effect=_migration_error()),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await ingest_web(
                collection="kb",
                start_url="https://example.com",
                _user=user,
                db=db,
            )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == _MIGRATION_DETAIL


@pytest.mark.asyncio
async def test_create_ingest_job_maps_migration_freeze_to_409() -> None:
    import io
    from unittest.mock import AsyncMock, patch

    from fastapi import UploadFile

    from xagent.web.api.kb import create_ingest_job

    user, db, access = _migration_mocks()
    upload = UploadFile(filename="doc.pdf", file=io.BytesIO(b"content"))
    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new=AsyncMock()),
        patch(
            "xagent.web.api.kb._ensure_background_job_queue_available_async",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.kb.ensure_collection_writable_async",
            new=AsyncMock(side_effect=_migration_error()),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_ingest_job(
                collection="kb",
                file=upload,
                parse_method=None,
                _user=user,
                db=db,
            )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == _MIGRATION_DETAIL


@pytest.mark.asyncio
async def test_delete_document_route_maps_migration_freeze_to_409() -> None:
    from unittest.mock import AsyncMock, patch

    from xagent.web.api.kb import delete_document_api

    user, db, access = _migration_mocks()
    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch(
            "xagent.web.api.kb._ensure_collection_access_for_document_delete",
            new=AsyncMock(),
        ),
        patch("xagent.web.api.kb.get_vector_index_store") as vector_store,
        patch(
            "xagent.core.tools.core.RAG_tools.management.migration_state.ensure_collection_writable_async",
            new=AsyncMock(side_effect=_migration_error()),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_document_api(
                collection_name="kb",
                filename="doc.pdf",
                _user=user,
                db=db,
            )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == _MIGRATION_DETAIL
    vector_store.assert_not_called()


@pytest.mark.asyncio
async def test_delete_route_admission_covers_the_core_guard() -> None:
    """A freeze after the route admission must not refuse the delete half-way."""
    from unittest.mock import patch

    from xagent.core.tools.core.RAG_tools.management import (
        migration_state as state_module,
    )
    from xagent.web.api.kb import delete_collection_api

    user, db, access = _migration_mocks()
    success = CollectionOperationResult(collection="kb", status="success", message="")

    def _perform(name: str, *args: object) -> object:
        # Freeze lands after the route admitted, before the core guard.
        _freeze_state(name)
        state_module.ensure_collection_writable(name)
        return success

    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch("xagent.web.api.kb.notify_knowledge_base_deleted"),
        patch(
            "xagent.web.api.kb._perform_kb_collection_delete",
            side_effect=_perform,
        ),
    ):
        result = await delete_collection_api(collection_name="kb", _user=user, db=db)

    assert result is success


@pytest.mark.asyncio
async def test_batch_delete_route_admission_covers_the_core_guard() -> None:
    from unittest.mock import patch

    from xagent.core.tools.core.RAG_tools.management import (
        migration_state as state_module,
    )
    from xagent.web.api.kb import (
        BatchDeleteCollectionsRequest,
        batch_delete_collections_api,
    )

    user, db, access = _migration_mocks()
    success = CollectionOperationResult(collection="kb", status="success", message="")

    def _perform(name: str, *args: object) -> object:
        _freeze_state(name)
        state_module.ensure_collection_writable(name)
        return success

    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch("xagent.web.api.kb.notify_knowledge_base_deleted"),
        patch(
            "xagent.web.api.kb._perform_kb_collection_delete",
            side_effect=_perform,
        ) as delete_one,
    ):
        result = await batch_delete_collections_api(
            body=BatchDeleteCollectionsRequest(collection_names=["a", "b"]),
            _user=user,
            db=db,
        )

    assert result.deleted == ["a", "b"]
    assert result.failed == []
    assert [call.args[0] for call in delete_one.call_args_list] == ["a", "b"]


@pytest.mark.asyncio
async def test_rename_route_admission_covers_the_coordinator_leg() -> None:
    """The coordinator must still rename everything when a freeze lands mid-route."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from xagent.core.tools.core.RAG_tools.management import (
        migration_state as state_module,
    )
    from xagent.web.api.kb import rename_collection_api

    user, db, access = _migration_mocks()
    renamed = {"vector": False, "status": False, "metadata": False}

    class _Facade:
        async def rename_collection_data(self, **kwargs: object) -> list[str]:
            # Route precheck saw idle; the freeze lands before this core leg.
            _freeze_state("kb")
            await state_module.ensure_collection_writable_async("kb")
            renamed["vector"] = True
            return []

        async def rename_collection_metadata(self, **kwargs: object) -> None:
            renamed["metadata"] = True

        def rename_collection_status(self, **kwargs: object) -> list[str]:
            renamed["status"] = True
            return []

    physical = MagicMock(
        status="not_found",
        error=None,
        old_collection_dir=None,
        new_collection_dir=None,
    )
    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new=AsyncMock()),
        patch(
            "xagent.web.api.kb._list_collections_with_retry",
            new=AsyncMock(return_value=SimpleNamespace(collections=[])),
        ),
        patch(
            "xagent.web.api.kb._resolve_collection_mutation_scope",
            return_value=SimpleNamespace(
                owner_user_ids=[1], file_ids_by_owner={1: set()}
            ),
        ),
        patch(
            "xagent.web.api.kb.get_upload_path",
            return_value=MagicMock(exists=lambda: False),
        ),
        patch(
            "xagent.web.api.kb.rename_collection_storage", return_value=physical
        ) as physical_rename,
        patch(
            "xagent.web.api.kb._get_api_compatibility_facade", return_value=_Facade()
        ),
        patch("xagent.web.api.kb.notify_knowledge_base_renamed"),
    ):
        result = await rename_collection_api(
            collection_name="kb", new_name="kb-new", _user=user, db=db
        )

    physical_rename.assert_called_once()
    assert renamed == {"vector": True, "status": True, "metadata": True}
    assert result["status"] == "partial_success"


@pytest.mark.asyncio
async def test_delete_helper_lets_migration_refusal_through() -> None:
    """A direct core refusal must stay CollectionMigratingError, not become a 500."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from xagent.web.api.kb import _perform_kb_collection_delete

    _freeze_state("kb")
    with patch(
        "xagent.web.api.kb._resolve_collection_mutation_scope",
        return_value=SimpleNamespace(owner_user_ids=[], file_ids_by_owner={}),
    ):
        with pytest.raises(CollectionMigratingError):
            _perform_kb_collection_delete("kb", 1, True, MagicMock())


class _FakeRollbackFacade:
    """Enough of the API facade for a failed-ingest rollback round-trip."""

    def __init__(self) -> None:
        self.cleanups = 0

    def with_result(self, api_result, result):
        return KBApiOperationResult(result=result)

    async def run_failed_ingest_rollback_async(self, api_result, rollback):
        from types import SimpleNamespace

        try:
            await rollback()
        except Exception as exc:  # mirrors the facade: errors become a result
            return SimpleNamespace(operation_result=api_result, error=exc)
        return SimpleNamespace(operation_result=api_result, error=None)

    def failed_ingest_cleanup_decision(self, api_result, *, successful_documents=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            keeps_new_collection_metadata=False, side_effects_may_remain=False
        )

    def failed_batch_ingest_cleanup_decision(
        self, api_results, *, successful_documents=None
    ):
        from types import SimpleNamespace

        return SimpleNamespace(
            keeps_new_collection_metadata=False, side_effects_may_remain=False
        )

    def with_rollback_complete(self, api_result, value, *, force=False):
        return api_result

    async def cleanup_failed_new_collection_metadata(self, **kwargs) -> None:
        self.cleanups += 1


@pytest.mark.asyncio
async def test_ingest_rollback_delete_runs_after_a_mid_flight_freeze(
    tmp_path,
) -> None:
    """Rollback runs in the route's admission after a freeze lands mid-ingest."""
    import io
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi import UploadFile

    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
    from xagent.core.tools.core.RAG_tools.management import (
        migration_state as state_module,
    )
    from xagent.web.api.kb import ingest

    user, db, access = _migration_mocks()
    rollback_calls: list[str] = []
    facade = _FakeRollbackFacade()
    pin = SimpleNamespace(run=lambda fn, *args, **kwargs: fn(*args, **kwargs))
    upload = UploadFile(filename="doc.pdf", file=io.BytesIO(b"content"))

    def _failing_ingest(**kwargs: object) -> KBApiOperationResult:
        _freeze_state("kb")
        return KBApiOperationResult(
            result=IngestionResult(
                status="error", message="unrelated ingest failure", doc_id="d1"
            )
        )

    def _rollback(**kwargs: object) -> None:
        # The guard the rollback's document delete hits.
        state_module.ensure_collection_writable(str(kwargs["collection_name"]))
        rollback_calls.append(str(kwargs["collection_name"]))

    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new=AsyncMock()),
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError()),
        patch("xagent.web.api.kb._enforce_storage_gate"),
        patch(
            "xagent.web.api.kb._resolve_kb_ingestion_models",
            new=AsyncMock(return_value=(pin, ("m",))),
        ),
        patch("xagent.web.api.kb.check_standalone_gate_async", new=AsyncMock()),
        patch(
            "xagent.web.api.kb._document_existed_before_ingest",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "xagent.web.api.kb.get_upload_path",
            return_value=tmp_path / "uploads" / "doc.pdf",
        ),
        patch(
            "xagent.web.api.kb._upsert_uploaded_file_record",
            return_value=MagicMock(file_id="f1"),
        ),
        patch(
            "xagent.web.api.kb.run_document_ingestion_with_outcome",
            side_effect=_failing_ingest,
        ),
        patch(
            "xagent.web.api.kb._rollback_failed_ingestion",
            side_effect=_rollback,
        ),
        patch(
            "xagent.web.api.kb._get_api_compatibility_facade",
            return_value=facade,
        ),
    ):
        response = await ingest(
            collection="kb",
            file=upload,
            parse_method=None,
            chunk_strategy=None,
            chunk_size=None,
            chunk_overlap=None,
            separators=None,
            embedding_model_id="text-embedding-v4",
            embedding_batch_size=None,
            max_retries=None,
            retry_delay=None,
            _user=user,
            db=db,
        )

    assert response.status_code == 500
    assert rollback_calls == ["kb"]


@pytest.mark.asyncio
async def test_ingest_cloud_rollback_delete_runs_after_a_mid_flight_freeze(
    tmp_path,
) -> None:
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
    from xagent.core.tools.core.RAG_tools.management import (
        migration_state as state_module,
    )
    from xagent.web.api.kb import CloudFile, CloudIngestRequest, ingest_cloud

    user, db, access = _migration_mocks()
    rollback_calls: list[str] = []
    facade = _FakeRollbackFacade()
    pin = SimpleNamespace(run=lambda fn, *args, **kwargs: fn(*args, **kwargs))

    service = MagicMock()
    service.files.return_value.get.return_value.execute.return_value = {
        "name": "a.pdf",
        "mimeType": "application/pdf",
        "size": "4",
    }
    service.files.return_value.get_media.return_value.headers = {}

    class _Downloader:
        def __init__(self, output, request):
            self._output = output

        def next_chunk(self):
            self._output.write(b"data")
            return None, True

    def _failing_ingest(**kwargs: object) -> KBApiOperationResult:
        _freeze_state("kb")
        return KBApiOperationResult(
            result=IngestionResult(
                status="error", message="unrelated cloud failure", doc_id="a.pdf"
            )
        )

    def _rollback(**kwargs: object) -> None:
        state_module.ensure_collection_writable(str(kwargs["collection_name"]))
        rollback_calls.append(str(kwargs["collection_name"]))

    (tmp_path / "uploads").mkdir(parents=True)
    request = CloudIngestRequest(
        files=[CloudFile(provider="google-drive", fileId="f1", fileName="a.pdf")],
        collection="kb",
    )

    with (
        patch(
            "xagent.web.api.kb._effective_knowledge_base_user",
            return_value=(user, access),
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new=AsyncMock()),
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError()),
        patch("xagent.web.api.kb._enforce_storage_gate"),
        patch("xagent.web.api.kb.sessionmaker"),
        patch("xagent.web.api.kb._load_google_credentials", return_value=object()),
        patch("xagent.web.api.kb.build", return_value=service),
        patch("xagent.web.api.kb.MediaIoBaseDownload", _Downloader),
        patch(
            "xagent.web.api.kb._resolve_kb_ingestion_models",
            new=AsyncMock(return_value=(pin, ("m",))),
        ),
        patch("xagent.web.api.kb.check_standalone_gate_async", new=AsyncMock()),
        patch(
            "xagent.web.api.kb._document_existed_before_ingest",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "xagent.web.api.kb.get_upload_path",
            return_value=tmp_path / "uploads" / "a.pdf",
        ),
        patch(
            "xagent.web.api.kb._upsert_uploaded_file_record",
            return_value=MagicMock(file_id="f1"),
        ),
        patch(
            "xagent.web.api.kb.run_document_ingestion_with_outcome",
            side_effect=_failing_ingest,
        ),
        patch(
            "xagent.web.api.kb._rollback_failed_cloud_ingestion",
            side_effect=_rollback,
        ),
        patch(
            "xagent.web.api.kb._save_collection_config_after_ingest",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.kb._get_api_compatibility_facade",
            return_value=facade,
        ),
    ):
        results = await ingest_cloud(request=request, _user=user, db=db)

    assert rollback_calls == ["kb"]
    messages = json.dumps([result.model_dump() for result in results])
    assert "being upgraded" not in messages
