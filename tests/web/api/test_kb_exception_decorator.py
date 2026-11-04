"""Unit tests for KB API exception decorator.

These tests cover the mapping behavior of `handle_kb_exceptions` in `xagent.web.api.kb`.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

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
