"""Response contract for the application-level namespace-authority handler.

A storage containment violation and an execution-scope authority mismatch are
permanent server-side faults. One handler answers both for every route, so its
status, its body shape per API surface, and what it refuses to publish are
asserted here; ``tests/web/test_file_upload.py`` drives the two faults end to
end through real endpoints.
"""

import json

import pytest
from starlette.requests import Request
from starlette.websockets import WebSocket

from xagent.core.execution_scope import (
    ExecutionScope,
    ExecutionScopeAbstentionMismatchError,
    ExecutionScopeAuthorityError,
)
from xagent.core.file_storage import StorageKeyScopeError
from xagent.web.app import app, storage_namespace_authority_error_handler

_MESSAGE = "Storage namespace authority violation."


def _http_request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )


def _scope_error() -> StorageKeyScopeError:
    return StorageKeyScopeError(
        "Storage key 'users/999/clients/tenant-sentinel/uploads/x' is outside "
        "the bound prefix 'users/7'"
    )


@pytest.mark.parametrize(
    "exception_type",
    [
        StorageKeyScopeError,
        ExecutionScopeAuthorityError,
    ],
)
def test_handler_is_registered_for_both_fault_types(exception_type):
    assert app.exception_handlers[exception_type] is (
        storage_namespace_authority_error_handler
    )


def test_abstention_mismatch_resolves_to_the_same_handler():
    """The abstention subclass has no registration of its own.

    Starlette walks the raised exception's MRO, so a subclass added to the
    authority hierarchy is answered here without another registration.
    """
    assert ExecutionScopeAbstentionMismatchError not in app.exception_handlers
    assert issubclass(
        ExecutionScopeAbstentionMismatchError, ExecutionScopeAuthorityError
    )


@pytest.mark.asyncio
async def test_api_paths_get_a_500_with_no_namespace_values():
    response = await storage_namespace_authority_error_handler(
        _http_request("/api/files/download/abc"), _scope_error()
    )

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body == {"detail": _MESSAGE}
    assert "tenant-sentinel" not in response.body.decode()
    assert "users/7" not in response.body.decode()


@pytest.mark.asyncio
async def test_v1_paths_keep_the_sdk_error_envelope():
    response = await storage_namespace_authority_error_handler(
        _http_request("/v1/tasks"), _scope_error()
    )

    assert response.status_code == 500
    assert json.loads(response.body) == {
        "error": {"code": "internal_error", "message": _MESSAGE}
    }


@pytest.mark.asyncio
async def test_authority_mismatch_body_names_no_scope_field_values():
    exc = ExecutionScopeAuthorityError(
        "5",
        resolver_scope=ExecutionScope(workspace_segments=("clients", "resolver-side")),
        snapshot_scope=ExecutionScope(workspace_segments=("clients", "snapshot-side")),
        mismatched_fields={
            "workspace_segments": (
                ("clients", "snapshot-side"),
                ("clients", "resolver-side"),
            )
        },
        resolver_scope_is_authoritative=True,
    )

    response = await storage_namespace_authority_error_handler(
        _http_request("/api/files/upload"), exc
    )

    assert response.status_code == 500
    assert json.loads(response.body) == {"detail": _MESSAGE}
    assert "resolver-side" not in response.body.decode()
    assert "snapshot-side" not in response.body.decode()


@pytest.mark.asyncio
async def test_websocket_scope_re_raises_instead_of_answering():
    """A websocket cannot receive an HTTP response.

    Starlette would send the returned response onto the connection, so these
    scopes stay with the connection handler that owns them.
    """
    websocket = WebSocket(
        {
            "type": "websocket",
            "path": "/ws/chat",
            "raw_path": b"/ws/chat",
            "query_string": b"",
            "headers": [],
            "scheme": "ws",
            "server": ("testserver", 80),
        },
        receive=None,  # type: ignore[arg-type]
        send=None,  # type: ignore[arg-type]
    )
    exc = _scope_error()

    with pytest.raises(StorageKeyScopeError) as excinfo:
        await storage_namespace_authority_error_handler(websocket, exc)  # type: ignore[arg-type]

    assert excinfo.value is exc
