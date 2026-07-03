"""Shared fixtures + helpers for the ``tests/web/api/`` suite.

Each test file gets:

  - An autouse ``_test_db`` fixture that lays a fresh SQLite schema
    before each test and drops it after.
  - A module-level ``client`` (TestClient) wired up against the routers
    every test file needs: auth, agents, v1.
  - Auth helpers (``_setup_admin`` / ``_login`` / ``_admin_headers`` /
    ``_register_second_user``) for the bootstrap pattern those routers
    share.
  - ``_direct_db_session`` for tests that need to peek at DB rows
    after a request without going back through HTTP.

Test files explicitly import what they use:
    from .conftest import client, _admin_headers, _direct_db_session

This keeps the dependency edges obvious and avoids the "what's
magically in scope?" question that pure-fixture conftests get.
"""

import logging
import os
import shutil
import tempfile
from typing import Iterator

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from xagent.web.api.agent_api_keys import router as agent_api_keys_router
from xagent.web.api.agents import router as agents_router
from xagent.web.api.auth import auth_router
from xagent.web.api.conversation_logs import router as conversation_logs_router
from xagent.web.api.files import file_router
from xagent.web.api.me import router as me_router
from xagent.web.api.share import share_router
from xagent.web.api.triggers import router as triggers_router
from xagent.web.api.v1 import v1_router
from xagent.web.api.v1.errors import V1ApiError, v1_api_error_handler
from xagent.web.api.widget import widget_router
from xagent.web.api.workforces import router as workforces_router
from xagent.web.models.database import Base, get_db, get_engine

logger = logging.getLogger(__name__)


def _override_get_db() -> Iterator[Session]:
    """Yield the (test-scoped) DB session FastAPI's Depends(get_db) needs."""
    db = None
    try:
        db = next(get_db())
        yield db
    finally:
        if db is not None:
            db.close()


# Build the FastAPI app once at module load. ``_test_db`` recreates the
# underlying SQLite file between tests, so the same TestClient object
# is safe to reuse across tests.
app_for_tests = FastAPI()
app_for_tests.include_router(auth_router)
app_for_tests.include_router(me_router)
app_for_tests.include_router(conversation_logs_router)
app_for_tests.include_router(agents_router)
app_for_tests.include_router(agent_api_keys_router)
app_for_tests.include_router(workforces_router)
app_for_tests.include_router(file_router)
app_for_tests.include_router(widget_router)
app_for_tests.include_router(share_router)
app_for_tests.include_router(triggers_router)
app_for_tests.include_router(v1_router)
app_for_tests.add_exception_handler(V1ApiError, v1_api_error_handler)  # type: ignore[arg-type]


# Mirror production's /v1/* envelope guarantee in tests: any non-V1ApiError
# exception inside a /v1/* route returns the stable {"error": {...}} shape
# instead of FastAPI's default {"detail": ...}. The real app (web/app.py)
# does the same inside its global_exception_handler -- this lightweight
# copy keeps the contract testable without dragging the entire web/app.py
# initialization (uploads dir, dynamic memory store, etc) into the test
# fixture.
@app_for_tests.exception_handler(Exception)
async def _v1_internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception in {request.url}: {exc}", exc_info=True)
    if request.url.path.startswith("/v1/"):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error.",
                }
            },
        )
    raise exc


# Same path-aware rewrite for Pydantic / FastAPI request-body validation
# (422). FastAPI raises ``RequestValidationError`` before the endpoint
# runs, so the v1 endpoints can't translate it themselves -- the global
# handler is the only place. /v1/* gets ``{"error":{"code":"invalid_input"}}``,
# /api/* falls through to FastAPI's default {"detail": [...]}.
@app_for_tests.exception_handler(RequestValidationError)
async def _v1_validation_error_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/v1/"):
        errors = exc.errors()
        first = errors[0] if errors else {}
        msg = first.get("msg") or "Invalid request body"
        loc = ".".join(str(p) for p in first.get("loc", []) if p not in (None, "body"))
        if loc:
            msg = f"{msg} ({loc})"
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_input", "message": msg}},
        )
    # /api/* uses FastAPI's default shape
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


app_for_tests.dependency_overrides[get_db] = _override_get_db
client = TestClient(app_for_tests, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_trigger_rate_limiter() -> Iterator[None]:
    """Fresh rate-limit window per test.

    The trigger rate limiter is a process-wide singleton with an in-memory
    moving window; without a reset, fast suites reusing the same user id
    would eventually trip the CRUD limit across unrelated tests.
    """
    from xagent.web.services.trigger_rate_limit import reset_trigger_rate_limiter

    reset_trigger_rate_limiter()
    yield


@pytest.fixture
def _test_db() -> Iterator[None]:
    """Per-test SQLite DB so tables come up empty for every test.

    Mirrors the pattern from ``test_agents_kb_tool_validation.py``:
    temp sqlite file, ``init_db`` lays out the schema, drop all on
    teardown.

    Not ``autouse=True`` -- some sibling test files
    (``test_tools_api.py``) define their own DB fixture and would
    double-init if this one ran automatically. Tests that want it
    request it explicitly, typically via a class-level autouse
    wrapper:

        @pytest.fixture(autouse=True)
        def _db(self, _test_db):
            pass
    """
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{temp_db_path}"
    init_db(db_url=db_url)

    yield

    Base.metadata.drop_all(bind=get_engine())
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass


# ===== Auth helpers =====


def _setup_admin() -> None:
    """Idempotently bootstrap the admin user via /api/auth/setup-admin."""
    status = client.get("/api/auth/setup-status")
    assert status.status_code == 200
    if status.json().get("needs_setup", True):
        resp = client.post(
            "/api/auth/setup-admin",
            json={
                "username": "admin",
                "email": "admin@example.com",
                "password": "admin123",
            },
        )
        assert resp.status_code == 200


def _login(username: str = "admin", password: str = "admin123") -> dict[str, str]:
    """Log in and return the bearer header dict ready to splat into a request."""
    resp = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _admin_headers() -> dict[str, str]:
    """Setup admin if needed and return the admin's auth header."""
    _setup_admin()
    return _login()


def _register_second_user(
    username: str = "bob", password: str = "bobpass1"
) -> dict[str, str]:
    """Register a second user via the public endpoint, return their auth header.

    Used by cross-user-isolation tests (e.g. accessing another user's
    agent must return 404 without revealing that the agent exists).
    """
    resp = client.post(
        "/api/auth/register",
        json={
            "username": username,
            "email": f"{username}@example.com",
            "password": password,
        },
    )
    assert resp.status_code == 200, resp.text
    return _login(username, password)


# ===== Domain helpers =====


def _direct_db_session() -> Session:
    """Open a session against the same test DB FastAPI is using.

    Use this when a test needs to peek at DB rows directly to confirm
    side effects of an HTTP call (e.g. assert a row's ``revoked_at`` is
    non-NULL after DELETE). Always close the session in a try/finally.
    """
    return next(get_db())
