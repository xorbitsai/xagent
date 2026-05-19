"""Tests for KB web ingestion input validation."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.kb import kb_router


@pytest.fixture
def mock_user():
    """Minimal user-like object for ingest dependency."""

    return type("User", (), {"id": 1, "is_admin": False})()


@pytest.fixture
def app_with_kb(mock_user):
    """FastAPI app with kb_router and mocked auth."""

    from unittest.mock import MagicMock

    from xagent.web.api.kb import get_current_user
    from xagent.web.models.database import get_db

    def override_get_current_user():
        return mock_user

    def override_get_db():
        yield MagicMock()

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    return app


@pytest.mark.parametrize(
    "start_url",
    [
        "xinference.cn",
        " www.xinference.cn ",
        "ftp://xinference.cn",
        "https://",
        "http://",
        "://xinference.cn",
        "",
        "   ",
    ],
)
def test_ingest_web_rejects_invalid_start_url(app_with_kb, start_url: str) -> None:
    client = TestClient(app_with_kb)

    resp = client.post(
        "/api/kb/ingest-web",
        data={
            "collection": "test_coll",
            "start_url": start_url,
        },
    )

    assert resp.status_code == 422
    body = resp.json()
    detail = body.get("detail", "")

    # Different FastAPI/Starlette versions may treat empty form fields as missing
    # ("Field required") or pass through as empty string (our custom validator).
    # Accept either behavior for blank inputs.
    if not start_url.strip():
        if isinstance(detail, list):
            assert any(
                (isinstance(item, dict) and item.get("msg") == "Field required")
                for item in detail
            )
        else:
            assert "Invalid start_url" in str(detail)
    else:
        assert "Invalid start_url" in str(detail)


def test_ingest_web_rejects_invalid_render_wait_until(app_with_kb) -> None:
    """Invalid render_wait_until should return 422 at the API boundary."""

    client = TestClient(app_with_kb)
    resp = client.post(
        "/api/kb/ingest-web",
        data={
            "collection": "test_coll",
            "start_url": "https://example.com",
            "render_wait_until": "invalid",
        },
    )

    assert resp.status_code == 422
    assert "Invalid render_wait_until" in str(resp.json().get("detail", ""))


def test_ingest_web_accepts_stripped_url(app_with_kb) -> None:
    """Whitespace/newline around a valid URL should be accepted after strip()."""

    from unittest.mock import AsyncMock, patch

    mock_result = {
        "status": "success",
        "collection": "test_coll",
        "pages_crawled": 1,
        "pages_failed": 0,
        "chunks_stored": 5,
        "message": "OK",
    }

    with patch(
        "xagent.web.api.kb.run_web_ingestion",
        new_callable=AsyncMock,
        return_value=mock_result,
    ) as mock_run:
        client = TestClient(app_with_kb)
        resp = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "test_coll",
                "start_url": "  https://xinference.cn\n",
                "max_pages": "1",
                "max_depth": "1",
                "respect_robots_txt": "false",
            },
        )

        assert resp.status_code != 422

        if mock_run.called:
            call_kwargs = mock_run.call_args[1]
            crawl_config = call_kwargs.get("crawl_config")
            assert crawl_config is not None
            assert crawl_config.start_url == "https://xinference.cn"
