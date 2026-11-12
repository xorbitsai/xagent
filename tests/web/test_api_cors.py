"""Cross-origin clients must be able to read the retry hint."""

from fastapi.testclient import TestClient

from xagent.config import get_cors_origins
from xagent.web.app import app


def test_allowed_origin_exposes_retry_after() -> None:
    origin = get_cors_origins()[0]

    response = TestClient(app).get("/api/deployment-config", headers={"Origin": origin})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "Retry-After" in response.headers["access-control-expose-headers"]
