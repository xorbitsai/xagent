"""API keys are not browser logins: idle expiry never touches them (#3771)."""

from datetime import datetime, timedelta, timezone

import pytest

from xagent.web.models.browser_session import BrowserSession

from ..conftest import _admin_headers, _direct_db_session, client
from .test_auth import _create_agent_and_key, _create_personal_key

pytestmark = pytest.mark.usefixtures("_test_db")


def test_api_keys_keep_working_after_the_browser_login_idles_out(monkeypatch):
    monkeypatch.setenv("XAGENT_BROWSER_IDLE_TIMEOUT_ENABLED", "true")
    personal_key, _ = _create_personal_key()
    agent_id, agent_key, _ = _create_agent_and_key()
    browser = _admin_headers()
    db = _direct_db_session()
    try:
        db.query(BrowserSession).update(
            {
                BrowserSession.last_activity_at: datetime.now(timezone.utc)
                - timedelta(hours=1)
            }
        )
        db.commit()
    finally:
        db.close()

    expired = client.get("/api/auth/verify", headers=browser)
    assert expired.status_code == 401
    assert expired.headers["Error-Type"] == "SessionIdleExpired"
    me = client.get("/v1/me", headers={"Authorization": f"Bearer {personal_key}"})
    assert me.status_code == 200, me.text
    agents = client.get(
        "/v1/agents", headers={"Authorization": f"Bearer {personal_key}"}
    )
    assert agents.status_code == 200, agents.text
    # The agent-bound key still authenticates: an unknown task is a 404, not 401.
    task = client.get(
        "/v1/chat/tasks/999999", headers={"Authorization": f"Bearer {agent_key}"}
    )
    assert task.status_code == 404, task.text
    assert agent_id
