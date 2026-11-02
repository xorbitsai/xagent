"""Personal keys on the ``/v1/chat/tasks`` routes.

A personal key resolves to its holder. Tasks it creates are owned by the
holder, record ``sdk_credential_kind = 'personal'``, and are reachable only
by personal keys of that holder; runtime keys never reach them, and a
personal key never reaches a task a runtime key or another surface created.
"""

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from xagent.web.api.v1.deps import (
    AgentPrincipalSnapshot,
    ApiKeyPrincipal,
    PersonalApiKeySnapshot,
    RuntimeApiKeySnapshot,
    UserPrincipalSnapshot,
    _resolve_principal_from_credentials,
)

from ..conftest import client
from .test_auth import _assert_invalid_api_key, _create_personal_key

pytestmark = pytest.mark.usefixtures("_test_db")

HOLDER = UserPrincipalSnapshot(id=5, username="holder", email=None, is_admin=False)
AGENT = AgentPrincipalSnapshot(
    id=3, user_id=9, execution_mode="balanced", status="draft", origin="user"
)


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def test_api_key_principal_rejects_inconsistent_kind() -> None:
    personal = PersonalApiKeySnapshot(key_prefix="abc123")
    runtime = RuntimeApiKeySnapshot(key_prefix="abc123")
    with pytest.raises(ValueError):
        ApiKeyPrincipal(key=personal, agent=AGENT)
    with pytest.raises(ValueError):
        ApiKeyPrincipal(key=runtime, agent=AGENT, user=HOLDER)

    holder = ApiKeyPrincipal(key=personal, user=HOLDER)
    agent_key = ApiKeyPrincipal(key=runtime, agent=AGENT)
    assert (holder.owner_user_id, agent_key.owner_user_id) == (5, 9)
    # Same prefix, different key tables: the stream slot identity differs,
    # and a runtime key keeps the bare prefix it has always counted under.
    assert holder.stream_principal_id == "personal:abc123"
    assert agent_key.stream_principal_id == "abc123"
    with pytest.raises(RuntimeError):
        _ = ApiKeyPrincipal(key=runtime).owner_user_id


def test_personal_key_resolves_to_its_holder() -> None:
    full_key, prefix = _create_personal_key()
    me = client.get("/v1/me", headers=_bearer(full_key)).json()

    principal = _resolve_principal_from_credentials(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    )

    assert principal == ApiKeyPrincipal(
        key=PersonalApiKeySnapshot(key_prefix=prefix),
        user=UserPrincipalSnapshot(
            id=me["user_id"], username="admin", email="admin@example.com", is_admin=True
        ),
    )


def test_personal_key_rejected_on_runtime_only_routes() -> None:
    full_key, _prefix = _create_personal_key()

    run = client.post(
        "/v1/workforces/1/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "go"}},
    )
    _assert_invalid_api_key(run)
    a2a = client.get(
        "/api/a2a/agents/1/tasks", headers={**_bearer(full_key), "A2A-Version": "1.0"}
    )
    assert a2a.status_code == 401, a2a.text
    assert a2a.json()["error"]["details"][0]["reason"] == "INVALID_API_KEY"
