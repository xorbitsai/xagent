"""Integration tests for scoped personal API-key management."""

from datetime import datetime, timedelta, timezone

import pytest

from xagent.web.models.user import User
from xagent.web.models.user_api_key import UserApiKey
from xagent.web.schemas.user_api_key import PersonalAPIKeyRevokeResponse
from xagent.web.services.api_keys import UserApiKeyService
from xagent.web.services.personal_key_scope import (
    PersonalKeyAccessScope,
    set_personal_key_scope_hook,
)

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _clear_personal_key_scope_hook():
    set_personal_key_scope_hook(None)
    yield
    set_personal_key_scope_hook(None)


def _create_personal_key(headers: dict[str, str]) -> dict:
    response = client.post("/api/me/personal-keys", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User.id).filter(User.username == username).one()[0])
    finally:
        db.close()


def _assert_utc_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert parsed.utcoffset() == timedelta(0)
    return parsed


def test_list_returns_only_self_keys_without_disclosing_one_time_secrets():
    headers = _admin_headers()
    created = _create_personal_key(headers)
    bob_headers = _register_second_user()
    bobs_key = _create_personal_key(bob_headers)

    response = client.get("/api/personal-api-keys", headers=headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["can_manage_others"] is False
    item = next(item for item in body["items"] if item["id"] == created["id"])
    assert item["key_prefix"] == created["key_prefix"]
    assert item["owner"] == {
        "id": _user_id("admin"),
        "username": "admin",
        "email": "admin@example.com",
    }
    assert created["full_key"] not in response.text
    assert "full_key" not in item
    assert all(item["id"] != bobs_key["id"] for item in body["items"])


def test_list_derives_active_expired_and_revoked_status_with_revocation_precedence():
    headers = _admin_headers()
    active = _create_personal_key(headers)
    expired = _create_personal_key(headers)
    revoked_and_expired = _create_personal_key(headers)
    now = datetime.now(timezone.utc)
    expired_at = (now - timedelta(days=1)).replace(tzinfo=None)
    revoked_at = now.replace(tzinfo=None)

    db = _direct_db_session()
    try:
        db.query(UserApiKey).filter(UserApiKey.id == expired["id"]).update(
            {"expires_at": expired_at}
        )
        db.query(UserApiKey).filter(UserApiKey.id == revoked_and_expired["id"]).update(
            {"expires_at": expired_at, "revoked_at": revoked_at}
        )
        db.commit()
    finally:
        db.close()

    response = client.get("/api/personal-api-keys", headers=headers)

    assert response.status_code == 200, response.text
    items_by_id = {item["id"]: item for item in response.json()["items"]}
    assert items_by_id[active["id"]]["status"] == "active"
    assert items_by_id[expired["id"]]["status"] == "expired"
    assert items_by_id[revoked_and_expired["id"]]["status"] == "revoked"
    assert datetime.fromisoformat(
        items_by_id[active["id"]]["created_at"]
    ).utcoffset() == (timedelta(0))
    assert datetime.fromisoformat(items_by_id[expired["id"]]["expires_at"]) == (
        expired_at.replace(tzinfo=timezone.utc)
    )
    assert datetime.fromisoformat(
        items_by_id[revoked_and_expired["id"]]["revoked_at"]
    ) == (revoked_at.replace(tzinfo=timezone.utc))


def test_default_scope_returns_404_for_another_owners_key():
    admin_headers = _admin_headers()
    bob_headers = _register_second_user()
    bobs_key = _create_personal_key(bob_headers)

    response = client.delete(
        f"/api/personal-api-keys/{bobs_key['id']}", headers=admin_headers
    )

    assert response.status_code == 404


def test_authorized_scope_lists_and_revokes_other_owner_keys_idempotently():
    admin_headers = _admin_headers()
    bob_headers = _register_second_user()
    bobs_key = _create_personal_key(bob_headers)
    bob_id = _user_id("bob")

    set_personal_key_scope_hook(
        lambda _db, actor: PersonalKeyAccessScope(
            owner_user_ids=(int(actor.id), bob_id), can_manage_others=True
        )
    )

    listed = client.get("/api/personal-api-keys", headers=admin_headers)

    assert listed.status_code == 200, listed.text
    assert listed.json()["can_manage_others"] is True
    item = next(item for item in listed.json()["items"] if item["id"] == bobs_key["id"])
    assert item["owner"]["id"] == bob_id
    assert item["owner"]["username"] == "bob"

    first_revoke = client.delete(
        f"/api/personal-api-keys/{bobs_key['id']}", headers=admin_headers
    )
    second_revoke = client.delete(
        f"/api/personal-api-keys/{bobs_key['id']}", headers=admin_headers
    )

    assert first_revoke.status_code == 200, first_revoke.text
    assert first_revoke.json()["revoked"] is True
    assert second_revoke.status_code == 200, second_revoke.text
    assert second_revoke.json() == {
        "revoked": False,
        "revoked_at": first_revoke.json()["revoked_at"],
    }


def test_key_disappearing_after_scope_lookup_returns_404(monkeypatch):
    headers = _admin_headers()
    created = _create_personal_key(headers)

    def _missing_lifecycle_row(_service, _user_id, _key_id):
        return PersonalAPIKeyRevokeResponse(revoked=False, revoked_at=None)

    monkeypatch.setattr(UserApiKeyService, "revoke_key", _missing_lifecycle_row)

    response = client.delete(f"/api/personal-api-keys/{created['id']}", headers=headers)

    assert response.status_code == 404
    assert response.json() == {"detail": "Personal API key not found"}


def test_legacy_personal_key_contract_uses_utc_timestamps():
    headers = _admin_headers()

    create_response = client.post("/api/me/personal-keys", headers=headers)

    assert create_response.status_code == 200, create_response.text
    created = create_response.json()
    assert set(created) == {
        "id",
        "full_key",
        "key_prefix",
        "created_at",
        "expires_at",
    }
    assert created["full_key"].startswith("xag_personal_")
    _assert_utc_timestamp(created["created_at"])

    expires_at = datetime(2030, 1, 2, 3, 4, 5)
    db = _direct_db_session()
    try:
        db.query(UserApiKey).filter(UserApiKey.id == created["id"]).update(
            {"expires_at": expires_at}
        )
        db.commit()
    finally:
        db.close()

    list_response = client.get("/api/me/personal-keys", headers=headers)

    assert list_response.status_code == 200, list_response.text
    listed = next(item for item in list_response.json() if item["id"] == created["id"])
    _assert_utc_timestamp(listed["created_at"])
    assert _assert_utc_timestamp(listed["expires_at"]) == expires_at.replace(
        tzinfo=timezone.utc
    )

    revoke_response = client.delete(
        f"/api/me/personal-keys/{created['id']}", headers=headers
    )

    assert revoke_response.status_code == 200, revoke_response.text
    revoked = revoke_response.json()
    assert revoked["revoked"] is True
    _assert_utc_timestamp(revoked["revoked_at"])
