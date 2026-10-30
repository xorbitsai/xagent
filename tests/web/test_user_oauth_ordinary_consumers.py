"""Ordinary xagent surfaces must not observe actor-owned OAuth credentials."""

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.cloud_storage import (
    delete_connected_account,
    get_google_credentials,
    list_connected_accounts,
)
from xagent.web.models.database import Base
from xagent.web.models.google_oauth_grant import (
    GoogleOAuthAppAssociation,
    GoogleOAuthGrant,
)
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.gmail_provisioning import (
    GmailProvisioningError,
    ensure_gmail_mailbox_provisioned,
)
from xagent.web.services.gmail_triggers import GmailOAuthReconnectRequiredError
from xagent.web.services.google_oauth_grants import (
    create_google_oauth_grant_association,
)
from xagent.web.services.triggers import TriggerServiceError, _resolve_gmail_resource

ACTOR = "toby:slack:41:UALICE"


@pytest.fixture
def oauth_rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ordinary-consumers.db'}")
    Base.metadata.create_all(engine)
    local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = local()
    user = User(username="ordinary-user", password_hash="hash")
    db.add(user)
    db.commit()
    db.refresh(user)
    ordinary = UserOAuth(
        user_id=int(user.id),
        provider="google-drive",
        resource_owner_key=None,
        provider_user_id="ordinary",
        email="ordinary@example.com",
        access_token="ordinary",
    )
    actor = UserOAuth(
        user_id=int(user.id),
        provider="google-drive",
        resource_owner_key=ACTOR,
        provider_user_id="actor",
        email="actor@example.com",
        access_token="actor",
    )
    actor_gmail = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        resource_owner_key=ACTOR,
        provider_user_id="actor-gmail",
        email="actor@gmail.example",
        access_token="actor-gmail",
    )
    db.add_all([ordinary, actor, actor_gmail])
    db.commit()
    for row in (ordinary, actor, actor_gmail):
        db.refresh(row)
    try:
        yield db, user, ordinary, actor, actor_gmail
    finally:
        db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_cloud_account_listing_contains_only_ordinary_rows(oauth_rows) -> None:
    db, user, ordinary, _actor, _actor_gmail = oauth_rows

    accounts = await list_connected_accounts(provider=None, db=db, user=user)

    assert [account["id"] for account in accounts] == [int(ordinary.id)]


@pytest.mark.asyncio
async def test_cloud_account_listing_hides_reconnect_tombstones(oauth_rows) -> None:
    db, user, ordinary, _actor, _actor_gmail = oauth_rows
    ordinary.access_token = ""
    db.commit()

    accounts = await list_connected_accounts(provider=None, db=db, user=user)

    assert accounts == []


@pytest.mark.asyncio
async def test_cloud_account_delete_cannot_address_an_actor_row(oauth_rows) -> None:
    db, user, _ordinary, actor, _actor_gmail = oauth_rows

    with pytest.raises(HTTPException) as exc_info:
        await delete_connected_account(int(actor.id), db=db, user=user)

    assert exc_info.value.status_code == 404
    assert db.get(UserOAuth, int(actor.id)) is not None


@pytest.mark.asyncio
async def test_cloud_account_delete_removes_last_shared_google_grant(
    oauth_rows,
) -> None:
    db, user, ordinary, _actor, _actor_gmail = oauth_rows
    shared = UserOAuth(
        user_id=int(user.id),
        provider="google",
        provider_user_id="ordinary",
        access_token="shared-access",
        refresh_token="shared-refresh",
        scope="scope.drive",
    )
    db.add(shared)
    db.flush()
    ordinary.refresh_token = "shared-refresh"
    ordinary.scope = "scope.drive"
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-drive",
        legacy_oauth_account=ordinary,
    )
    db.commit()
    ordinary_id = int(ordinary.id)
    shared_id = int(shared.id)

    accounts = await list_connected_accounts(provider=None, db=db, user=user)
    assert [account["id"] for account in accounts] == [ordinary_id]
    with pytest.raises(HTTPException) as exc_info:
        await delete_connected_account(shared_id, db=db, user=user)
    assert exc_info.value.status_code == 404
    assert db.get(UserOAuth, shared_id) is not None

    await delete_connected_account(ordinary_id, db=db, user=user)

    assert db.get(UserOAuth, ordinary_id) is None
    assert db.get(UserOAuth, shared_id) is None
    assert db.query(GoogleOAuthGrant).count() == 0
    assert db.query(GoogleOAuthAppAssociation).count() == 0


@pytest.mark.asyncio
async def test_cloud_account_delete_preserves_shared_google_sibling(oauth_rows) -> None:
    db, user, ordinary, _actor, _actor_gmail = oauth_rows
    shared = UserOAuth(
        user_id=int(user.id),
        provider="google",
        provider_user_id="ordinary",
        access_token="shared-access",
        refresh_token="shared-refresh",
        scope="scope.drive scope.sheets",
    )
    sheets = UserOAuth(
        user_id=int(user.id),
        provider="google-sheets",
        provider_user_id="ordinary",
        access_token="shared-access",
        refresh_token="shared-refresh",
        scope="scope.drive scope.sheets",
    )
    db.add_all([shared, sheets])
    db.flush()
    ordinary.refresh_token = "shared-refresh"
    ordinary.scope = "scope.drive scope.sheets"
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-drive",
        legacy_oauth_account=ordinary,
    )
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="client-a",
        app_id="google-sheets",
        legacy_oauth_account=sheets,
    )
    db.commit()
    ordinary_id = int(ordinary.id)
    shared_id = int(shared.id)
    sheets_id = int(sheets.id)

    await delete_connected_account(ordinary_id, db=db, user=user)

    assert db.get(UserOAuth, ordinary_id) is None
    assert db.get(UserOAuth, shared_id) is not None
    assert db.get(UserOAuth, sheets_id) is not None
    assert db.query(GoogleOAuthGrant).count() == 1
    association = db.query(GoogleOAuthAppAssociation).one()
    assert association.app_id == "google-sheets"
    assert association.legacy_oauth_account_id == sheets_id


def test_cloud_credentials_cannot_select_an_actor_row_by_id(oauth_rows) -> None:
    db, user, _ordinary, actor, _actor_gmail = oauth_rows

    with pytest.raises(HTTPException) as exc_info:
        get_google_credentials(int(user.id), db, int(actor.id))

    assert exc_info.value.status_code == 404


def test_cloud_credentials_require_reconnect_for_a_token_cleared_row(
    oauth_rows,
) -> None:
    db, user, ordinary, _actor, _actor_gmail = oauth_rows
    ordinary.access_token = ""
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        get_google_credentials(int(user.id), db, int(ordinary.id))

    assert exc_info.value.status_code == 401
    assert "reconnect" in str(exc_info.value.detail).lower()


def test_gmail_watch_provisioning_rejects_an_actor_row(oauth_rows) -> None:
    db, _user, _ordinary, _actor, actor_gmail = oauth_rows

    with pytest.raises(GmailProvisioningError, match="ordinary Gmail account"):
        ensure_gmail_mailbox_provisioned(db, actor_gmail)


def test_gmail_watch_provisioning_rejects_a_non_gmail_row(oauth_rows) -> None:
    db, _user, ordinary, _actor, _actor_gmail = oauth_rows

    with pytest.raises(GmailProvisioningError, match="ordinary Gmail account"):
        ensure_gmail_mailbox_provisioned(db, ordinary)


def test_gmail_consumers_require_reconnect_for_a_token_cleared_row(
    oauth_rows,
) -> None:
    db, user, _ordinary, _actor, _actor_gmail = oauth_rows
    tombstone = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        email="stale@gmail.example",
        access_token="",
    )
    db.add(tombstone)
    db.commit()
    db.refresh(tombstone)

    with pytest.raises(GmailOAuthReconnectRequiredError, match="reconnect required"):
        ensure_gmail_mailbox_provisioned(db, tombstone)

    with pytest.raises(TriggerServiceError, match="reconnect required"):
        _resolve_gmail_resource(
            db,
            user_id=int(user.id),
            oauth_account_id=int(tombstone.id),
        )


def test_gmail_trigger_binding_rejects_an_actor_row(oauth_rows) -> None:
    db, user, _ordinary, _actor, actor_gmail = oauth_rows

    with pytest.raises(TriggerServiceError, match="Gmail account not found"):
        _resolve_gmail_resource(
            db,
            user_id=int(user.id),
            oauth_account_id=int(actor_gmail.id),
        )


def test_gmail_trigger_binding_rejects_a_non_gmail_row(oauth_rows) -> None:
    db, user, ordinary, _actor, _actor_gmail = oauth_rows

    with pytest.raises(
        TriggerServiceError,
        match="Selected account is not a Gmail account",
    ):
        _resolve_gmail_resource(
            db,
            user_id=int(user.id),
            oauth_account_id=int(ordinary.id),
        )
