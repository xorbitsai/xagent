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
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.gmail_provisioning import (
    GmailProvisioningError,
    ensure_gmail_mailbox_provisioned,
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
async def test_cloud_account_delete_cannot_address_an_actor_row(oauth_rows) -> None:
    db, user, _ordinary, actor, _actor_gmail = oauth_rows

    with pytest.raises(HTTPException) as exc_info:
        await delete_connected_account(int(actor.id), db=db, user=user)

    assert exc_info.value.status_code == 404
    assert db.get(UserOAuth, int(actor.id)) is not None


def test_cloud_credentials_cannot_select_an_actor_row_by_id(oauth_rows) -> None:
    db, user, _ordinary, actor, _actor_gmail = oauth_rows

    with pytest.raises(HTTPException) as exc_info:
        get_google_credentials(int(user.id), db, int(actor.id))

    assert exc_info.value.status_code == 404


def test_gmail_watch_provisioning_rejects_an_actor_row(oauth_rows) -> None:
    db, _user, _ordinary, _actor, actor_gmail = oauth_rows

    with pytest.raises(GmailProvisioningError, match="ordinary Gmail account"):
        ensure_gmail_mailbox_provisioned(db, actor_gmail)


def test_gmail_watch_provisioning_rejects_a_non_gmail_row(oauth_rows) -> None:
    db, _user, ordinary, _actor, _actor_gmail = oauth_rows

    with pytest.raises(GmailProvisioningError, match="ordinary Gmail account"):
        ensure_gmail_mailbox_provisioned(db, ordinary)


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
