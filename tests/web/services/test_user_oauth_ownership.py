"""Owner-scoped lookup and revocation for builtin OAuth credentials."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.user_oauth import (
    delete_scoped_user_oauth_accounts,
    get_scoped_user_oauth_account,
    get_user_oauth_account_by_id,
    list_scoped_user_oauth_accounts,
    normalize_user_oauth_resource_owner_key,
)

ALICE = "toby:slack:41:UALICE"
BOB = "toby:slack:41:UBOB"


def test_owner_key_normalization_rejects_invalid_actor_keys() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        normalize_user_oauth_resource_owner_key(41)
    with pytest.raises(ValueError, match="must not be blank"):
        normalize_user_oauth_resource_owner_key("   ")
    with pytest.raises(ValueError, match="exceeds 512 characters"):
        normalize_user_oauth_resource_owner_key("x" * 513)


def _db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-service.db'}")
    Base.metadata.create_all(engine)
    local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = local()
    user = User(username="owner", password_hash="hash")
    db.add(user)
    db.commit()
    db.refresh(user)
    rows = [
        UserOAuth(
            user_id=int(user.id),
            provider="gmail",
            resource_owner_key=None,
            provider_user_id="ordinary",
            access_token="ordinary",
        ),
        UserOAuth(
            user_id=int(user.id),
            provider="gmail",
            resource_owner_key=ALICE,
            provider_user_id="alice",
            access_token="alice",
        ),
        UserOAuth(
            user_id=int(user.id),
            provider="gmail",
            resource_owner_key=BOB,
            provider_user_id="bob",
            access_token="bob",
        ),
    ]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    return engine, db, user, rows


def test_listing_never_crosses_owner_namespaces(tmp_path) -> None:
    engine, db, user, _rows = _db(tmp_path)
    try:
        ordinary = list_scoped_user_oauth_accounts(
            db, user_id=int(user.id), resource_owner_key=None
        )
        alice = list_scoped_user_oauth_accounts(
            db, user_id=int(user.id), resource_owner_key=ALICE
        )

        assert [row.access_token for row in ordinary] == ["ordinary"]
        assert [row.access_token for row in alice] == ["alice"]
    finally:
        db.close()
        engine.dispose()


def test_direct_id_lookup_requires_the_expected_owner(tmp_path) -> None:
    engine, db, user, rows = _db(tmp_path)
    alice_row = rows[1]
    try:
        assert (
            get_scoped_user_oauth_account(
                db,
                user_id=int(user.id),
                account_id=int(alice_row.id),
                resource_owner_key=ALICE,
            )
            is not None
        )
        assert (
            get_scoped_user_oauth_account(
                db,
                user_id=int(user.id),
                account_id=int(alice_row.id),
                resource_owner_key=BOB,
            )
            is None
        )
        assert (
            get_scoped_user_oauth_account(
                db,
                user_id=int(user.id),
                account_id=int(alice_row.id),
                resource_owner_key=None,
            )
            is None
        )
    finally:
        db.close()
        engine.dispose()


def test_scoped_id_lookup_refreshes_an_identity_mapped_account(tmp_path) -> None:
    engine, db, user, rows = _db(tmp_path)
    alice_row = rows[1]
    other_db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        other_db.query(UserOAuth).filter(UserOAuth.id == int(alice_row.id)).update(
            {UserOAuth.access_token: "alice-refreshed"},
            synchronize_session=False,
        )
        other_db.commit()

        account = get_scoped_user_oauth_account(
            db,
            user_id=int(user.id),
            account_id=int(alice_row.id),
            resource_owner_key=ALICE,
        )

        assert account is alice_row
        assert account.access_token == "alice-refreshed"
    finally:
        other_db.close()
        db.close()
        engine.dispose()


def test_owner_only_id_lookup_supports_internal_foreign_key_consumers(tmp_path) -> None:
    engine, db, _user, rows = _db(tmp_path)
    try:
        assert (
            get_user_oauth_account_by_id(
                db,
                account_id=int(rows[0].id),
                resource_owner_key=None,
            )
            is rows[0]
        )
        assert (
            get_user_oauth_account_by_id(
                db,
                account_id=int(rows[1].id),
                resource_owner_key=None,
            )
            is None
        )
    finally:
        db.close()
        engine.dispose()


def test_owner_only_id_lookup_refreshes_an_identity_mapped_account(tmp_path) -> None:
    engine, db, _user, rows = _db(tmp_path)
    ordinary_row = rows[0]
    other_db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        other_db.query(UserOAuth).filter(UserOAuth.id == int(ordinary_row.id)).update(
            {UserOAuth.access_token: "ordinary-refreshed"},
            synchronize_session=False,
        )
        other_db.commit()

        account = get_user_oauth_account_by_id(
            db,
            account_id=int(ordinary_row.id),
            resource_owner_key=None,
        )

        assert account is ordinary_row
        assert account.access_token == "ordinary-refreshed"
    finally:
        other_db.close()
        db.close()
        engine.dispose()


def test_destructive_provider_filter_must_be_explicit(tmp_path) -> None:
    engine, db, user, _rows = _db(tmp_path)
    try:
        with pytest.raises(TypeError):
            delete_scoped_user_oauth_accounts(
                db,
                user_id=int(user.id),
                resource_owner_key=ALICE,
            )
    finally:
        db.close()
        engine.dispose()


def test_empty_provider_filter_deletes_nothing(tmp_path) -> None:
    engine, db, user, _rows = _db(tmp_path)
    try:
        assert (
            delete_scoped_user_oauth_accounts(
                db,
                user_id=int(user.id),
                resource_owner_key=ALICE,
                providers=[],
            )
            == 0
        )
        assert db.query(UserOAuth).count() == 3
    finally:
        db.close()
        engine.dispose()


def test_none_provider_filter_deletes_all_for_owner_without_committing(
    tmp_path,
) -> None:
    engine, db, user, _rows = _db(tmp_path)
    other_db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        assert (
            delete_scoped_user_oauth_accounts(
                db,
                user_id=int(user.id),
                resource_owner_key=ALICE,
                providers=None,
            )
            == 1
        )
        assert (
            db.query(UserOAuth).filter(UserOAuth.resource_owner_key == ALICE).count()
            == 0
        )
        assert (
            other_db.query(UserOAuth)
            .filter(UserOAuth.resource_owner_key == ALICE)
            .count()
            == 1
        )
    finally:
        other_db.close()
        db.rollback()
        db.close()
        engine.dispose()


def test_actor_revocation_leaves_ordinary_and_other_actor_rows(tmp_path) -> None:
    engine, db, user, _rows = _db(tmp_path)
    try:
        deleted = delete_scoped_user_oauth_accounts(
            db,
            user_id=int(user.id),
            resource_owner_key=ALICE,
            providers=["gmail"],
        )
        db.commit()

        assert deleted == 1
        remaining = {
            (row.resource_owner_key, row.access_token)
            for row in db.query(UserOAuth).all()
        }
        assert remaining == {(None, "ordinary"), (BOB, "bob")}
    finally:
        db.close()
        engine.dispose()
