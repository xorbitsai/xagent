"""Storage identity tests for ordinary and actor-owned builtin OAuth rows."""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth


def _where(index) -> str:
    clause = index.dialect_options["sqlite"].get("where")
    if clause is None:
        clause = index.dialect_options["postgresql"].get("where")
    return str(clause if clause is not None else "").lower()


def test_resource_owner_key_is_nullable_and_bounded() -> None:
    column = UserOAuth.__table__.columns["resource_owner_key"]

    assert column.nullable is True
    assert column.type.length == 512


def test_model_declares_ordinary_and_actor_owned_partial_uniqueness() -> None:
    indexes = {index.name: index for index in UserOAuth.__table__.indexes}

    ordinary = indexes["uq_user_oauth_ordinary_account"]
    assert ordinary.unique is True
    assert tuple(column.name for column in ordinary.columns) == (
        "user_id",
        "provider",
        "provider_user_id",
    )
    assert "resource_owner_key is null" in _where(ordinary)

    actor = indexes["uq_user_oauth_actor_account"]
    assert actor.unique is True
    assert tuple(column.name for column in actor.columns) == (
        "user_id",
        "resource_owner_key",
        "provider",
        "provider_user_id",
    )
    assert "resource_owner_key is not null" in _where(actor)

    lookup = indexes["ix_user_oauth_owner_provider"]
    assert lookup.unique is False
    assert tuple(column.name for column in lookup.columns) == (
        "user_id",
        "resource_owner_key",
        "provider",
    )


def _oauth_relationship_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-relationship.db'}")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="oauth-owner", password_hash="hash")
    db.add(user)
    db.commit()
    db.refresh(user)
    db.add_all(
        [
            UserOAuth(
                user_id=int(user.id),
                provider="gmail",
                provider_user_id="ordinary",
                access_token="ordinary-token",
            ),
            UserOAuth(
                user_id=int(user.id),
                provider="gmail",
                resource_owner_key="actor:alice",
                provider_user_id="alice",
                access_token="actor-token",
            ),
        ]
    )
    db.commit()
    db.expire(user, ["oauth_accounts"])
    return engine, db, user


def test_user_oauth_accounts_relationship_contains_only_ordinary_rows(tmp_path) -> None:
    engine, db, user = _oauth_relationship_db(tmp_path)
    try:
        assert [row.resource_owner_key for row in user.oauth_accounts] == [None]
    finally:
        db.close()
        engine.dispose()


def test_sqlite_user_delete_cascades_hidden_actor_oauth_rows(tmp_path) -> None:
    engine, db, user = _oauth_relationship_db(tmp_path)
    try:
        assert [row.resource_owner_key for row in user.oauth_accounts] == [None]

        db.delete(user)
        db.commit()

        assert db.query(UserOAuth).count() == 0
    finally:
        db.close()
        engine.dispose()
