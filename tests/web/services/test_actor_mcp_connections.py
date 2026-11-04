from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from xagent.core.utils.encryption import _is_encrypted, encrypt_env_dict, get_cipher
from xagent.web.models.actor_mcp_connection import ActorMCPServerConnection
from xagent.web.models.database import Base
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services.actor_mcp_connections import (
    ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH,
    ActorMCPConnectionConflictError,
    ActorMCPConnectionCredentialCorruptionError,
    ActorMCPConnectionNotFoundOrStaleError,
    ActorMCPConnectionValidationError,
    create_actor_mcp_connection,
)
from xagent.web.services.actor_mcp_connections import (
    delete_actor_mcp_connection as _delete_exact,
)
from xagent.web.services.actor_mcp_connections import (
    get_actor_mcp_connection_credentials_internal as _get_internal,
)
from xagent.web.services.actor_mcp_connections import (
    get_actor_mcp_connection_metadata,
)
from xagent.web.services.actor_mcp_connections import (
    list_actor_mcp_connection_metadata as list_actor_mcp_connection_snapshots,
)
from xagent.web.services.actor_mcp_connections import (
    update_actor_mcp_connection_credentials as _update_exact,
)

ALICE_OWNER = "toby:slack:T1:alice"
BOB_OWNER = "toby:slack:T1:bob"


def get_actor_mcp_connection_snapshot(db: Session, **kwargs):
    try:
        metadata = get_actor_mcp_connection_metadata(
            db,
            user_id=kwargs["user_id"],
            resource_owner_key=kwargs["resource_owner_key"],
            app_id=kwargs["app_id"],
        )
        return _get_internal(
            db,
            catalog_app_generation=metadata.catalog_app_generation,
            **kwargs,
        )
    except ActorMCPConnectionNotFoundOrStaleError:
        return None


def update_actor_mcp_connection_credentials(db: Session, **kwargs):
    try:
        metadata = _update_exact(db, **kwargs)
    except ActorMCPConnectionNotFoundOrStaleError:
        return None
    return _get_internal(
        db,
        user_id=metadata.user_id,
        resource_owner_key=metadata.resource_owner_key,
        app_id=metadata.app_id,
        catalog_app_generation=metadata.catalog_app_generation,
        expected_lifecycle_generation=metadata.lifecycle_generation,
    )


def delete_actor_mcp_connection(db: Session, **kwargs) -> bool:
    try:
        _delete_exact(db, **kwargs)
    except ActorMCPConnectionNotFoundOrStaleError:
        return False
    return True


@pytest.fixture()
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _seed_app(db: Session, *, app_id: str = "posthog") -> PublicMCPApp:
    app = PublicMCPApp(
        app_id=app_id,
        name=app_id,
        transport="stdio",
        launch_config={"command": "persisted-values-are-not-trusted"},
    )
    db.add(app)
    db.flush()
    return app


def _seed_users(db: Session) -> tuple[User, User]:
    users = (
        User(username="actor-owner", password_hash="x"),
        User(username="other-user", password_hash="x"),
    )
    db.add_all(users)
    db.flush()
    return users


def _create(
    db: Session,
    *,
    user_id: int,
    owner: str,
    app_id: str,
    credentials: dict[str, str] | None = None,
):
    return create_actor_mcp_connection(
        db,
        user_id=user_id,
        resource_owner_key=owner,
        app_id=app_id,
        credentials=credentials
        if credentials is not None
        else {"POSTHOG_API_KEY": "secret-value", "POSTHOG_HOST": "https://x.test"},
    )


def test_credentials_are_encrypted_at_rest_and_decrypted_in_snapshot(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)

    snapshot = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )
    secret_snapshot = get_actor_mcp_connection_snapshot(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=snapshot.lifecycle_generation,
    )
    stored = db.query(ActorMCPServerConnection).one()

    assert secret_snapshot is not None
    assert secret_snapshot.credentials == {
        "POSTHOG_API_KEY": "secret-value",
        "POSTHOG_HOST": "https://x.test",
    }
    assert stored.encrypted_env is not None
    assert all(_is_encrypted(value) for value in stored.encrypted_env.values())
    assert "secret-value" not in str(stored.encrypted_env)
    assert "https://x.test" not in str(stored.encrypted_env)
    assert "secret-value" not in repr(stored)
    assert "secret-value" not in repr(secret_snapshot)
    assert "https://x.test" not in repr(secret_snapshot)


def test_reads_and_deletes_require_exact_user_owner_and_app_tuple(db: Session) -> None:
    user, other_user = _seed_users(db)
    app = _seed_app(db)
    created = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )

    assert (
        get_actor_mcp_connection_snapshot(
            db,
            user_id=user.id,
            resource_owner_key=BOB_OWNER,
            app_id=app.app_id,
            expected_lifecycle_generation=created.lifecycle_generation,
        )
        is None
    )
    assert (
        get_actor_mcp_connection_snapshot(
            db,
            user_id=other_user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            expected_lifecycle_generation=created.lifecycle_generation,
        )
        is None
    )
    assert (
        list_actor_mcp_connection_snapshots(
            db, user_id=other_user.id, resource_owner_key=ALICE_OWNER
        )
        == []
    )
    assert not delete_actor_mcp_connection(
        db,
        user_id=user.id,
        resource_owner_key=BOB_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=created.lifecycle_generation,
    )
    assert (
        get_actor_mcp_connection_metadata(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
        )
        == created
    )


def test_actor_tuple_uniqueness_is_enforced_by_app_and_catalog_generation(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    first_app = _seed_app(db, app_id="posthog")
    second_app = _seed_app(db, app_id="google-maps")
    db.add(
        ActorMCPServerConnection(
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=first_app.app_id,
            catalog_app_generation=first_app.generation,
        )
    )
    db.commit()

    db.add(
        ActorMCPServerConnection(
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=first_app.app_id,
            catalog_app_generation=second_app.generation,
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    db.add(
        ActorMCPServerConnection(
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=second_app.app_id,
            catalog_app_generation=first_app.generation,
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()


def test_partial_update_merges_omitted_fields_and_preserves_generation(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )

    updated = update_actor_mcp_connection_credentials(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=original.lifecycle_generation,
        credentials={"POSTHOG_API_KEY": "rotated"},
    )

    assert updated is not None
    assert updated.lifecycle_generation == original.lifecycle_generation
    assert updated.credentials == {
        "POSTHOG_API_KEY": "rotated",
        "POSTHOG_HOST": "https://x.test",
    }


def test_lifecycle_generation_cannot_be_updated(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )
    row = db.query(ActorMCPServerConnection).one()
    db.commit()
    original_generation = row.lifecycle_generation

    row.lifecycle_generation = uuid.uuid4()
    with pytest.raises(ValueError, match="immutable"):
        db.flush()
    db.rollback()

    assert (
        db.query(ActorMCPServerConnection).one().lifecycle_generation
        == original_generation
    )


def test_hard_delete_and_recreate_gets_a_new_generation(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )

    assert delete_actor_mcp_connection(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=original.lifecycle_generation,
    )
    assert db.query(ActorMCPServerConnection).count() == 0
    replacement = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
        credentials={
            "POSTHOG_API_KEY": "new-secret",
            "POSTHOG_HOST": "https://new.test",
        },
    )

    assert isinstance(replacement.lifecycle_generation, uuid.UUID)
    assert replacement.lifecycle_generation != original.lifecycle_generation
    replacement_secret = get_actor_mcp_connection_snapshot(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=replacement.lifecycle_generation,
    )
    assert replacement_secret is not None
    assert replacement_secret.credentials == {
        "POSTHOG_API_KEY": "new-secret",
        "POSTHOG_HOST": "https://new.test",
    }


def test_create_rejects_incomplete_unknown_blank_null_and_oversize_credentials(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    invalid_credentials = [
        {"POSTHOG_API_KEY": "secret-value"},
        {
            "POSTHOG_API_KEY": "secret-value",
            "POSTHOG_HOST": "https://x.test",
            "EXTRA": "x",
        },
        {"POSTHOG_API_KEY": "secret-value", "POSTHOG_HOST": ""},
        {"POSTHOG_API_KEY": "secret-value", "POSTHOG_HOST": None},
        {
            "POSTHOG_API_KEY": "x" * (ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH + 1),
            "POSTHOG_HOST": "https://x.test",
        },
    ]
    for credentials in invalid_credentials:
        with pytest.raises(ActorMCPConnectionValidationError):
            create_actor_mcp_connection(
                db,
                user_id=user.id,
                resource_owner_key=ALICE_OWNER,
                app_id=app.app_id,
                credentials=credentials,
            )
    assert db.query(ActorMCPServerConnection).count() == 0


def test_update_rejects_unknown_blank_null_and_oversize_credentials(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )
    invalid_credentials = [
        {"EXTRA": "x"},
        {"POSTHOG_API_KEY": ""},
        {"POSTHOG_API_KEY": None},
        {"POSTHOG_API_KEY": "x" * (ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH + 1)},
    ]
    for credentials in invalid_credentials:
        with pytest.raises(ActorMCPConnectionValidationError):
            update_actor_mcp_connection_credentials(
                db,
                user_id=user.id,
                resource_owner_key=ALICE_OWNER,
                app_id=app.app_id,
                expected_lifecycle_generation=original.lifecycle_generation,
                credentials=credentials,
            )
    db.expire_all()
    assert (
        get_actor_mcp_connection_metadata(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
        )
        == original
    )


def test_keyless_connection_persists_null_env(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db, app_id="chrome-devtools")

    snapshot = create_actor_mcp_connection(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        credentials=None,
    )

    secret_snapshot = get_actor_mcp_connection_snapshot(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=snapshot.lifecycle_generation,
    )
    assert secret_snapshot is not None
    assert secret_snapshot.credentials is None
    assert db.query(ActorMCPServerConnection).one().encrypted_env is None


def test_non_builtin_public_app_is_rejected(db: Session) -> None:
    user, _other = _seed_users(db)
    custom = _seed_app(db, app_id="custom-stdio")
    custom.launch_config = {
        "command": "untrusted-command",
        "required_env": ["POSTHOG_API_KEY", "POSTHOG_HOST"],
    }

    with pytest.raises(ActorMCPConnectionValidationError, match="built-in"):
        _create(
            db,
            user_id=user.id,
            owner=ALICE_OWNER,
            app_id=custom.app_id,
        )


@pytest.mark.parametrize(
    "required_env", [{"KEY": "bad"}, ["KEY", "KEY"], [""], [" KEY"], ["KEY "]]
)
def test_invalid_builtin_credential_schema_is_rejected(
    db: Session, monkeypatch, required_env
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    monkeypatch.setattr(
        "xagent.web.services.actor_mcp_connections.get_builtin_execution_fields",
        lambda _app_id: {
            "transport": "stdio",
            "launch_config": {"command": "trusted", "required_env": required_env},
        },
    )

    with pytest.raises(ActorMCPConnectionValidationError, match="credential schema"):
        _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)


def test_catalog_delete_cascades_and_recreate_does_not_revive_connection(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    db.commit()

    db.delete(app)
    db.commit()
    assert db.query(ActorMCPServerConnection).count() == 0

    replacement_app = _seed_app(db)
    assert replacement_app.generation != original.catalog_app_generation
    assert (
        get_actor_mcp_connection_snapshot(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=replacement_app.app_id,
            expected_lifecycle_generation=original.lifecycle_generation,
        )
        is None
    )


def test_old_generation_cannot_read_update_or_delete_replacement(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    assert delete_actor_mcp_connection(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=original.lifecycle_generation,
    )
    replacement = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
        credentials={
            "POSTHOG_API_KEY": "replacement-secret",
            "POSTHOG_HOST": "https://replacement.test",
        },
    )

    assert (
        get_actor_mcp_connection_snapshot(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            expected_lifecycle_generation=original.lifecycle_generation,
        )
        is None
    )
    assert (
        update_actor_mcp_connection_credentials(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            expected_lifecycle_generation=original.lifecycle_generation,
            credentials={"POSTHOG_API_KEY": "stale-writer"},
        )
        is None
    )
    assert not delete_actor_mcp_connection(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=original.lifecycle_generation,
    )
    assert (
        get_actor_mcp_connection_metadata(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
        )
        == replacement
    )


def test_update_sql_generation_fence_survives_delete_recreate_race(
    db: Session, monkeypatch
) -> None:
    import xagent.web.services.actor_mcp_connections as service

    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    replacement_generation = uuid.uuid4()
    replacement_credentials = {
        "POSTHOG_API_KEY": "replacement-secret",
        "POSTHOG_HOST": "https://replacement.test",
    }
    real_update = service.update_actor_mcp_connection_env

    def replace_before_update(session: Session, **kwargs) -> bool:
        session.execute(
            ActorMCPServerConnection.__table__.delete().where(
                ActorMCPServerConnection.user_id == user.id,
                ActorMCPServerConnection.resource_owner_key == ALICE_OWNER,
                ActorMCPServerConnection.app_id == app.app_id,
            )
        )
        session.execute(
            ActorMCPServerConnection.__table__.insert().values(
                lifecycle_generation=replacement_generation,
                user_id=user.id,
                resource_owner_key=ALICE_OWNER,
                app_id=app.app_id,
                catalog_app_generation=app.generation,
                encrypted_env=encrypt_env_dict(replacement_credentials),
            )
        )
        return real_update(session, **kwargs)

    monkeypatch.setattr(
        service, "update_actor_mcp_connection_env", replace_before_update
    )

    assert (
        update_actor_mcp_connection_credentials(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            expected_lifecycle_generation=original.lifecycle_generation,
            credentials={"POSTHOG_API_KEY": "stale-writer"},
        )
        is None
    )
    db.expire_all()
    replacement = get_actor_mcp_connection_snapshot(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=replacement_generation,
    )
    assert replacement is not None
    assert replacement.credentials == replacement_credentials


def test_create_is_idempotent_and_conflict_keeps_transaction_usable(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)

    assert (
        _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id) == original
    )
    with pytest.raises(
        ActorMCPConnectionConflictError,
        match="generation-protected update",
    ):
        _create(
            db,
            user_id=user.id,
            owner=ALICE_OWNER,
            app_id=app.app_id,
            credentials={
                "POSTHOG_API_KEY": "different",
                "POSTHOG_HOST": "https://different.test",
            },
        )

    db.add(User(username="transaction-still-usable", password_hash="x"))
    db.flush()
    assert db.query(User).filter(User.username == "transaction-still-usable").one()


@pytest.mark.parametrize("winner_uses_different_credentials", [False, True])
def test_simulated_create_race_preserves_outer_transaction(
    tmp_path, monkeypatch, winner_uses_different_credentials: bool
) -> None:
    import xagent.web.services.actor_mcp_connections as service

    engine = create_engine(f"sqlite:///{tmp_path / 'create-race.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as setup:
        user, _other = _seed_users(setup)
        app = _seed_app(setup)
        setup.commit()
        user_id = user.id
        app_id = app.app_id
        catalog_generation = app.generation

    real_lookup = service.get_current_actor_mcp_connection_for_create
    calls = 0

    def racing_lookup(session: Session, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            with Session(engine) as winner:
                winner.add(
                    ActorMCPServerConnection(
                        user_id=user_id,
                        resource_owner_key=ALICE_OWNER,
                        app_id=app_id,
                        catalog_app_generation=catalog_generation,
                        encrypted_env=encrypt_env_dict(
                            {
                                "POSTHOG_API_KEY": (
                                    "winner-secret"
                                    if winner_uses_different_credentials
                                    else "secret-value"
                                ),
                                "POSTHOG_HOST": "https://x.test",
                            }
                        ),
                    )
                )
                winner.commit()
            return None
        return real_lookup(session, **kwargs)

    monkeypatch.setattr(
        service, "get_current_actor_mcp_connection_for_create", racing_lookup
    )
    with Session(engine) as contender:
        if winner_uses_different_credentials:
            with pytest.raises(ActorMCPConnectionConflictError):
                _create(
                    contender,
                    user_id=user_id,
                    owner=ALICE_OWNER,
                    app_id=app_id,
                )
        else:
            snapshot = _create(
                contender,
                user_id=user_id,
                owner=ALICE_OWNER,
                app_id=app_id,
            )
            secret_snapshot = get_actor_mcp_connection_snapshot(
                contender,
                user_id=user_id,
                resource_owner_key=ALICE_OWNER,
                app_id=app_id,
                expected_lifecycle_generation=snapshot.lifecycle_generation,
            )
            assert secret_snapshot is not None
            assert secret_snapshot.credentials == {
                "POSTHOG_API_KEY": "secret-value",
                "POSTHOG_HOST": "https://x.test",
            }
        contender.add(User(username="outer-write", password_hash="x"))
        contender.flush()
        assert contender.query(User).filter(User.username == "outer-write").one()
    engine.dispose()


def test_nonduplicate_integrity_error_is_not_misclassified(db: Session) -> None:
    app = _seed_app(db)
    with pytest.raises(IntegrityError):
        _create(db, user_id=999_999, owner=ALICE_OWNER, app_id=app.app_id)

    user = User(username="savepoint-remains-usable", password_hash="x")
    db.add(user)
    db.flush()
    assert user.id is not None


def test_list_returns_only_metadata_without_decrypting(
    db: Session, monkeypatch
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    created = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    monkeypatch.setattr(
        "xagent.web.services.actor_mcp_connections.decrypt_env_dict_strict",
        lambda _value: (_ for _ in ()).throw(AssertionError("must not decrypt")),
    )

    listed = list_actor_mcp_connection_snapshots(
        db, user_id=user.id, resource_owner_key=ALICE_OWNER
    )
    exact = get_actor_mcp_connection_metadata(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
    )

    assert len(listed) == 1
    assert exact == listed[0]
    assert listed[0].lifecycle_generation == created.lifecycle_generation
    assert listed[0].configured_field_names == frozenset(
        {"POSTHOG_API_KEY", "POSTHOG_HOST"}
    )
    assert "secret-value" not in repr(listed[0])


def test_internal_secret_get_requires_both_exact_generations(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    created = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)

    with pytest.raises(ActorMCPConnectionNotFoundOrStaleError):
        _get_internal(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            catalog_app_generation=uuid.uuid4(),
            expected_lifecycle_generation=created.lifecycle_generation,
        )
    with pytest.raises(ActorMCPConnectionNotFoundOrStaleError):
        _get_internal(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            catalog_app_generation=created.catalog_app_generation,
            expected_lifecycle_generation=uuid.uuid4(),
        )

    exact = _get_internal(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        catalog_app_generation=created.catalog_app_generation,
        expected_lifecycle_generation=created.lifecycle_generation,
    )
    assert exact.credentials == {
        "POSTHOG_API_KEY": "secret-value",
        "POSTHOG_HOST": "https://x.test",
    }
    assert "secret-value" not in repr(exact)


def test_stale_mutations_raise_stable_not_found_error(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    created = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    stale_generation = uuid.uuid4()

    with pytest.raises(ActorMCPConnectionNotFoundOrStaleError):
        _update_exact(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            expected_lifecycle_generation=stale_generation,
            credentials={"POSTHOG_API_KEY": "stale"},
        )
    with pytest.raises(ActorMCPConnectionNotFoundOrStaleError):
        _delete_exact(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            expected_lifecycle_generation=stale_generation,
        )

    assert (
        get_actor_mcp_connection_metadata(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
        )
        == created
    )


def test_catalog_fence_filters_read_list_and_delete(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    other_app = _seed_app(db, app_id="google-maps")
    created = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    row = db.query(ActorMCPServerConnection).one()
    db.execute(
        ActorMCPServerConnection.__table__.update()
        .where(ActorMCPServerConnection.id == row.id)
        .values(catalog_app_generation=other_app.generation)
    )
    db.expire_all()

    assert (
        get_actor_mcp_connection_snapshot(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            expected_lifecycle_generation=created.lifecycle_generation,
        )
        is None
    )
    assert (
        list_actor_mcp_connection_snapshots(
            db, user_id=user.id, resource_owner_key=ALICE_OWNER
        )
        == []
    )
    assert not delete_actor_mcp_connection(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=created.lifecycle_generation,
    )


def test_exact_get_refreshes_identity_map_state(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    created = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    row = db.query(ActorMCPServerConnection).one()
    replacement = {
        "POSTHOG_API_KEY": "database-value",
        "POSTHOG_HOST": "https://database.test",
    }
    db.execute(
        ActorMCPServerConnection.__table__.update()
        .where(ActorMCPServerConnection.id == row.id)
        .values(encrypted_env=encrypt_env_dict(replacement))
        .execution_options(synchronize_session=False)
    )

    refreshed = get_actor_mcp_connection_snapshot(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=created.lifecycle_generation,
    )
    assert refreshed is not None
    assert refreshed.credentials == replacement


@pytest.mark.parametrize("transport", ["STDIO", "stdio ", " stdio"])
def test_registry_transport_must_be_exact(
    db: Session, monkeypatch, transport: str
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    monkeypatch.setattr(
        "xagent.web.services.actor_mcp_connections.get_builtin_execution_fields",
        lambda _app_id: {
            "transport": transport,
            "launch_config": {
                "command": "trusted",
                "required_env": ["POSTHOG_API_KEY", "POSTHOG_HOST"],
            },
        },
    )
    with pytest.raises(ActorMCPConnectionValidationError, match="built-in stdio"):
        _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)


def test_unpaired_surrogate_is_rejected_without_echoing_secret(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    secret = "sensitive\ud800value"

    with pytest.raises(ActorMCPConnectionValidationError) as exc_info:
        _create(
            db,
            user_id=user.id,
            owner=ALICE_OWNER,
            app_id=app.app_id,
            credentials={
                "POSTHOG_API_KEY": secret,
                "POSTHOG_HOST": "https://x.test",
            },
        )
    assert "sensitive" not in str(exc_info.value)
    assert db.query(ActorMCPServerConnection).count() == 0


def test_wrong_key_and_corrupt_ciphertext_map_to_redacted_domain_error(
    db: Session,
) -> None:
    from cryptography.fernet import Fernet

    user, _other = _seed_users(db)
    app = _seed_app(db)
    created = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    wrong_key_token = (
        Fernet(Fernet.generate_key()).encrypt(b"never-expose-this").decode()
    )
    valid_token = encrypt_env_dict({"key": "never-expose-this"})["key"]
    corrupt_token = (
        f"{valid_token[:20]}{'A' if valid_token[20] != 'A' else 'B'}{valid_token[21:]}"
    )
    row = db.query(ActorMCPServerConnection).one()
    for token in (wrong_key_token, corrupt_token):
        db.execute(
            ActorMCPServerConnection.__table__.update()
            .where(ActorMCPServerConnection.id == row.id)
            .values(encrypted_env={"POSTHOG_API_KEY": token})
        )
        db.expire_all()

        with pytest.raises(ActorMCPConnectionCredentialCorruptionError) as exc_info:
            get_actor_mcp_connection_snapshot(
                db,
                user_id=user.id,
                resource_owner_key=ALICE_OWNER,
                app_id=app.app_id,
                expected_lifecycle_generation=created.lifecycle_generation,
            )
        assert token not in str(exc_info.value)
        assert "never-expose-this" not in repr(exc_info.value)


def test_plaintext_at_rest_is_treated_as_credential_corruption(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    created = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    row = db.query(ActorMCPServerConnection).one()
    db.execute(
        ActorMCPServerConnection.__table__.update()
        .where(ActorMCPServerConnection.id == row.id)
        .values(encrypted_env={"POSTHOG_API_KEY": "plaintext-secret"})
    )
    db.expire_all()

    with pytest.raises(ActorMCPConnectionCredentialCorruptionError) as exc_info:
        get_actor_mcp_connection_snapshot(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            expected_lifecycle_generation=created.lifecycle_generation,
        )
    assert "plaintext-secret" not in str(exc_info.value)


def test_token_shaped_input_round_trips_as_exact_plaintext(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    token_shaped_plaintext = get_cipher().encrypt(b"nested").decode()
    created = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
        credentials={
            "POSTHOG_API_KEY": token_shaped_plaintext,
            "POSTHOG_HOST": "https://x.test",
        },
    )

    exact = get_actor_mcp_connection_snapshot(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        expected_lifecycle_generation=created.lifecycle_generation,
    )
    assert exact is not None
    assert exact.credentials is not None
    assert exact.credentials["POSTHOG_API_KEY"] == token_shaped_plaintext


@pytest.mark.parametrize("field", ["app_id", "catalog_app_generation"])
def test_catalog_binding_is_immutable(db: Session, field: str) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    other_app = _seed_app(db, app_id="google-maps")
    _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    row = db.query(ActorMCPServerConnection).one()
    db.commit()
    setattr(
        row,
        field,
        other_app.app_id if field == "app_id" else other_app.generation,
    )

    with pytest.raises(ValueError, match="catalog binding"):
        db.flush()


def test_update_rejects_stale_catalog_generation(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    other_app = _seed_app(db, app_id="google-maps")
    original = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    row = db.query(ActorMCPServerConnection).one()
    db.execute(
        ActorMCPServerConnection.__table__.update()
        .where(ActorMCPServerConnection.id == row.id)
        .values(catalog_app_generation=other_app.generation)
    )
    db.expire_all()

    assert (
        update_actor_mcp_connection_credentials(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            expected_lifecycle_generation=original.lifecycle_generation,
            credentials={"POSTHOG_API_KEY": "rotated"},
        )
        is None
    )
