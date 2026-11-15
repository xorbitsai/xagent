"""Migration ``20261116_credential_revocation_outbox`` on SQLite and PostgreSQL.

The triggers reach a database two ways: this migration installs them on an
existing database, and ``models.credential_revocation`` installs them when
``create_all`` creates the tables (a fresh database is stamped at head and
never runs the migration). Each carries its own copy of the SQL, so the
tests here prove the two installations are the same, and that a migrated
database records a deletion.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import sqlalchemy as sa
from alembic import command

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.db.config import create_alembic_config
from xagent.web.models.database import Base

REVISION = "20261116_credential_revocation_outbox"
DOWN_REVISION = "20261115_user_oauth_issuer"
TRIGGERS = {
    "capture_user_oauth_revocation",
    "capture_user_oauth_token_drop",
    "capture_mcp_oauth_grant_revocation_on_delete",
    "capture_mcp_oauth_grant_revocation_on_revoke",
    "delete_mcp_oauth_client_grants_first",
}
FUNCTIONS = (
    "capture_user_oauth_revocation",
    "capture_mcp_oauth_grant_revocation",
    "delete_mcp_oauth_client_grants_first",
)


def _triggers(connection: sa.Connection) -> dict[str, str]:
    """Every outbox trigger (and PostgreSQL trigger function) by name."""
    if connection.dialect.name == "sqlite":
        rows = connection.execute(
            sa.text("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'")
        )
        return dict(rows.tuples().all())
    triggers = dict(
        connection.execute(
            sa.text(
                "SELECT tgname, pg_get_triggerdef(oid) FROM pg_trigger "
                "WHERE NOT tgisinternal"
            )
        )
        .tuples()
        .all()
    )
    functions = dict(
        connection.execute(
            sa.text("SELECT proname, prosrc FROM pg_proc WHERE proname = ANY(:names)"),
            {"names": list(FUNCTIONS)},
        )
        .tuples()
        .all()
    )
    return {**triggers, **{f"function {k}": v for k, v in functions.items()}}


def _columns(connection: sa.Connection) -> list[tuple[str, str, bool]]:
    return [
        (column["name"], str(column["type"]), column["nullable"])
        for column in sa.inspect(connection).get_columns("credential_revocations")
    ]


def _strip_outbox(connection: sa.Connection) -> None:
    """Turn a create_all database back into the state before this revision."""
    for trigger, table in (
        ("capture_user_oauth_revocation", "user_oauth"),
        ("capture_user_oauth_token_drop", "user_oauth"),
        ("capture_mcp_oauth_grant_revocation_on_delete", "mcp_oauth_grants"),
        ("capture_mcp_oauth_grant_revocation_on_revoke", "mcp_oauth_grants"),
        ("delete_mcp_oauth_client_grants_first", "mcp_oauth_clients"),
    ):
        on = f" ON {table}" if connection.dialect.name == "postgresql" else ""
        connection.execute(sa.text(f"DROP TRIGGER {trigger}{on}"))
    if connection.dialect.name == "postgresql":
        for function in FUNCTIONS:
            connection.execute(sa.text(f"DROP FUNCTION {function}()"))
    connection.execute(sa.text("DROP TABLE credential_revocations"))
    connection.execute(
        sa.text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
    )
    connection.execute(
        sa.text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def _assert_migration_matches_models(make_engine: Callable[[str], sa.Engine]) -> None:
    models = make_engine("models")
    Base.metadata.create_all(models)
    with models.connect() as connection:
        expected_triggers = _triggers(connection)
        expected_columns = _columns(connection)
    assert TRIGGERS <= expected_triggers.keys()

    migrated = make_engine("migrated")
    Base.metadata.create_all(migrated)
    config = create_alembic_config(migrated)
    # Alembic's env commits on the caller's connection, so this test commits
    # its own steps too instead of wrapping everything in one transaction.
    with migrated.connect() as connection:
        _strip_outbox(connection)
        connection.commit()
        assert not TRIGGERS & _triggers(connection).keys()
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        assert _triggers(connection) == expected_triggers
        assert _columns(connection) == expected_columns

        # It records: a deletion on the migrated database.
        connection.execute(
            sa.text(
                "INSERT INTO users (id, username, password_hash, is_admin) "
                "VALUES (1, 'alice', 'x', false)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO user_oauth (id, user_id, provider, oauth_provider, "
                "provider_user_id, access_token) "
                "VALUES (1, 1, 'gmail', 'google', 'subject', 'ciphertext')"
            )
        )
        connection.execute(sa.text("DELETE FROM user_oauth"))
        assert connection.execute(
            sa.text("SELECT kind, grant_key, access_token FROM credential_revocations")
        ).one() == ("user_oauth", "user_oauth:google:subject", "ciphertext")
        connection.commit()

        command.downgrade(config, DOWN_REVISION)
        assert not TRIGGERS & _triggers(connection).keys()
        assert not sa.inspect(connection).has_table("credential_revocations")
        connection.execute(sa.text("DELETE FROM users"))
        connection.commit()

        command.upgrade(config, REVISION)
        assert _triggers(connection) == expected_triggers


def test_sqlite_migration_installs_what_the_models_install(tmp_path) -> None:
    _assert_migration_matches_models(
        lambda name: sa.create_engine(f"sqlite:///{tmp_path / name}.db")
    )


@pytest.mark.postgresql
def test_postgresql_migration_installs_what_the_models_install() -> None:
    with disposable_database_factory("xagent_credential_revocations") as make:
        _assert_migration_matches_models(make)


def _assert_upgrade_retries_a_revoked_grant(
    make_engine: Callable[[str], sa.Engine], monkeypatch
) -> None:
    """A grant was marked revoked before this revision while its provider
    revocation failed, so its tokens are still live. The upgrade must leave
    a receipt for it, and the worker then revokes it."""
    import asyncio
    from datetime import datetime, timedelta, timezone

    import httpx
    from sqlalchemy.orm import sessionmaker

    from xagent.core.utils.encryption import decrypt_value_strict, encrypt_value
    from xagent.web.models.credential_revocation import CredentialRevocation
    from xagent.web.models.mcp import MCPServer
    from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
    from xagent.web.models.user import User
    from xagent.web.services import mcp_oauth as mcp_oauth_service
    from xagent.web.services.credential_revocations import (
        run_credential_revocation_batch,
    )

    engine = make_engine("cutover")
    Base.metadata.create_all(engine)
    with engine.connect() as connection:
        _strip_outbox(connection)
        connection.commit()
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="alice", password_hash="x", is_admin=False)
        server = MCPServer.from_config(
            {
                "name": "records",
                "managed": "external",
                "transport": "streamable_http",
                "url": "https://mcp.example.com/mcp",
                "auth": {"type": "mcp_oauth"},
            }
        )
        db.add_all([user, server])
        db.flush()
        client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="mcp-client",
            token_endpoint_auth_method="none",
            redirect_uri="https://app.example.com/api/mcp/oauth/callback",
            metadata_json={"revocation_endpoint": "https://auth.example.com/revoke"},
        )
        db.add(client)
        db.flush()
        for status, token in (("revoked", "still-live"), ("active", "in-use")):
            db.add(
                MCPOAuthGrant(
                    mcp_server_id=server.id,
                    user_id=user.id,
                    mcp_oauth_client_id=client.id,
                    resource_owner_key=f"xagent:user:{user.id}:{status}",
                    issuer="https://auth.example.com",
                    resource="https://mcp.example.com/mcp",
                    scope="records.read",
                    access_token=encrypt_value(f"{token}-access"),
                    refresh_token=encrypt_value(f"{token}-refresh"),
                    status=status,
                )
            )
        db.commit()

    config = create_alembic_config(engine)
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        connection.commit()

    with factory() as db:
        [receipt] = db.query(CredentialRevocation).all()
        assert receipt.status == "pending"
        assert decrypt_value_strict(receipt.access_token) == "still-live-access"
        assert receipt.client_id == "mcp-client"
        assert receipt.revocation_endpoint == "https://auth.example.com/revoke"

    sent: list[str] = []
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(httpx.QueryParams(request.content.decode())["token"])
        return httpx.Response(200)

    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **_: real_async_client(transport=httpx.MockTransport(handler)),
    )
    report = asyncio.run(
        run_credential_revocation_batch(
            factory, now=lambda: datetime.now(timezone.utc) + timedelta(seconds=1)
        )
    )
    assert report.outcomes == {"revoked": 1}
    assert sent == ["still-live-access", "still-live-refresh"]


def test_sqlite_upgrade_retries_a_grant_revoked_before_it(tmp_path, monkeypatch):
    _assert_upgrade_retries_a_revoked_grant(
        lambda name: sa.create_engine(f"sqlite:///{tmp_path / name}.db"), monkeypatch
    )


@pytest.mark.postgresql
def test_postgresql_upgrade_retries_a_grant_revoked_before_it(monkeypatch):
    with disposable_database_factory("xagent_credential_revocation_cutover") as make:
        _assert_upgrade_retries_a_revoked_grant(make, monkeypatch)


# --- legacy and partial user_oauth shapes -------------------------------------

#: A user_oauth older than ``token_type``: it holds token columns but no
#: Composio row can exist in it.
_LEGACY_USER_OAUTH = (
    "CREATE TABLE user_oauth (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, "
    "provider VARCHAR(50) NOT NULL, oauth_provider VARCHAR(50), "
    "provider_user_id VARCHAR, access_token VARCHAR NOT NULL, refresh_token VARCHAR)"
)
#: The stand-in some downstream suites create: no token column at all.
_PARTIAL_USER_OAUTH = (
    "CREATE TABLE user_oauth (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, "
    "resource_owner_key VARCHAR(512))"
)


def _legacy_schema(connection: sa.Connection, user_oauth: str) -> None:
    connection.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
    connection.execute(sa.text(user_oauth))
    connection.execute(
        sa.text(
            "CREATE TABLE oauth_providers (id INTEGER PRIMARY KEY, "
            "provider_name VARCHAR(50), client_id VARCHAR, client_secret VARCHAR)"
        )
    )


def _user_oauth_still_alters_and_deletes(connection: sa.Connection) -> list[str]:
    """Alter the table (SQLite re-validates every trigger), delete a row, and
    return the kinds of the records that delete wrote."""
    connection.execute(sa.text("ALTER TABLE user_oauth ADD COLUMN later VARCHAR"))
    connection.execute(sa.text("INSERT INTO users (id) VALUES (1)"))
    columns = {c["name"] for c in sa.inspect(connection).get_columns("user_oauth")}
    if "access_token" in columns:
        connection.execute(
            sa.text(
                "INSERT INTO user_oauth (id, user_id, provider, oauth_provider, "
                "provider_user_id, access_token) "
                "VALUES (1, 1, 'gmail', 'google', 'sub', 'ciphertext')"
            )
        )
    else:
        connection.execute(
            sa.text("INSERT INTO user_oauth (id, user_id) VALUES (1, 1)")
        )
    connection.execute(sa.text("DELETE FROM user_oauth"))
    return list(
        connection.execute(sa.text("SELECT kind FROM credential_revocations")).scalars()
    )


def _assert_legacy_shape(
    engine: sa.Engine, path: str, user_oauth: str, recorded: list[str]
) -> None:
    from xagent.web.models.credential_revocation import (
        CredentialRevocation,
        install_missing_credential_revocation_triggers,
    )

    with engine.connect() as connection:
        _legacy_schema(connection, user_oauth)
        if path == "create_all":
            CredentialRevocation.__table__.create(connection)
            install_missing_credential_revocation_triggers(connection)
        else:
            connection.execute(
                sa.text("CREATE TABLE alembic_version (version_num VARCHAR(255))")
            )
            connection.execute(
                sa.text("INSERT INTO alembic_version VALUES (:revision)"),
                {"revision": DOWN_REVISION},
            )
        connection.commit()
        if path == "migration":
            config = create_alembic_config(engine)
            config.attributes["connection"] = connection
            command.upgrade(config, REVISION)
            connection.commit()
        assert _user_oauth_still_alters_and_deletes(connection) == recorded


_LEGACY_CASES = pytest.mark.parametrize(
    ("user_oauth", "recorded"),
    [(_LEGACY_USER_OAUTH, ["user_oauth"]), (_PARTIAL_USER_OAUTH, [])],
    ids=["without-token-type", "without-token-columns"],
)
_PATHS = pytest.mark.parametrize("path", ["create_all", "migration"])


@_LEGACY_CASES
@_PATHS
def test_sqlite_triggers_fit_older_user_oauth_shapes(
    tmp_path, path, user_oauth, recorded
) -> None:
    """A user_oauth without ``token_type`` gets triggers that treat every row
    as native; one without token columns gets none. Either way later schema
    changes and deletes keep working."""
    _assert_legacy_shape(
        sa.create_engine(f"sqlite:///{tmp_path / 'legacy.db'}"),
        path,
        user_oauth,
        recorded,
    )


@pytest.mark.postgresql
@_LEGACY_CASES
@_PATHS
def test_postgresql_triggers_fit_older_user_oauth_shapes(
    path, user_oauth, recorded
) -> None:
    with disposable_database_factory("xagent_credential_revocation_legacy") as make:
        _assert_legacy_shape(make("legacy"), path, user_oauth, recorded)


@pytest.mark.postgresql
def test_postgresql_each_schema_gets_its_own_triggers() -> None:
    """Trigger names are per table in PostgreSQL. A second schema of the same
    database must not see the first schema's triggers as its own and skip
    installing them."""
    from xagent.web.models.credential_revocation import CREDENTIAL_REVOCATION_TRIGGERS

    with disposable_database_factory("xagent_credential_revocation_schemas") as make:
        admin = make("schemas")
        with admin.begin() as connection:
            connection.execute(sa.text("CREATE SCHEMA first"))
            connection.execute(sa.text("CREATE SCHEMA second"))
        installed: dict[str, set[str]] = {}
        for schema in ("first", "second"):
            engine = sa.create_engine(
                admin.url,
                creator=admin.pool._creator,  # same credentials, own search_path
            )

            @sa.event.listens_for(engine, "connect")
            def _search_path(dbapi_connection, _record, schema=schema):  # type: ignore[no-untyped-def]
                with dbapi_connection.cursor() as cursor:
                    cursor.execute(f"SET search_path TO {schema}")

            Base.metadata.create_all(engine)
            with engine.connect() as connection:
                installed[schema] = set(
                    connection.execute(
                        sa.text(
                            "SELECT trigger.tgname FROM pg_trigger AS trigger "
                            "JOIN pg_class AS relation "
                            "ON relation.oid = trigger.tgrelid "
                            "JOIN pg_namespace AS namespace "
                            "ON namespace.oid = relation.relnamespace "
                            "WHERE NOT trigger.tgisinternal "
                            "AND namespace.nspname = :schema"
                        ),
                        {"schema": schema},
                    ).scalars()
                )
            engine.dispose()
    expected = {trigger.name for trigger in CREDENTIAL_REVOCATION_TRIGGERS}
    assert installed == {"first": expected, "second": expected}
