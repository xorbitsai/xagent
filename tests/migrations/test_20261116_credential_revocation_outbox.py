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
