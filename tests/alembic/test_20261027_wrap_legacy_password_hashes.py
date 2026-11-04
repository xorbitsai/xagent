"""Migration coverage for wrapping legacy SHA-256 password hashes in bcrypt."""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.api.auth import hash_password, needs_password_rehash, verify_password

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261027_wrap_legacy_password_hashes.py"
)
REVISION = "20261027_wrap_legacy_password_hashes"
DOWN_REVISION = "20261027_employment_hero_leave"

LEGACY_PASSWORD = "legacy-password"
UPPERCASE_PASSWORD = "uppercase-legacy-password"
BCRYPT_PASSWORD = "bcrypt-password"
UNUSABLE_MARKER = "!"
# 64 characters but not hex, so it must not be mistaken for a legacy digest.
NON_HEX_64 = "z" * 64


def _migration():
    return load_migration_module(MIGRATION_PATH, "wrap_legacy_password_migration")


def _users(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        "users",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("username", sa.String(100), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
    )


def _sha256_hex(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _hashes(connection, table: sa.Table) -> dict[str, str]:
    return {
        row.username: row.password_hash
        for row in connection.execute(
            sa.select(table.c.username, table.c.password_hash)
        )
    }


def _exercise_upgrade(engine: sa.engine.Engine) -> None:
    migration = _migration()
    metadata = sa.MetaData()
    table = _users(metadata)
    bcrypt_hash = hash_password(BCRYPT_PASSWORD)
    try:
        metadata.create_all(engine)
        with engine.begin() as connection:
            connection.execute(
                sa.insert(table),
                [
                    {
                        "username": "legacy",
                        "password_hash": _sha256_hex(LEGACY_PASSWORD),
                    },
                    {
                        "username": "legacy-upper",
                        "password_hash": _sha256_hex(UPPERCASE_PASSWORD).upper(),
                    },
                    {"username": "bcrypt", "password_hash": bcrypt_hash},
                    {"username": "unusable", "password_hash": UNUSABLE_MARKER},
                    {"username": "non-hex", "password_hash": NON_HEX_64},
                ],
            )

            # A batch size smaller than the row count exercises pagination.
            with (
                patch.object(
                    migration, "op", Operations(MigrationContext.configure(connection))
                ),
                patch.object(migration, "BATCH_SIZE", 2),
            ):
                migration.upgrade()
                after_first = _hashes(connection, table)
                migration.upgrade()
                after_second = _hashes(connection, table)
                migration.downgrade()
                after_downgrade = _hashes(connection, table)
    finally:
        metadata.drop_all(engine)
        engine.dispose()

    for username, password in (
        ("legacy", LEGACY_PASSWORD),
        ("legacy-upper", UPPERCASE_PASSWORD),
    ):
        wrapped = after_first[username]
        assert wrapped.startswith("$2b$12$")
        assert not needs_password_rehash(wrapped)
        assert verify_password(password, wrapped) is True
        assert verify_password("wrong-password", wrapped) is False

    assert after_first["bcrypt"] == bcrypt_hash
    assert verify_password(BCRYPT_PASSWORD, after_first["bcrypt"]) is True
    assert after_first["unusable"] == UNUSABLE_MARKER
    assert after_first["non-hex"] == NON_HEX_64

    # Re-running is a no-op and downgrade does not restore unsalted digests.
    assert after_second == after_first
    assert after_downgrade == after_first


def test_revision_metadata() -> None:
    migration = _migration()
    assert migration.revision == REVISION
    assert migration.down_revision == DOWN_REVISION


def test_revision_is_single_head() -> None:
    script = ScriptDirectory.from_config(
        create_alembic_config(sa.create_engine("sqlite://"))
    )
    heads = script.get_heads()
    assert len(heads) == 1
    assert REVISION in {
        rev.revision for rev in script.iterate_revisions(heads[0], "base")
    }


def test_sqlite_wraps_legacy_hashes_only() -> None:
    _exercise_upgrade(sa.create_engine("sqlite://"))


def test_upgrade_skips_schema_without_users() -> None:
    migration = _migration()
    engine = sa.create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            with patch.object(
                migration, "op", Operations(MigrationContext.configure(connection))
            ):
                migration.upgrade()
            assert sa.inspect(connection).get_table_names() == []
    finally:
        engine.dispose()


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_password_wrap") as make:
        yield make


@pytest.mark.postgresql
def test_postgresql_wraps_legacy_hashes_only(postgresql_engine_factory) -> None:
    _exercise_upgrade(postgresql_engine_factory("upgrade"))
