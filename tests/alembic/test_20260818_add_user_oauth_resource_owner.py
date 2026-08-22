"""Tests for actor-aware builtin OAuth storage identity."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

MIGRATION_NAME = "20260818_add_user_oauth_resource_owner.py"
ORDINARY_INDEX = "uq_user_oauth_ordinary_account"
ACTOR_INDEX = "uq_user_oauth_actor_account"
REMOVED_LOOKUP_INDEX = "ix_user_oauth_owner_provider"
OLD_CONSTRAINT = "uq_user_provider_account"
OWNER_COLUMN = "resource_owner_key"


def _migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions"
        / MIGRATION_NAME
    )
    spec = importlib.util.spec_from_file_location(
        "user_oauth_actor_migration", migration_file
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_users_table(connection) -> None:
    _operations(connection).create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def _create_old_table(
    connection,
    *,
    create_users: bool = True,
    include_user_fk: bool = True,
) -> None:
    operations = _operations(connection)
    if create_users:
        _create_users_table(connection)
    foreign_keys = (
        (sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),)
        if include_user_fk
        else ()
    )
    operations.create_table(
        "user_oauth",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("access_token", sa.String(), nullable=False),
        sa.Column("refresh_token", sa.String(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("token_type", sa.String(50), nullable=True),
        sa.Column("scope", sa.String(), nullable=True),
        sa.Column("provider_user_id", sa.String(), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        *foreign_keys,
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "provider",
            "provider_user_id",
            name=OLD_CONSTRAINT,
        ),
    )
    operations.create_index("ix_user_oauth_id", "user_oauth", ["id"])
    operations.create_table(
        "gmail_watch_states",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("oauth_account_id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("history_id", sa.String(255), nullable=False),
        sa.Column("watch_expiration", sa.DateTime(timezone=True), nullable=True),
        sa.Column("topic_name", sa.String(512), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("callback_id", sa.String(128), nullable=True),
        sa.Column("push_audience", sa.Text(), nullable=True),
        sa.Column("previous_push_audience", sa.Text(), nullable=True),
        sa.Column(
            "previous_push_audience_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("subscription_name", sa.String(512), nullable=True),
        sa.Column("status", sa.String(32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["oauth_account_id"], ["user_oauth.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("oauth_account_id"),
    )


def _create_interrupted_owner_table(
    connection,
    *,
    existing_indexes: tuple[str, ...] = (),
    include_user_fk: bool = True,
) -> None:
    """Create the exact SQLite shape left by interrupted index installation."""
    operations = _operations(connection)
    _create_users_table(connection)
    operations.create_table(
        "user_oauth",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("access_token", sa.String(), nullable=False),
        sa.Column("provider_user_id", sa.String(), nullable=True),
        sa.Column("resource_owner_key", sa.String(512), nullable=True),
        *(
            (sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),)
            if include_user_fk
            else ()
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    connection.execute(text("INSERT INTO users (id) VALUES (7)"))
    connection.execute(
        text(
            "INSERT INTO user_oauth "
            "(id, user_id, provider, access_token, provider_user_id) "
            "VALUES (1, 7, 'gmail', 'ordinary', 'provider-account')"
        )
    )
    if ORDINARY_INDEX in existing_indexes:
        connection.execute(
            text(
                f"CREATE UNIQUE INDEX {ORDINARY_INDEX} ON user_oauth "
                "(user_id, provider, provider_user_id) "
                "WHERE resource_owner_key IS NULL"
            )
        )
    if ACTOR_INDEX in existing_indexes:
        connection.execute(
            text(
                f"CREATE UNIQUE INDEX {ACTOR_INDEX} ON user_oauth "
                "(user_id, resource_owner_key, provider, provider_user_id) "
                "WHERE resource_owner_key IS NOT NULL"
            )
        )


def _index_map(connection) -> dict[str, dict]:
    return {
        index["name"]: index for index in inspect(connection).get_indexes("user_oauth")
    }


def _where(index: dict) -> str:
    options = index.get("dialect_options") or {}
    clause = options.get("sqlite_where")
    return str(clause if clause is not None else "").lower()


def test_upgrade_rejects_leftover_sqlite_batch_table_before_rebuild(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-batch-temp.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_old_table(connection)
        connection.execute(text("INSERT INTO users (id) VALUES (7)"))
        connection.execute(
            text(
                "INSERT INTO user_oauth "
                "(id, user_id, provider, access_token, provider_user_id) "
                "VALUES (1, 7, 'gmail', 'ordinary', 'provider-account')"
            )
        )
        connection.execute(
            text("CREATE TABLE _alembic_tmp_user_oauth (id INTEGER PRIMARY KEY)")
        )

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="temporary table"):
                migration.upgrade()

        assert (
            connection.execute(
                text("SELECT access_token FROM user_oauth WHERE id = 1")
            ).scalar_one()
            == "ordinary"
        )


def test_upgrade_rejects_orphan_sqlite_batch_table_before_missing_table_return(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-orphan-temp.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE _alembic_tmp_user_oauth "
                "(id INTEGER PRIMARY KEY, access_token TEXT)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO _alembic_tmp_user_oauth (id, access_token) "
                "VALUES (1, 'stranded')"
            )
        )

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="temporary table"):
                migration.upgrade()

        assert (
            connection.execute(
                text("SELECT access_token FROM _alembic_tmp_user_oauth WHERE id = 1")
            ).scalar_one()
            == "stranded"
        )


def test_downgrade_rejects_orphan_sqlite_batch_table_before_missing_table_return(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-down-orphan-temp.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE _alembic_tmp_user_oauth "
                "(id INTEGER PRIMARY KEY, access_token TEXT)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO _alembic_tmp_user_oauth (id, access_token) "
                "VALUES (1, 'stranded')"
            )
        )

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="temporary table"):
                migration.downgrade()

        assert (
            connection.execute(
                text("SELECT access_token FROM _alembic_tmp_user_oauth WHERE id = 1")
            ).scalar_one()
            == "stranded"
        )


@pytest.mark.parametrize(
    "existing_indexes",
    [
        (),
        (ORDINARY_INDEX,),
        (ACTOR_INDEX,),
    ],
)
def test_upgrade_repairs_interrupted_owner_index_installation(
    tmp_path, existing_indexes: tuple[str, ...]
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-interrupted.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_interrupted_owner_table(
            connection,
            existing_indexes=existing_indexes,
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        indexes = _index_map(connection)
        assert tuple(indexes[ORDINARY_INDEX]["column_names"]) == (
            "user_id",
            "provider",
            "provider_user_id",
        )
        assert "resource_owner_key is null" in _where(indexes[ORDINARY_INDEX])
        assert tuple(indexes[ACTOR_INDEX]["column_names"]) == (
            "user_id",
            "resource_owner_key",
            "provider",
            "provider_user_id",
        )
        assert "resource_owner_key is not null" in _where(indexes[ACTOR_INDEX])
        assert (
            connection.execute(
                text("SELECT access_token FROM user_oauth WHERE id = 1")
            ).scalar_one()
            == "ordinary"
        )


@pytest.mark.parametrize(
    ("existing_index", "missing_index"),
    [
        (ORDINARY_INDEX, ACTOR_INDEX),
        (ACTOR_INDEX, ORDINARY_INDEX),
    ],
)
def test_interrupted_owner_index_repair_rejects_missing_name_collision(
    tmp_path,
    existing_index: str,
    missing_index: str,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-repair-collision.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_interrupted_owner_table(
            connection,
            existing_indexes=(existing_index,),
        )
        connection.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))
        connection.execute(text(f"CREATE INDEX {missing_index} ON unrelated (id)"))

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="already exist"):
                migration.upgrade()

        indexes = _index_map(connection)
        assert existing_index in indexes
        assert missing_index not in indexes
        assert (
            connection.execute(text("SELECT count(*) FROM user_oauth")).scalar_one()
            == 1
        )


def test_interrupted_owner_schema_rejects_missing_user_cascade_foreign_key(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-interrupted-no-fk.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_interrupted_owner_table(connection, include_user_fk=False)

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="missing its user cascade"):
                migration.upgrade()


def test_interrupted_owner_index_repair_rejects_duplicate_identity(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-interrupted-duplicate.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_interrupted_owner_table(connection)
        connection.execute(
            text(
                "INSERT INTO user_oauth "
                "(id, user_id, provider, access_token, provider_user_id) "
                "VALUES (2, 7, 'gmail', 'duplicate', 'provider-account')"
            )
        )

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(IntegrityError):
                migration.upgrade()

        assert ORDINARY_INDEX not in _index_map(connection)
        assert (
            connection.execute(text("SELECT count(*) FROM user_oauth")).scalar_one()
            == 2
        )


def test_upgrade_requires_users_table_for_the_cascade_contract(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-missing-users.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_old_table(
            connection,
            create_users=False,
            include_user_fk=False,
        )

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="requires the users table"):
                migration.upgrade()


def test_upgrade_installs_missing_user_cascade_foreign_key(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-missing-user-fk.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_old_table(connection, include_user_fk=False)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        foreign_keys = inspect(connection).get_foreign_keys("user_oauth")
        assert any(
            tuple(foreign_key.get("constrained_columns") or ()) == ("user_id",)
            and foreign_key.get("referred_table") == "users"
            and tuple(foreign_key.get("referred_columns") or ()) == ("id",)
            and str((foreign_key.get("options") or {}).get("ondelete")).upper()
            == "CASCADE"
            for foreign_key in foreign_keys
        )


def test_upgrade_preserves_rows_and_installs_owner_aware_identity(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_old_table(connection)
        connection.execute(text("INSERT INTO users (id) VALUES (7)"))
        connection.execute(
            text(
                "INSERT INTO user_oauth "
                "(id, user_id, provider, access_token, refresh_token, expires_at, "
                "token_type, scope, provider_user_id, email, updated_at) "
                "VALUES (1, 7, 'gmail', 'ordinary', 'refresh', "
                "'2030-01-02 03:04:05', 'Bearer', 'gmail.read', "
                "'provider-account', 'owner@example.com', '2029-01-01 00:00:00')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO gmail_watch_states "
                "(id, user_id, oauth_account_id, email, history_id, topic_name) "
                "VALUES (11, 7, 1, 'owner@example.com', 'history-1', 'topic-1')"
            )
        )
        before = inspect(connection)
        legacy_columns = {
            column["name"]: column for column in before.get_columns("user_oauth")
        }
        legacy_user_fks = before.get_foreign_keys("user_oauth")
        legacy_watch_fks = before.get_foreign_keys("gmail_watch_states")

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        inspector = inspect(connection)
        columns = {
            column["name"]: column for column in inspector.get_columns("user_oauth")
        }
        assert columns["resource_owner_key"]["nullable"] is True
        assert columns["resource_owner_key"]["type"].length == 512
        for name, legacy in legacy_columns.items():
            current = columns[name]
            assert str(current["type"]) == str(legacy["type"])
            assert current["nullable"] == legacy["nullable"]
            assert current["default"] == legacy["default"]
        assert inspector.get_foreign_keys("user_oauth") == legacy_user_fks
        assert inspector.get_foreign_keys("gmail_watch_states") == legacy_watch_fks
        watch_fk = next(
            fk
            for fk in inspector.get_foreign_keys("gmail_watch_states")
            if fk["referred_table"] == "user_oauth"
        )
        assert watch_fk["options"].get("ondelete") == "CASCADE"
        assert connection.execute(
            text("SELECT id, oauth_account_id FROM gmail_watch_states")
        ).all() == [(11, 1)]
        assert (
            connection.execute(
                text("SELECT resource_owner_key FROM user_oauth WHERE id = 1")
            ).scalar_one()
            is None
        )

        constraints = {
            constraint["name"]
            for constraint in inspector.get_unique_constraints("user_oauth")
        }
        assert OLD_CONSTRAINT not in constraints

        indexes = _index_map(connection)
        assert tuple(indexes[ORDINARY_INDEX]["column_names"]) == (
            "user_id",
            "provider",
            "provider_user_id",
        )
        assert indexes[ORDINARY_INDEX]["unique"] == 1
        assert "resource_owner_key is null" in _where(indexes[ORDINARY_INDEX])
        assert tuple(indexes[ACTOR_INDEX]["column_names"]) == (
            "user_id",
            "resource_owner_key",
            "provider",
            "provider_user_id",
        )
        assert indexes[ACTOR_INDEX]["unique"] == 1
        assert "resource_owner_key is not null" in _where(indexes[ACTOR_INDEX])
        assert REMOVED_LOOKUP_INDEX not in indexes

        connection.execute(
            text(
                "INSERT INTO user_oauth "
                "(user_id, provider, access_token, provider_user_id, resource_owner_key) "
                "VALUES "
                "(7, 'gmail', 'alice', 'provider-account', 'toby:slack:41:UALICE'), "
                "(7, 'gmail', 'bob', 'provider-account', 'toby:slack:41:UBOB')"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO user_oauth "
                    "(user_id, provider, access_token, provider_user_id) "
                    "VALUES (7, 'gmail', 'duplicate', 'provider-account')"
                )
            )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO user_oauth "
                    "(user_id, provider, access_token, provider_user_id, resource_owner_key) "
                    "VALUES (7, 'gmail', 'duplicate', 'provider-account', "
                    "'toby:slack:41:UALICE')"
                )
            )


def test_upgrade_preserves_nullable_provider_identity_semantics(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-null.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_old_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        connection.execute(
            text(
                "INSERT INTO user_oauth "
                "(user_id, provider, access_token, provider_user_id, resource_owner_key) "
                "VALUES "
                "(7, 'gmail', 'ordinary-1', NULL, NULL), "
                "(7, 'gmail', 'ordinary-2', NULL, NULL), "
                "(7, 'gmail', 'actor-1', NULL, 'toby:slack:41:UALICE'), "
                "(7, 'gmail', 'actor-2', NULL, 'toby:slack:41:UALICE')"
            )
        )

        assert (
            connection.execute(text("SELECT count(*) FROM user_oauth")).scalar_one()
            == 4
        )


def test_downgrade_restores_ordinary_schema_before_actor_rows_exist(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-down.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_old_table(connection)
        connection.execute(
            text(
                "INSERT INTO user_oauth "
                "(id, user_id, provider, access_token, provider_user_id) "
                "VALUES (1, 7, 'gmail', 'ordinary', 'provider-account')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        inspector = inspect(connection)
        columns = {column["name"] for column in inspector.get_columns("user_oauth")}
        assert "resource_owner_key" not in columns
        constraints = {
            constraint["name"]
            for constraint in inspector.get_unique_constraints("user_oauth")
        }
        assert OLD_CONSTRAINT in constraints
        indexes = _index_map(connection)
        assert ORDINARY_INDEX not in indexes
        assert ACTOR_INDEX not in indexes
        assert REMOVED_LOOKUP_INDEX not in indexes
        assert (
            connection.execute(
                text("SELECT access_token FROM user_oauth WHERE id = 1")
            ).scalar_one()
            == "ordinary"
        )


def test_downgrade_refuses_to_collapse_actor_owned_rows(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-refuse-down.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_old_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "INSERT INTO user_oauth "
                    "(user_id, provider, access_token, provider_user_id, resource_owner_key) "
                    "VALUES (7, 'gmail', 'alice', 'provider-account', "
                    "'toby:slack:41:UALICE')"
                )
            )
            with pytest.raises(RuntimeError, match="actor-owned UserOAuth"):
                migration.downgrade()

        assert "resource_owner_key" in {
            column["name"] for column in inspect(connection).get_columns("user_oauth")
        }


def test_upgrade_rejects_dialect_without_partial_unique_indexes_before_inspection() -> (
    None
):
    migration = _migration_module()
    fake_op = SimpleNamespace(
        get_bind=lambda: SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
    )

    with patch.object(migration, "op", fake_op):
        with pytest.raises(RuntimeError, match="partial unique indexes"):
            migration.upgrade()


def test_upgrade_without_user_oauth_table_is_a_noop(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        assert "user_oauth" not in inspect(connection).get_table_names()


def test_sqlite_upgrade_rejects_owner_index_name_collision_before_table_rebuild(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-index-collision.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_old_table(connection)
        connection.execute(
            text(f"CREATE INDEX {ORDINARY_INDEX} ON user_oauth (user_id)")
        )

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="already exist"):
                migration.upgrade()

        constraints = {
            constraint["name"]
            for constraint in inspect(connection).get_unique_constraints("user_oauth")
        }
        assert OLD_CONSTRAINT in constraints
        assert "resource_owner_key" not in {
            column["name"] for column in inspect(connection).get_columns("user_oauth")
        }


@pytest.mark.parametrize("relation_type", ["TABLE", "VIEW"])
def test_sqlite_upgrade_rejects_relation_name_collision_before_rebuild(
    tmp_path,
    relation_type: str,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-relation-collision.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_old_table(connection)
        if relation_type == "TABLE":
            connection.execute(text(f"CREATE TABLE {ORDINARY_INDEX} (id INTEGER)"))
        else:
            connection.execute(text(f"CREATE VIEW {ORDINARY_INDEX} AS SELECT 1 AS id"))

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="already exist"):
                migration.upgrade()

        constraints = {
            constraint["name"]
            for constraint in inspect(connection).get_unique_constraints("user_oauth")
        }
        assert OLD_CONSTRAINT in constraints
        assert "resource_owner_key" not in {
            column["name"] for column in inspect(connection).get_columns("user_oauth")
        }


def test_sqlite_upgrade_rejects_cross_table_index_name_collision_before_rebuild(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-global-collision.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_old_table(connection)
        connection.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))
        connection.execute(text(f"CREATE INDEX {ACTOR_INDEX} ON unrelated (id)"))

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="already exist"):
                migration.upgrade()

        constraints = {
            constraint["name"]
            for constraint in inspect(connection).get_unique_constraints("user_oauth")
        }
        assert OLD_CONSTRAINT in constraints
        assert "resource_owner_key" not in {
            column["name"] for column in inspect(connection).get_columns("user_oauth")
        }


@pytest.mark.parametrize(
    ("columns", "constraints"),
    [
        ({OWNER_COLUMN}, {OLD_CONSTRAINT}),
        (set(), set()),
    ],
)
def test_upgrade_rejects_partially_owner_aware_schema(
    columns: set[str], constraints: set[str]
) -> None:
    migration = _migration_module()
    fake_op = SimpleNamespace(
        get_bind=lambda: SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    )

    with (
        patch.object(migration, "op", fake_op),
        patch.object(migration, "_sqlite_batch_temp_table_exists", return_value=False),
        patch.object(migration, "_table_exists", return_value=True),
        patch.object(migration, "_users_table_exists", return_value=True),
        patch.object(migration, "_column_names", return_value=columns),
        patch.object(migration, "_constraint_names", return_value=constraints),
        patch.object(migration, "_user_cascade_fk_is_current", return_value=True),
    ):
        with pytest.raises(RuntimeError, match="partially owner-aware"):
            migration.upgrade()


def test_existing_owner_aware_schema_requires_semantic_index_definitions(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-current-drift.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_users_table(connection)
        _operations(connection).create_table(
            "user_oauth",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("provider", sa.String(50), nullable=False),
            sa.Column("access_token", sa.String(), nullable=False),
            sa.Column("provider_user_id", sa.String(), nullable=True),
            sa.Column("resource_owner_key", sa.String(512), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        connection.execute(
            text(f"CREATE INDEX {ORDINARY_INDEX} ON user_oauth (user_id)")
        )
        connection.execute(
            text(
                f"CREATE UNIQUE INDEX {ACTOR_INDEX} ON user_oauth "
                "(user_id, resource_owner_key, provider, provider_user_id) "
                "WHERE resource_owner_key IS NOT NULL"
            )
        )

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="incorrect indexes"):
                migration.upgrade()


@pytest.mark.parametrize(
    ("owner_type", "nullable", "server_default"),
    [
        (sa.String(255), True, None),
        (sa.String(512), False, None),
        (sa.String(512), True, sa.text("'actor:unexpected'")),
    ],
)
def test_existing_owner_aware_schema_requires_owner_column_semantics(
    tmp_path,
    owner_type: sa.String,
    nullable: bool,
    server_default: object | None,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'oauth-owner-column-drift.db'}")
    migration = _migration_module()

    with engine.begin() as connection:
        _create_users_table(connection)
        _operations(connection).create_table(
            "user_oauth",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("provider", sa.String(50), nullable=False),
            sa.Column("access_token", sa.String(), nullable=False),
            sa.Column("provider_user_id", sa.String(), nullable=True),
            sa.Column(
                "resource_owner_key",
                owner_type,
                nullable=nullable,
                server_default=server_default,
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        connection.execute(
            text(
                f"CREATE UNIQUE INDEX {ORDINARY_INDEX} ON user_oauth "
                "(user_id, provider, provider_user_id) "
                "WHERE resource_owner_key IS NULL"
            )
        )
        connection.execute(
            text(
                f"CREATE UNIQUE INDEX {ACTOR_INDEX} ON user_oauth "
                "(user_id, resource_owner_key, provider, provider_user_id) "
                "WHERE resource_owner_key IS NOT NULL"
            )
        )

        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="incorrect owner column"):
                migration.upgrade()


def test_postgresql_upgrade_creates_indexes_before_old_constraint_drop() -> None:
    """Verify call order; PostgreSQL integration tests cover transactional DDL."""
    migration = _migration_module()
    events: list[str] = []
    fake_op = SimpleNamespace(
        get_bind=lambda: SimpleNamespace(dialect=SimpleNamespace(name="postgresql")),
        add_column=lambda *_args, **_kwargs: events.append("add-column"),
        create_foreign_key=lambda *_args, **_kwargs: events.append("create-user-fk"),
        drop_constraint=lambda *_args, **_kwargs: events.append("drop-constraint"),
    )

    with (
        patch.object(migration, "op", fake_op),
        patch.object(migration, "_table_exists", return_value=True),
        patch.object(migration, "_users_table_exists", return_value=True),
        patch.object(migration, "_column_names", return_value=set()),
        patch.object(migration, "_constraint_names", return_value={OLD_CONSTRAINT}),
        patch.object(migration, "_user_cascade_fk_is_current", return_value=False),
        patch.object(
            migration,
            "_create_owner_indexes",
            side_effect=lambda: events.append("create-indexes"),
        ),
    ):
        migration.upgrade()

    assert events == [
        "add-column",
        "create-user-fk",
        "create-indexes",
        "drop-constraint",
    ]


def test_create_owner_indexes_attempts_all_postgresql_indexes() -> None:
    """The database, not a mocked preflight, rejects relation-name collisions.

    The PostgreSQL collision-and-retry safety net is covered by
    ``test_postgresql_owner_index_collision_allows_retry_after_remediation``.
    """
    migration = _migration_module()
    created: list[tuple[str, tuple[str, ...], bool, str]] = []

    def capture_index(name, _table, columns, *, unique, postgresql_where, **_kwargs):
        created.append((name, tuple(columns), unique, str(postgresql_where).lower()))

    fake_op = SimpleNamespace(
        get_bind=lambda: SimpleNamespace(dialect=SimpleNamespace(name="postgresql")),
        create_index=capture_index,
    )

    with patch.object(migration, "op", fake_op):
        migration._create_owner_indexes()

    assert created == [
        (
            ORDINARY_INDEX,
            ("user_id", "provider", "provider_user_id"),
            True,
            "resource_owner_key is null",
        ),
        (
            ACTOR_INDEX,
            ("user_id", "resource_owner_key", "provider", "provider_user_id"),
            True,
            "resource_owner_key is not null",
        ),
    ]


def test_postgresql_index_creation_failure_keeps_old_constraint() -> None:
    migration = _migration_module()
    events: list[str] = []
    fake_op = SimpleNamespace(
        get_bind=lambda: SimpleNamespace(dialect=SimpleNamespace(name="postgresql")),
        add_column=lambda *_args, **_kwargs: events.append("add-column"),
        drop_constraint=lambda *_args, **_kwargs: events.append("drop-constraint"),
    )

    with (
        patch.object(migration, "op", fake_op),
        patch.object(migration, "_table_exists", return_value=True),
        patch.object(migration, "_users_table_exists", return_value=True),
        patch.object(migration, "_column_names", return_value=set()),
        patch.object(migration, "_constraint_names", return_value={OLD_CONSTRAINT}),
        patch.object(migration, "_user_cascade_fk_is_current", return_value=True),
        patch.object(
            migration,
            "_create_owner_indexes",
            side_effect=RuntimeError("index creation failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="index creation failed"):
            migration.upgrade()

    assert events == ["add-column"]


def test_downgrade_rejects_unsupported_dialect_before_inspection() -> None:
    migration = _migration_module()
    fake_op = SimpleNamespace(
        get_bind=lambda: SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
    )

    with patch.object(migration, "op", fake_op):
        with pytest.raises(RuntimeError, match="partial unique indexes"):
            migration.downgrade()
