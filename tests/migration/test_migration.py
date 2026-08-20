import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock, patch

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from xagent.db import try_upgrade_db
from xagent.db.config import create_alembic_config
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas


class TestTryUpgradeDb:
    def test_unsupported_dialect_fails_before_fresh_schema_or_migration(self):
        engine = MagicMock()
        engine.dialect.name = "mysql"

        with pytest.raises(RuntimeError, match="partial unique indexes"):
            try_upgrade_db(engine)

        engine.connect.assert_not_called()

    def test_non_string_dialect_name_does_not_bypass_startup_invariant(self):
        engine = MagicMock()
        engine.dialect.name = None

        with pytest.raises(RuntimeError, match="partial unique indexes"):
            try_upgrade_db(engine)

        engine.connect.assert_not_called()

    def test_postgresql_serializes_startup_migration_before_reading_revision(
        self,
        monkeypatch,
    ):
        events: list[str] = []
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        connection = MagicMock()
        connection.in_transaction.return_value = False
        engine.connect.return_value.__enter__.return_value = connection
        connection.execute.side_effect = lambda statement, *_args, **_kwargs: (
            events.append(
                "lock"
                if "pg_advisory_lock" in str(statement)
                and "unlock" not in str(statement)
                else "unlock"
            )
        )

        def get_revision(connectable):
            assert connectable is connection
            events.append("revision")
            return "abc123"

        def upgrade(config, revision):
            events.append("upgrade")
            assert revision == "head"
            assert config.attributes["connection"] is connection

        monkeypatch.setattr(
            "xagent.db.migration.get_alembic_revision",
            get_revision,
        )
        monkeypatch.setattr(
            "xagent.db.migration._check_revision_is_known",
            lambda _config, _engine, _version: None,
        )
        monkeypatch.setattr("xagent.db.migration.command.upgrade", upgrade)

        try_upgrade_db(engine)

        assert events == ["lock", "revision", "upgrade", "unlock"]
        assert connection.commit.call_count == 2

    def test_database_startup_lock_covers_fresh_schema_creation_and_seed(
        self,
        monkeypatch,
    ) -> None:
        from xagent.web.models.database import Base, _initialize_database_schema

        engine = MagicMock()
        gate = threading.Barrier(2)
        startup_mutex = threading.Lock()
        state_lock = threading.Lock()
        database_state = {"empty": True}
        active_initializers = 0
        max_active_initializers = 0
        seed_calls = 0
        empty_observations: list[bool] = []

        @contextmanager
        def startup_lock(_engine):
            nonlocal active_initializers, max_active_initializers
            with startup_mutex:
                connection = MagicMock()
                with state_lock:
                    active_initializers += 1
                    max_active_initializers = max(
                        max_active_initializers,
                        active_initializers,
                    )
                try:
                    yield connection
                finally:
                    with state_lock:
                        active_initializers -= 1

        def is_empty(_bind):
            with state_lock:
                observed = bool(database_state["empty"])
                empty_observations.append(observed)
                return observed

        def create_all(*, bind):
            assert bind is not engine
            with state_lock:
                database_state["empty"] = False

        def seed(_connection):
            nonlocal seed_calls
            with state_lock:
                seed_calls += 1

        monkeypatch.setattr(
            "xagent.db.migration.database_startup_lock",
            startup_lock,
        )
        monkeypatch.setattr("xagent.db.migration.is_database_empty", is_empty)
        monkeypatch.setattr("xagent.db.migration.try_upgrade_db", Mock())
        monkeypatch.setattr(Base.metadata, "create_all", create_all)
        monkeypatch.setattr(
            "xagent.web.builtin_mcp_registry.seed_builtin_oauth_and_public_mcp_apps",
            seed,
        )
        monkeypatch.setattr(
            "xagent.web.builtin_mcp_registry.validate_builtin_public_mcp_apps",
            lambda _connection: [],
        )

        def initialize() -> None:
            gate.wait()
            _initialize_database_schema(engine)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(initialize) for _ in range(2)]
            for future in futures:
                future.result(timeout=2)

        assert max_active_initializers == 1
        assert empty_observations == [True, False]
        assert seed_calls == 1

    def test_postgresql_releases_startup_migration_lock_after_failure(
        self,
        monkeypatch,
    ):
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        connection = MagicMock()
        connection.in_transaction.return_value = True
        engine.connect.return_value.__enter__.return_value = connection

        monkeypatch.setattr(
            "xagent.db.migration.get_alembic_revision",
            lambda _engine: "abc123",
        )
        monkeypatch.setattr(
            "xagent.db.migration._check_revision_is_known",
            lambda _config, _engine, _version: None,
        )
        monkeypatch.setattr(
            "xagent.db.migration.command.upgrade",
            Mock(side_effect=RuntimeError("upgrade failed")),
        )

        with pytest.raises(RuntimeError, match="upgrade failed"):
            try_upgrade_db(engine)

        assert (
            connection.execute.call_args_list[0]
            .args[0]
            .text.startswith("SELECT pg_advisory_lock")
        )
        assert (
            connection.execute.call_args_list[-1]
            .args[0]
            .text.startswith("SELECT pg_advisory_unlock")
        )
        assert connection.rollback.call_count == 1
        assert connection.commit.call_count == 2

    def test_sqlite_current_head_preserves_preexisting_fk_violations(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path / 'legacy-orphan.db'}")
        apply_sqlite_concurrency_pragmas(engine)
        script = ScriptDirectory.from_config(create_alembic_config(engine))

        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 0
                conn.exec_driver_sql(
                    "CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)"
                )
                conn.exec_driver_sql(
                    "INSERT INTO alembic_version (version_num) VALUES (?)",
                    (script.get_current_head(),),
                )
                conn.exec_driver_sql("CREATE TABLE agents (id INTEGER PRIMARY KEY)")
                conn.exec_driver_sql(
                    "CREATE TABLE workforces ("
                    "id INTEGER PRIMARY KEY, "
                    "manager_agent_id INTEGER NOT NULL, "
                    "FOREIGN KEY(manager_agent_id) REFERENCES agents(id) "
                    "ON DELETE RESTRICT)"
                )
                conn.exec_driver_sql(
                    "INSERT INTO workforces (id, manager_agent_id) VALUES (1, 99)"
                )
                conn.commit()
                conn.exec_driver_sql("PRAGMA foreign_keys=ON")
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
                conn.rollback()

            try_upgrade_db(engine)

            with engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
                assert (
                    conn.exec_driver_sql("SELECT COUNT(*) FROM workforces").scalar_one()
                    == 1
                )
                assert conn.exec_driver_sql("PRAGMA foreign_key_check").all() == [
                    ("workforces", 1, "agents", 0)
                ]
        finally:
            engine.dispose()

    def test_owner_table_rebuild_preserves_inbound_gmail_row_via_try_upgrade_db(
        self, tmp_path
    ):
        engine = create_engine(f"sqlite:///{tmp_path / 'owner-inbound.db'}")
        apply_sqlite_concurrency_pragmas(engine)
        config = create_alembic_config(engine)
        parent = "20260818_seed_jira_mcp_app"

        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE TABLE users (id INTEGER PRIMARY KEY, "
                        "username VARCHAR(50) NOT NULL, password_hash VARCHAR(255) "
                        "NOT NULL, is_admin BOOLEAN NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE user_oauth (id INTEGER PRIMARY KEY, "
                        "user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, "
                        "provider VARCHAR(50) NOT NULL, access_token VARCHAR NOT NULL, "
                        "provider_user_id VARCHAR, CONSTRAINT uq_user_provider_account "
                        "UNIQUE (user_id, provider, provider_user_id))"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE gmail_watch_states (id INTEGER PRIMARY KEY, "
                        "user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, "
                        "oauth_account_id INTEGER NOT NULL UNIQUE REFERENCES "
                        "user_oauth(id) ON DELETE CASCADE, email VARCHAR(255) NOT NULL, "
                        "history_id VARCHAR(255) NOT NULL, topic_name VARCHAR(512) NOT NULL)"
                    )
                )
            command.stamp(config, parent)
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO users (id, username, password_hash, is_admin) "
                        "VALUES (7, 'owner', 'hash', 0)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO user_oauth "
                        "(id, user_id, provider, access_token, provider_user_id) "
                        "VALUES (9, 7, 'gmail', 'token', 'provider-account')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO gmail_watch_states "
                        "(id, user_id, oauth_account_id, email, history_id, topic_name) "
                        "VALUES (11, 7, 9, 'owner@example.com', 'history', 'topic')"
                    )
                )

            try_upgrade_db(engine)

            with engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
                assert conn.execute(
                    text("SELECT id, oauth_account_id FROM gmail_watch_states")
                ).all() == [(11, 9)]
                assert conn.exec_driver_sql("PRAGMA foreign_key_check").all() == []
                assert "resource_owner_key" in {
                    column["name"] for column in inspect(conn).get_columns("user_oauth")
                }
        finally:
            engine.dispose()

    @pytest.mark.parametrize("with_existing_violation", [False, True])
    def test_sqlite_upgrade_rejects_new_fk_violations(
        self,
        tmp_path,
        monkeypatch,
        with_existing_violation,
    ):
        engine = create_engine(f"sqlite:///{tmp_path / 'new-orphan.db'}")
        apply_sqlite_concurrency_pragmas(engine)

        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 0
                conn.exec_driver_sql("CREATE TABLE parents (id INTEGER PRIMARY KEY)")
                conn.exec_driver_sql(
                    "CREATE TABLE children ("
                    "key TEXT PRIMARY KEY, "
                    "parent_id INTEGER NOT NULL, "
                    "FOREIGN KEY(parent_id) REFERENCES parents(id)) "
                    "WITHOUT ROWID"
                )
                if with_existing_violation:
                    conn.exec_driver_sql(
                        "INSERT INTO children (key, parent_id) VALUES ('legacy', 99)"
                    )
                conn.commit()
                conn.exec_driver_sql("PRAGMA foreign_keys=ON")
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
                conn.rollback()

            def introduce_new_violation(config, revision):
                assert revision == "head"
                connection = config.attributes["connection"]
                assert (
                    connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 0
                )
                connection.exec_driver_sql(
                    "INSERT INTO children (key, parent_id) VALUES ('new', 99)"
                )

            monkeypatch.setattr(
                "xagent.db.migration.get_alembic_revision", lambda _engine: "abc123"
            )
            monkeypatch.setattr(
                "xagent.db.migration._check_revision_is_known",
                lambda _config, _engine, _version: None,
            )
            monkeypatch.setattr(
                "xagent.db.migration.command.upgrade", introduce_new_violation
            )

            with pytest.raises(
                RuntimeError,
                match="SQLite migration produced new foreign-key violations",
            ):
                try_upgrade_db(engine)

            with engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
                assert conn.exec_driver_sql(
                    "SELECT key FROM children ORDER BY key"
                ).scalars().all() == (["legacy"] if with_existing_violation else [])
        finally:
            engine.dispose()

    def test_sqlite_upgrade_distinguishes_without_rowid_violation_rows(
        self,
        tmp_path,
        monkeypatch,
    ):
        engine = create_engine(f"sqlite:///{tmp_path / 'without-rowid-swap.db'}")
        apply_sqlite_concurrency_pragmas(engine)

        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
                conn.exec_driver_sql("CREATE TABLE parents (id INTEGER PRIMARY KEY)")
                conn.exec_driver_sql(
                    "CREATE TABLE children ("
                    "key TEXT PRIMARY KEY, "
                    "parent_id INTEGER NOT NULL, "
                    "FOREIGN KEY(parent_id) REFERENCES parents(id)) "
                    "WITHOUT ROWID"
                )
                conn.exec_driver_sql("INSERT INTO parents (id) VALUES (2)")
                conn.exec_driver_sql(
                    "INSERT INTO children (key, parent_id) VALUES "
                    "('legacy', 99), ('was-valid', 2)"
                )
                conn.commit()
                conn.exec_driver_sql("PRAGMA foreign_keys=ON")
                conn.rollback()

            def swap_violation_row(config, revision):
                assert revision == "head"
                connection = config.attributes["connection"]
                connection.exec_driver_sql("INSERT INTO parents (id) VALUES (99)")
                connection.exec_driver_sql("DELETE FROM parents WHERE id = 2")

            monkeypatch.setattr(
                "xagent.db.migration.get_alembic_revision", lambda _engine: "abc123"
            )
            monkeypatch.setattr(
                "xagent.db.migration._check_revision_is_known",
                lambda _config, _engine, _version: None,
            )
            monkeypatch.setattr(
                "xagent.db.migration.command.upgrade", swap_violation_row
            )

            with pytest.raises(
                RuntimeError,
                match="SQLite migration produced new foreign-key violations",
            ):
                try_upgrade_db(engine)

            with engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
                assert conn.exec_driver_sql(
                    "SELECT id FROM parents ORDER BY id"
                ).scalars().all() == [2]
                assert conn.exec_driver_sql(
                    "SELECT key, parent_id FROM children ORDER BY key"
                ).all() == [("legacy", 99), ("was-valid", 2)]
        finally:
            engine.dispose()

    def test_sqlite_upgrade_distinguishes_fks_to_the_same_parent(
        self,
        tmp_path,
        monkeypatch,
    ):
        engine = create_engine(f"sqlite:///{tmp_path / 'changed-orphan.db'}")
        apply_sqlite_concurrency_pragmas(engine)

        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
                conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
                conn.exec_driver_sql(
                    "CREATE TABLE child ("
                    "id INTEGER PRIMARY KEY, "
                    "created_by INTEGER NOT NULL, "
                    "updated_by INTEGER NOT NULL, "
                    "FOREIGN KEY(created_by) REFERENCES users(id), "
                    "FOREIGN KEY(updated_by) REFERENCES users(id))"
                )
                conn.exec_driver_sql("INSERT INTO users (id) VALUES (2)")
                conn.exec_driver_sql(
                    "INSERT INTO child (id, created_by, updated_by) VALUES (1, 1, 2)"
                )
                conn.commit()
                conn.exec_driver_sql("PRAGMA foreign_keys=ON")
                conn.rollback()

            def move_violation_to_other_fk(config, revision):
                assert revision == "head"
                connection = config.attributes["connection"]
                connection.exec_driver_sql("INSERT INTO users (id) VALUES (1)")
                connection.exec_driver_sql("DELETE FROM users WHERE id = 2")

            monkeypatch.setattr(
                "xagent.db.migration.get_alembic_revision", lambda _engine: "abc123"
            )
            monkeypatch.setattr(
                "xagent.db.migration._check_revision_is_known",
                lambda _config, _engine, _version: None,
            )
            monkeypatch.setattr(
                "xagent.db.migration.command.upgrade", move_violation_to_other_fk
            )

            with pytest.raises(
                RuntimeError,
                match="SQLite migration produced new foreign-key violations",
            ):
                try_upgrade_db(engine)

            with engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
                assert conn.exec_driver_sql(
                    "SELECT id FROM users ORDER BY id"
                ).scalars().all() == [2]
                assert conn.exec_driver_sql("PRAGMA foreign_key_check").all() == [
                    ("child", 1, "users", 1)
                ]
        finally:
            engine.dispose()

    def test_sqlite_upgrade_preserves_inbound_fk_rows_when_enforcement_is_enabled(
        self, tmp_path
    ):
        engine = create_engine(f"sqlite:///{tmp_path / 'workforce-upgrade.db'}")
        apply_sqlite_concurrency_pragmas(engine)

        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE TABLE alembic_version "
                        "(version_num VARCHAR(255) NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO alembic_version (version_num) "
                        "VALUES ('20260715_add_public_mcp_app_audits')"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE workforces ("
                        "id INTEGER PRIMARY KEY, "
                        "name VARCHAR(200) NOT NULL, "
                        "manager_instructions TEXT)"
                    )
                )
                for table_name in (
                    "workforce_agents",
                    "workforce_runs",
                    "workforce_builder_messages",
                ):
                    conn.execute(
                        text(
                            f"CREATE TABLE {table_name} ("
                            "id INTEGER PRIMARY KEY, "
                            "workforce_id INTEGER NOT NULL, "
                            "FOREIGN KEY(workforce_id) REFERENCES workforces(id) "
                            "ON DELETE CASCADE)"
                        )
                    )
                conn.execute(
                    text(
                        "INSERT INTO workforces "
                        "(id, name, manager_instructions) "
                        "VALUES (1, 'Existing Workforce', 'legacy instructions')"
                    )
                )
                for table_name in (
                    "workforce_agents",
                    "workforce_runs",
                    "workforce_builder_messages",
                ):
                    conn.execute(
                        text(
                            f"INSERT INTO {table_name} (id, workforce_id) VALUES (1, 1)"
                        )
                    )

            try_upgrade_db(engine)

            columns = {
                column["name"] for column in inspect(engine).get_columns("workforces")
            }
            assert "manager_instructions" not in columns
            with engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
                assert conn.exec_driver_sql("PRAGMA foreign_key_check").all() == []
                assert (
                    conn.execute(text("SELECT COUNT(*) FROM workforces")).scalar() == 1
                )
                for table_name in (
                    "workforce_agents",
                    "workforce_runs",
                    "workforce_builder_messages",
                ):
                    assert (
                        conn.execute(
                            text(f"SELECT COUNT(*) FROM {table_name}")
                        ).scalar()
                        == 1
                    )
        finally:
            engine.dispose()

    def test_sqlite_upgrade_restores_fk_enforcement_after_failure(
        self, tmp_path, monkeypatch
    ):
        engine = create_engine(f"sqlite:///{tmp_path / 'failed-upgrade.db'}")
        apply_sqlite_concurrency_pragmas(engine)

        def fail_upgrade(config, revision):
            assert revision == "head"
            connection = config.attributes["connection"]
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 0
            raise RuntimeError("Upgrade failed")

        monkeypatch.setattr(
            "xagent.db.migration.get_alembic_revision", lambda _engine: "abc123"
        )
        monkeypatch.setattr(
            "xagent.db.migration._check_revision_is_known",
            lambda _config, _engine, _version: None,
        )
        monkeypatch.setattr("xagent.db.migration.command.upgrade", fail_upgrade)

        try:
            with pytest.raises(RuntimeError, match="Upgrade failed"):
                try_upgrade_db(engine)

            with engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        finally:
            engine.dispose()

    def test_stamps_new_database_with_persistent_wide_version_table(self):
        engine = create_engine("sqlite:///:memory:")

        try_upgrade_db(engine)

        columns = inspect(engine).get_columns("alembic_version")
        version_num = next(
            column for column in columns if column["name"] == "version_num"
        )
        assert version_num["type"].length == 255

        with engine.begin() as conn:
            version = conn.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar()

        script = ScriptDirectory.from_config(create_alembic_config(engine))
        assert version == script.get_current_head()

    def test_upgrade_backfills_legacy_sdk_tasks_as_hidden(self):
        engine = create_engine("sqlite:///:memory:")
        cfg = create_alembic_config(engine)

        with engine.begin() as conn:
            conn.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
            )
            conn.execute(
                text(
                    "INSERT INTO alembic_version (version_num) "
                    "VALUES ('20260616_add_agent_triggers')"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE tasks ("
                    "id INTEGER PRIMARY KEY, "
                    "source VARCHAR(20), "
                    "is_visible BOOLEAN NOT NULL)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO tasks (id, source, is_visible) VALUES "
                    "(1, 'sdk', 1), "
                    "(2, 'internal', 1), "
                    "(3, 'sdk', 0)"
                )
            )

            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")

            rows = conn.execute(
                text("SELECT id, is_visible FROM tasks ORDER BY id")
            ).all()

        assert rows == [(1, 0), (2, 1), (3, 0)]

    def test_upgrade_backfills_external_conversation_sources_conservatively(self):
        engine = create_engine("sqlite:///:memory:")
        cfg = create_alembic_config(engine)

        with engine.begin() as conn:
            conn.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
            )
            conn.execute(
                text(
                    "INSERT INTO alembic_version (version_num) "
                    "VALUES ('20260624_add_mcp_concurrency_config')"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE tasks ("
                    "id INTEGER PRIMARY KEY, "
                    "source VARCHAR(20), "
                    "is_visible BOOLEAN NOT NULL, "
                    "channel_name VARCHAR(100), "
                    "agent_config JSON)"
                )
            )
            conn.exec_driver_sql(
                "INSERT INTO tasks "
                "(id, source, is_visible, channel_name, agent_config) VALUES "
                "(1, 'sdk', 1, NULL, NULL), "
                "(2, 'internal', 1, 'Web Widget', '{\"guest_id\":\"g1\"}'), "
                "(3, 'internal', 1, 'Shared Agent', "
                '\'{"auth_mode":"share","share_agent_id":7}\'), '
                "(4, 'trigger', 0, NULL, '{\"trigger_type\":\"webhook\"}'), "
                "(5, 'internal', 1, 'Desktop', NULL), "
                "(6, 'widget', 0, 'Web Widget', '{\"guest_id\":\"g2\"}')"
            )

            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")

            first_rows = conn.execute(
                text("SELECT id, source, is_visible FROM tasks ORDER BY id")
            ).all()
            command.downgrade(cfg, "20260624_add_mcp_concurrency_config")
            command.upgrade(cfg, "head")
            rows = conn.execute(
                text("SELECT id, source, is_visible FROM tasks ORDER BY id")
            ).all()

        expected_rows = [
            (1, "sdk", 0),
            (2, "widget", 0),
            (3, "shared_link", 0),
            (4, "trigger", 0),
            (5, "internal", 1),
            (6, "widget", 0),
        ]
        assert first_rows == expected_rows
        assert rows == expected_rows

    def test_upgrades_through_check_from_known_older_revision(self):
        engine = create_engine("sqlite:///:memory:")

        with engine.begin() as conn:
            conn.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
            )
            conn.execute(
                text(
                    "INSERT INTO alembic_version (version_num) "
                    "VALUES ('20260616_add_agent_triggers')"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE tasks ("
                    "id INTEGER PRIMARY KEY, "
                    "source VARCHAR(20), "
                    "is_visible BOOLEAN NOT NULL)"
                )
            )

        try_upgrade_db(engine)

        with engine.begin() as conn:
            version = conn.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar()
        script = ScriptDirectory.from_config(create_alembic_config(engine))
        assert version == script.get_current_head()

    def test_treats_empty_string_revision_as_unversioned(self):
        engine = create_engine("sqlite:///:memory:")

        with engine.begin() as conn:
            conn.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
            )
            conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('')"))

        with pytest.raises(
            RuntimeError, match="Database exists without alembic revision"
        ):
            try_upgrade_db(engine)

    def test_raises_friendly_error_when_db_revision_is_unknown(self):
        engine = create_engine("sqlite:///:memory:")

        with engine.begin() as conn:
            conn.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
            )
            conn.execute(
                text(
                    "INSERT INTO alembic_version (version_num) "
                    "VALUES ('29991231_revision_from_the_future')"
                )
            )

        with pytest.raises(RuntimeError, match="newer version of xagent"):
            try_upgrade_db(engine)

    @patch("xagent.db.migration._check_revision_is_known")
    @patch("xagent.db.migration.command.upgrade")
    @patch("xagent.db.migration.create_alembic_config")
    @patch("xagent.db.migration.get_alembic_revision")
    def test_successful_upgrade(
        self, mock_get_revision, mock_create_config, mock_upgrade, _mock_check
    ):
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        mock_get_revision.return_value = "abc123"
        mock_config = mock_create_config.return_value
        mock_config.attributes = {}

        # PostgreSQL startup serialization and Alembic share this connection.
        connection = Mock()
        engine.connect.return_value.__enter__.return_value = connection

        try_upgrade_db(engine)

        mock_create_config.assert_called_once_with(engine)
        mock_upgrade.assert_called_once_with(mock_config, "head")
        assert mock_config.attributes["connection"] == connection

    @patch("xagent.db.migration.is_database_empty")
    @patch("xagent.db.migration.command.stamp")
    @patch("xagent.db.migration.create_alembic_config")
    @patch("xagent.db.migration.get_alembic_revision")
    def test_stamps_when_new_database(
        self, mock_get_revision, mock_create_config, mock_stamp, mock_is_empty
    ):
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        mock_get_revision.return_value = None
        mock_is_empty.return_value = True
        mock_config = mock_create_config.return_value
        mock_config.attributes = {}

        connection = Mock()
        engine.connect.return_value.__enter__.return_value = connection

        try_upgrade_db(engine)

        mock_create_config.assert_called_once_with(engine)
        mock_stamp.assert_called_once_with(mock_config, "head")
        assert mock_config.attributes["connection"] == connection

    @patch("xagent.db.migration.is_database_empty")
    @patch("xagent.db.migration.create_alembic_config")
    @patch("xagent.db.migration.get_alembic_revision")
    def test_raises_when_existing_database_unversioned(
        self, mock_get_revision, mock_create_config, mock_is_empty
    ):
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        mock_get_revision.return_value = None
        mock_is_empty.return_value = False  # Database has tables but no revision

        with pytest.raises(
            RuntimeError, match="Database exists without alembic revision"
        ):
            try_upgrade_db(engine)

    @patch("xagent.db.migration._check_revision_is_known")
    @patch("xagent.db.migration.command.upgrade")
    @patch("xagent.db.migration.create_alembic_config")
    @patch("xagent.db.migration.get_alembic_revision")
    def test_raises_error_on_upgrade_failure(
        self, mock_get_revision, mock_create_config, mock_upgrade, _mock_check
    ):
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        mock_get_revision.return_value = "abc123"
        mock_upgrade.side_effect = Exception("Upgrade failed")

        with pytest.raises(Exception, match="Upgrade failed"):
            try_upgrade_db(engine)

    @patch("xagent.db.migration._check_revision_is_known")
    @patch("xagent.db.migration.logger")
    @patch("xagent.db.migration.command.upgrade")
    @patch("xagent.db.migration.create_alembic_config")
    @patch("xagent.db.migration.get_alembic_revision")
    def test_logs_upgrade_process(
        self,
        mock_get_revision,
        mock_create_config,
        mock_upgrade,
        mock_logger,
        _mock_check,
    ):
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        mock_get_revision.return_value = "abc123"
        mock_config = mock_create_config.return_value
        mock_config.attributes = {}

        try_upgrade_db(engine)

        mock_logger.info.assert_any_call("Starting database upgrade process")
        mock_logger.info.assert_any_call(
            "Current version: %s, upgrading to head",
            "abc123",
        )

    @patch("xagent.db.migration.logger")
    @patch("xagent.db.migration.get_alembic_revision")
    def test_logs_error_on_failure(self, mock_get_revision, mock_logger):
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        mock_get_revision.side_effect = RuntimeError("DB error")

        with pytest.raises(RuntimeError, match="DB error"):
            try_upgrade_db(engine)

        mock_logger.error.assert_called_once()
