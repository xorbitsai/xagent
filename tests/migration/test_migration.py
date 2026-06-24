from unittest.mock import MagicMock, Mock, patch

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from xagent.db import try_upgrade_db
from xagent.db.config import create_alembic_config


class TestTryUpgradeDb:
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

    @patch("xagent.db.migration.command.upgrade")
    @patch("xagent.db.migration.create_alembic_config")
    @patch("xagent.db.migration.get_alembic_revision")
    def test_successful_upgrade(
        self, mock_get_revision, mock_create_config, mock_upgrade
    ):
        engine = MagicMock()
        mock_get_revision.return_value = "abc123"
        mock_config = mock_create_config.return_value
        mock_config.attributes = {}

        # Mock connection context manager
        connection = Mock()
        engine.begin.return_value.__enter__.return_value = connection

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
        mock_get_revision.return_value = None
        mock_is_empty.return_value = True
        mock_config = mock_create_config.return_value
        mock_config.attributes = {}

        connection = Mock()
        engine.begin.return_value.__enter__.return_value = connection

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
        engine = Mock()
        mock_get_revision.return_value = None
        mock_is_empty.return_value = False  # Database has tables but no revision

        with pytest.raises(
            RuntimeError, match="Database exists without alembic revision"
        ):
            try_upgrade_db(engine)

    @patch("xagent.db.migration.command.upgrade")
    @patch("xagent.db.migration.create_alembic_config")
    @patch("xagent.db.migration.get_alembic_revision")
    def test_raises_error_on_upgrade_failure(
        self, mock_get_revision, mock_create_config, mock_upgrade
    ):
        engine = MagicMock()
        mock_get_revision.return_value = "abc123"
        mock_upgrade.side_effect = Exception("Upgrade failed")

        with pytest.raises(Exception, match="Upgrade failed"):
            try_upgrade_db(engine)

    @patch("xagent.db.migration.logger")
    @patch("xagent.db.migration.command.upgrade")
    @patch("xagent.db.migration.create_alembic_config")
    @patch("xagent.db.migration.get_alembic_revision")
    def test_logs_upgrade_process(
        self, mock_get_revision, mock_create_config, mock_upgrade, mock_logger
    ):
        engine = MagicMock()
        mock_get_revision.return_value = "abc123"
        mock_config = mock_create_config.return_value
        mock_config.attributes = {}

        try_upgrade_db(engine)

        mock_logger.info.assert_any_call("Starting database upgrade process")
        mock_logger.info.assert_any_call("Current version: abc123, upgrading to head")

    @patch("xagent.db.migration.logger")
    @patch("xagent.db.migration.get_alembic_revision")
    def test_logs_error_on_failure(self, mock_get_revision, mock_logger):
        engine = Mock()
        mock_get_revision.side_effect = RuntimeError("DB error")

        with pytest.raises(RuntimeError, match="DB error"):
            try_upgrade_db(engine)

        mock_logger.error.assert_called_once()
