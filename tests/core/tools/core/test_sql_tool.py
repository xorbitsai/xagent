"""
Tests for SQL Tool.
"""

import os
import sqlite3
from unittest.mock import MagicMock, Mock, patch

import pytest

from xagent.core.tools.core.sql_tool import (
    _get_connection_url,
    _row_to_dict,
    execute_sql_query,
    get_database_type,
)
from xagent.core.workspace import SPILL_DIR_NAME, TaskWorkspace


class TestGetConnectionUrl:
    """Test connection URL retrieval from environment variables."""

    def test_get_connection_url_success(self, monkeypatch):
        """Test successful URL retrieval."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///test.db")
        url = _get_connection_url("test")
        assert url.drivername == "sqlite"
        assert url.database == "test.db"

    def test_get_connection_url_case_insensitive(self, monkeypatch):
        """Test connection name is case-insensitive."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_ANALYTICS", "postgresql://localhost/db")
        url = _get_connection_url("analytics")
        assert url.drivername == "postgresql"

    def test_get_connection_url_not_found(self, monkeypatch):
        """Test error when connection not found."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///test.db")
        with pytest.raises(ValueError, match="Database connection 'missing' not found"):
            _get_connection_url("missing")

    def test_get_connection_url_no_databases(self, monkeypatch):
        """Test error when no databases configured."""
        # Clear any existing XAGENT_EXTERNAL_DB_* variables
        for key in list(os.environ.keys()):
            if key.startswith("XAGENT_EXTERNAL_DB_"):
                monkeypatch.delenv(key)
        with pytest.raises(ValueError, match="not found"):
            _get_connection_url("test")


class TestGetDatabaseType:
    """Test database type detection."""

    def test_get_database_type_sqlite(self, monkeypatch):
        """Test SQLite database type detection."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///test.db")
        db_type = get_database_type("test")
        assert db_type == "sqlite"

    def test_get_database_type_postgresql(self, monkeypatch):
        """Test PostgreSQL database type detection."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_ANALYTICS", "postgresql://localhost/db")
        db_type = get_database_type("analytics")
        assert db_type == "postgresql"

    def test_get_database_type_mysql(self, monkeypatch):
        """Test MySQL database type detection."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_PROD", "mysql+pymysql://localhost/prod")
        db_type = get_database_type("prod")
        assert db_type == "mysql"

    def test_get_database_type_not_found(self, monkeypatch):
        """Test error when connection not found."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///test.db")
        with pytest.raises(ValueError, match="not found"):
            get_database_type("missing")


class TestRowToDict:
    """Test SQLAlchemy Row to dict conversion."""

    def test_row_to_dict_basic(self):
        """Test basic row conversion."""
        mock_row = Mock()
        mock_row._mapping = {"id": 1, "name": "test"}
        result = _row_to_dict(mock_row)
        assert result == {"id": 1, "name": "test"}

    def test_row_to_dict_empty(self):
        """Test empty row conversion."""
        mock_row = Mock()
        mock_row._mapping = {}
        result = _row_to_dict(mock_row)
        assert result == {}


class TestExecuteSqlQuery:
    """Test SQL query execution."""

    def test_execute_sql_query_no_connection(self, monkeypatch):
        """Test error when connection not found."""
        # Clear any existing XAGENT_EXTERNAL_DB_* variables
        for key in list(os.environ.keys()):
            if key.startswith("XAGENT_EXTERNAL_DB_"):
                monkeypatch.delenv(key)
        # The function raises ValueError when connection not found
        with pytest.raises(ValueError, match="not found"):
            execute_sql_query("missing", "SELECT 1")

    @patch("xagent.core.tools.core.sql_tool.create_engine")
    def test_execute_sql_query_basic_select(self, mock_create_engine, monkeypatch):
        """Test basic SELECT query."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")

        # Mock the engine and connection
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        # Mock result
        mock_result = MagicMock()
        mock_result.returns_rows = True
        mock_row = Mock()
        mock_row._mapping = {"id": 1, "name": "test"}
        mock_result.all.return_value = [mock_row]
        mock_conn.execute.return_value = mock_result

        result = execute_sql_query("test", "SELECT * FROM users")
        assert result["success"] is True
        assert result["row_count"] == 1
        assert len(result["rows"]) == 1
        assert result["rows"][0] == {"id": 1, "name": "test"}

    @patch("xagent.core.tools.core.sql_tool.create_engine")
    def test_execute_sql_query_insert(self, mock_create_engine, monkeypatch):
        """Test INSERT query."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")

        # Mock the engine and connection
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        # Mock result for INSERT
        mock_result = MagicMock()
        mock_result.returns_rows = False
        mock_result.rowcount = 5
        mock_conn.execute.return_value = mock_result

        result = execute_sql_query("test", "INSERT INTO users VALUES (1, 'test')")
        assert result["success"] is True
        assert result["row_count"] == 5
        assert (
            result["message"]
            == "Query executed successfully on 'test', affected 5 row(s)"
        )

    @patch("xagent.core.tools.core.sql_tool.create_engine")
    def test_execute_sql_query_with_export_csv(
        self, mock_create_engine, monkeypatch, tmp_path
    ):
        """Test query with CSV export."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")

        # Mock the engine and connection
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        # Create mock Row objects for fetchmany
        mock_row1 = Mock()
        mock_row1._mapping = {"id": 1, "name": "test1"}
        mock_row2 = Mock()
        mock_row2._mapping = {"id": 2, "name": "test2"}

        # Mock result
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchmany.side_effect = [
            [mock_row1, mock_row2],
            [],  # End of results
        ]
        mock_conn.execute.return_value = mock_result

        def mock_create_record(self, file_id, file_path, db_session=None):
            return None

        monkeypatch.setattr(TaskWorkspace, "_create_file_record", mock_create_record)
        workspace = TaskWorkspace("test_sql_export", str(tmp_path))

        result = execute_sql_query(
            "test",
            "SELECT * FROM users",
            output_file="test.csv",
            workspace=workspace,
        )
        assert result["success"] is True
        assert result["row_count"] == 2
        assert "exported" in result["message"].lower()
        assert isinstance(result.get("file_id"), str)
        assert result["filename"] == "test.csv"
        assert result["mime_type"] == "text/csv"
        assert result["relative_path"] == "output/test.csv"
        assert result["file_ref"]["file_id"] == result["file_id"]
        assert (workspace.output_dir / "test.csv").exists()

    @pytest.mark.parametrize(
        "extension",
        [
            pytest.param(".csv", id="csv"),
            pytest.param(".jsonl", id="jsonlines"),
            pytest.param(".parquet", id="parquet"),
        ],
    )
    @patch("xagent.core.tools.core.sql_tool.create_engine")
    def test_execute_sql_query_refuses_to_export_into_the_engine_directory(
        self, mock_create_engine, monkeypatch, tmp_path, extension
    ):
        """output_file is model-facing; the engine-owned subtree is refused
        before anything is opened, on every export format, whether or not
        the directory exists yet, and the engine's own bytes are untouched."""
        if extension == ".parquet":
            pytest.importorskip("pyarrow")
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id"]
        mock_result.fetchmany.side_effect = [[], []]
        mock_conn.execute.return_value = mock_result
        workspace = TaskWorkspace("test_sql_export_refused", str(tmp_path))
        reserved = workspace.output_dir / SPILL_DIR_NAME
        engine_file = reserved / f"acme-stored-result{extension}"

        with pytest.raises(ValueError, match="engine-owned"):
            execute_sql_query(
                "test",
                "SELECT * FROM users",
                output_file=f"{SPILL_DIR_NAME}/acme-stored-result{extension}",
                workspace=workspace,
            )
        assert not reserved.exists()

        reserved.mkdir()
        engine_file.write_bytes(b"engine bytes")
        with pytest.raises(ValueError, match="engine-owned"):
            execute_sql_query(
                "test",
                "SELECT * FROM users",
                output_file=f"{SPILL_DIR_NAME}/acme-stored-result{extension}",
                workspace=workspace,
            )
        assert engine_file.read_bytes() == b"engine bytes"

    @pytest.mark.parametrize(
        "statement",
        [
            pytest.param("CREATE TABLE t (id INTEGER)", id="ddl"),
            pytest.param("INSERT INTO t0 VALUES (1)", id="dml"),
            pytest.param("SELECT 1 AS one", id="select"),
        ],
    )
    @pytest.mark.parametrize(
        "extension",
        [
            pytest.param(".csv", id="csv"),
            pytest.param(".jsonl", id="jsonlines"),
            pytest.param(".parquet", id="parquet"),
        ],
    )
    @patch("xagent.core.tools.core.sql_tool.create_engine")
    def test_a_refused_export_executes_nothing(
        self, mock_create_engine, monkeypatch, tmp_path, extension, statement
    ):
        """The export target is resolved before the statement runs, so a
        statement whose export is refused never reaches the database --
        for every export format and whether the statement reads or writes."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        workspace = TaskWorkspace("test_sql_refused_no_execute", str(tmp_path))

        with pytest.raises(ValueError, match="engine-owned"):
            execute_sql_query(
                "test",
                statement,
                output_file=f"{SPILL_DIR_NAME}/result{extension}",
                workspace=workspace,
            )
        mock_conn.execute.assert_not_called()

    @patch("xagent.core.tools.core.sql_tool.create_engine")
    def test_an_unsupported_export_format_is_reported_before_the_target_is_resolved(
        self, mock_create_engine, monkeypatch, tmp_path
    ):
        """The format check keeps its place ahead of the target resolution:
        an unsupported extension is reported as such even inside the engine
        directory, and nothing runs."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        workspace = TaskWorkspace("test_sql_unsupported_first", str(tmp_path))

        with pytest.raises(ValueError, match="Unsupported file format"):
            execute_sql_query(
                "test",
                "SELECT 1 AS one",
                output_file=f"{SPILL_DIR_NAME}/result.xlsx",
                workspace=workspace,
            )
        mock_conn.execute.assert_not_called()

    def test_a_refused_export_leaves_the_database_untouched(
        self, monkeypatch, tmp_path
    ):
        """On a real SQLite database a CREATE TABLE is committed the moment it
        runs, so it is the statement that shows whether anything ran: after a
        refused export the table does not exist."""
        db_path = tmp_path / "external.db"
        sqlite3.connect(db_path).close()
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", f"sqlite:///{db_path}")
        workspace = TaskWorkspace("test_sql_refused_untouched", str(tmp_path))

        with pytest.raises(ValueError, match="engine-owned"):
            execute_sql_query(
                "test",
                "CREATE TABLE created_by_a_refused_export (id INTEGER)",
                output_file=f"{SPILL_DIR_NAME}/result.csv",
                workspace=workspace,
            )

        with sqlite3.connect(db_path) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        assert tables == []
        assert not (workspace.output_dir / SPILL_DIR_NAME).exists()

    @patch("xagent.core.tools.core.sql_tool.create_engine")
    def test_execute_sql_query_export_parquet_no_pyarrow(
        self, mock_create_engine, monkeypatch
    ):
        """Test Parquet export fails without pyarrow."""
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")

        # Mock workspace
        mock_workspace = MagicMock()
        mock_workspace.resolve_path.return_value = "/tmp/test.parquet"

        # Mock pyarrow import to fail
        with patch.dict("sys.modules", {"pyarrow": None}):
            # The function should raise ImportError when trying to import pyarrow
            with pytest.raises(ImportError, match="pyarrow"):
                execute_sql_query(
                    "test",
                    "SELECT * FROM users",
                    output_file="test.parquet",
                    workspace=mock_workspace,
                )

    @patch("xagent.core.tools.core.sql_tool.create_engine")
    def test_execute_sql_query_with_empty_parquet_export(
        self, mock_create_engine, monkeypatch, tmp_path
    ):
        """Test empty Parquet export still writes a registered file."""
        pq = pytest.importorskip("pyarrow.parquet")
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchmany.return_value = []
        mock_conn.execute.return_value = mock_result

        def mock_create_record(self, file_id, file_path, db_session=None):
            return None

        monkeypatch.setattr(TaskWorkspace, "_create_file_record", mock_create_record)
        workspace = TaskWorkspace("test_sql_empty_parquet_export", str(tmp_path))

        result = execute_sql_query(
            "test",
            "SELECT * FROM users WHERE 1 = 0",
            output_file="empty.parquet",
            workspace=workspace,
        )

        output_file = workspace.output_dir / "empty.parquet"
        exported_table = pq.read_table(output_file)

        assert result["success"] is True
        assert result["row_count"] == 0
        assert isinstance(result.get("file_id"), str)
        assert result["filename"] == "empty.parquet"
        assert result["relative_path"] == "output/empty.parquet"
        assert result["file_ref"]["file_id"] == result["file_id"]
        assert output_file.exists()
        assert exported_table.num_rows == 0
        assert exported_table.column_names == ["id", "name"]
