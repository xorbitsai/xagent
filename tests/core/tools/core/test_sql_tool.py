"""
Tests for SQL Tool.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from xagent.core.tools.core.sql_tool import (
    _get_connection_url,
    _row_to_dict,
    execute_sql_query,
    get_database_type,
)
from xagent.core.workspace import TaskWorkspace


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
