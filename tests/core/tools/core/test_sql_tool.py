"""
Tests for SQL Tool.
"""

import os
import sqlite3
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy.exc import DatabaseError

from xagent.core.tools.core.sql_tool import (
    _can_stream_query,
    _execute_sql_query_local,
    _get_connection_url,
    _row_to_dict,
    execute_sql_query,
    get_database_type,
    is_platform_database_url,
    validate_external_database_url,
)
from xagent.core.workspace import SPILL_DIR_NAME, TaskWorkspace
from xagent.web.services.tool_credentials import set_sql_connection


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "select 1;",
        "WITH q AS (SELECT 1) SELECT * FROM q",
        "VALUES (1)",
    ],
)
def test_read_queries_remain_eligible_for_server_side_cursors(query):
    assert _can_stream_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO metrics VALUES (42)",
        "UPDATE metrics SET value=1",
        "DELETE FROM metrics",
        "CREATE TABLE metrics (value int)",
        "SELECT 42 INTO metrics",
        "SELECT 1; INSERT INTO metrics VALUES (42)",
        "WITH q AS (DELETE FROM metrics RETURNING *) SELECT * FROM q",
        "WITH q AS (SELECT 1) INSERT INTO metrics SELECT * FROM q RETURNING *",
        "/* user statement */ UPDATE metrics SET value=1",
    ],
)
def test_other_queries_use_normal_cursors_without_rejection_or_retry(query):
    assert not _can_stream_query(query)


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

    def test_rejects_platform_database_with_equivalent_sqlite_path(
        self, monkeypatch, tmp_path
    ):
        platform_db = tmp_path / "platform.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{platform_db}")
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", f"sqlite:///{platform_db}")
        with pytest.raises(ValueError, match="platform database"):
            _get_connection_url("test")

    def test_rejects_platform_database_with_driver_and_credential_variants(
        self, monkeypatch
    ):
        monkeypatch.setenv("DATABASE_URL", "postgresql://platform@DB.Example/app")
        monkeypatch.setenv(
            "XAGENT_EXTERNAL_DB_TEST",
            "postgresql+psycopg://other:secret@db.example:5432/app",
        )
        with pytest.raises(ValueError, match="platform database"):
            _get_connection_url("test")

    def test_allows_distinct_platform_database(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://platform@db.example/app")
        monkeypatch.setenv(
            "XAGENT_EXTERNAL_DB_TEST", "postgresql://other@other.example/other"
        )
        assert _get_connection_url("test").database == "other"

    def test_rejects_other_database_on_platform_server(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://platform@db.example/app")
        monkeypatch.setenv(
            "XAGENT_EXTERNAL_DB_TEST", "postgresql://other@db.example/other"
        )
        with pytest.raises(ValueError, match="platform database"):
            _get_connection_url("test")

    def test_default_platform_sqlite_database_is_compared(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(
            "xagent.config.get_default_sqlite_db_path", lambda: "/tmp/xagent.db"
        )
        assert is_platform_database_url("sqlite:////tmp/xagent.db")

    @pytest.mark.parametrize(
        "candidate_template",
        [
            "sqlite:///file:{path}?uri=on",
            "sqlite:///file:{path}?uri=t",
            "sqlite:///file:{path}?uri=y",
            "sqlite:///file:{encoded_path}?uri=true",
            "sqlite:///file://localhost{path}?uri=yes",
            "sqlite:///file:{path}?uri=false&uri=true",
            "sqlite:///file:{path}?uri=0&uri=0",
        ],
    )
    def test_rejects_platform_sqlite_file_uri(
        self, monkeypatch, tmp_path, candidate_template
    ):
        platform_db = tmp_path / "platform.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{platform_db}")
        encoded_path = str(platform_db).replace("/", "%2F")
        monkeypatch.setenv(
            "XAGENT_EXTERNAL_DB_TEST",
            candidate_template.format(path=platform_db, encoded_path=encoded_path),
        )
        with pytest.raises(ValueError, match="SQLite URI mode"):
            _get_connection_url("test")

    @pytest.mark.parametrize(
        ("platform", "candidate"),
        [
            (
                "postgresql://platform@postgres/xagent",
                "postgresql://other@public.example/xagent?host=postgres",
            ),
            (
                "postgresql://platform@127.0.0.1/xagent",
                "postgresql://other@public.example/xagent?hostaddr=127.0.0.1",
            ),
            (
                "postgresql://platform@public.example:5433/xagent",
                "postgresql://other@public.example/xagent?port=5433",
            ),
            (
                "postgresql://platform@public.example/xagent",
                "postgresql://other@public.example/other?dbname=xagent",
            ),
            (
                "mysql://platform@public.example/xagent",
                "mysql+pymysql://other@public.example/other?database=xagent",
            ),
            (
                "postgresql://platform@/xagent?host=/var/run/postgresql",
                "postgresql://other@public.example/xagent?host=/var/run/postgresql",
            ),
        ],
    )
    def test_rejects_effective_network_target_overrides(
        self, monkeypatch, platform, candidate
    ):
        monkeypatch.setenv("DATABASE_URL", platform)
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", candidate)
        with pytest.raises(ValueError, match="platform database"):
            _get_connection_url("test")

    def test_allows_legitimate_external_unix_socket(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://platform@postgres/xagent")
        candidate = "postgresql://other@/analytics?host=/cloudsql/customer-project"
        assert validate_external_database_url(candidate).database == "analytics"

    @pytest.mark.parametrize(
        ("candidate", "key"),
        [
            ("postgresql://other@public/other?service=platform", "service"),
            ("mssql+pyodbc://other@public/other?odbc_connect=opaque", "odbc_connect"),
            (
                "mysql+pymysql://other@public/other?read_default_file=/tmp/db.cnf",
                "read_default_file",
            ),
        ],
    )
    def test_rejects_opaque_driver_target_configuration(
        self, monkeypatch, candidate, key
    ):
        monkeypatch.setenv("DATABASE_URL", "postgresql://platform@postgres/xagent")
        with pytest.raises(ValueError, match=key):
            validate_external_database_url(candidate)

    def test_rejects_multi_host_failover_target(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://platform@postgres/xagent")
        with pytest.raises(ValueError, match="multiple 'host' targets"):
            validate_external_database_url(
                "postgresql://other@public/xagent?host=postgres,public"
            )

    @pytest.mark.parametrize(
        "candidate",
        [
            "postgresql://other@postgres,public/xagent",
            "postgresql://xagent:xagent_password@postgres:5432",
        ],
    )
    def test_rejects_driver_effective_postgres_target_spellings(
        self, monkeypatch, candidate
    ):
        monkeypatch.setenv("DATABASE_URL", "postgresql://platform@postgres:5432/xagent")
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", candidate)
        with pytest.raises(ValueError, match="platform database"):
            _get_connection_url("test")

    def test_mariadb_and_mysql_are_same_backend(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://platform@db/xagent")
        assert is_platform_database_url("mariadb+pymysql://other@DB:3306/xagent")

    def test_invalid_platform_url_fails_closed(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://platform@db:notaport/xagent")
        with pytest.raises(ValueError, match="platform database"):
            validate_external_database_url(
                "postgresql://other@public.example/analytics"
            )


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


def test_saved_sql_connection_rejects_platform_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://platform@db.example/app")
    db = MagicMock()

    with pytest.raises(ValueError, match="platform database"):
        set_sql_connection(
            db,
            user_id=1,
            name="platform",
            connection_url="postgresql+psycopg://other@DB.EXAMPLE:5432/app",
        )

    db.commit.assert_not_called()


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

    @pytest.mark.parametrize("max_rows", [None, 10])
    @pytest.mark.parametrize("where", ["", "WHERE 1 = 0"])
    def test_select_columns_come_from_cursor_metadata(self, max_rows, where):
        result = _execute_sql_query_local(
            "test",
            f"SELECT 1 AS id, 'test' AS name {where}",
            connection_url="sqlite:///:memory:",
            max_rows=max_rows,
            max_result_bytes=4096,
        )

        assert result["success"] is True
        assert result["columns"] == ["id", "name"]
        assert result["rows"] == ([] if where else [{"id": 1, "name": "test"}])
        assert result["row_count"] == (0 if where else 1)

    def test_model_facing_description_explains_worker_export_limits(self):
        from xagent.core.tools.adapters.vibe.sql_tool import SqlQueryTool

        tools = {tool.name: tool for tool in SqlQueryTool().get_tools()}
        description = " ".join(tools["execute_sql_query"].description.split())

        assert (
            "worker exports remain subject to row and serialized-result byte limits"
            in description
        )
        assert "Local execution streams exports" in description
        assert "LIMIT, filters, or pagination" in description
        assert "avoid response size limits" not in description

    def test_execute_sql_query_no_connection(self, monkeypatch):
        """Test error when connection not found."""
        # Clear any existing XAGENT_EXTERNAL_DB_* variables
        for key in list(os.environ.keys()):
            if key.startswith("XAGENT_EXTERNAL_DB_"):
                monkeypatch.delenv(key)
        # The function raises ValueError when connection not found
        with pytest.raises(ValueError, match="not found"):
            execute_sql_query("missing", "SELECT 1")

    @patch("xagent.core.tools.core.sql_tool.execute_sql_query_remote")
    @patch("xagent.core.tools.core.sql_tool.create_engine")
    def test_configured_worker_prevents_local_execution(
        self, mock_create_engine, mock_remote, monkeypatch
    ):
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")
        monkeypatch.setenv("XAGENT_SQL_WORKER_URL", "http://sql-worker:8090")
        mock_remote.return_value = {"success": True, "rows": []}

        result = execute_sql_query("test", "SELECT 1")

        assert result == {"success": True, "rows": []}
        mock_create_engine.assert_not_called()
        mock_remote.assert_called_once_with(
            connection_name="test",
            connection_url="sqlite:///:memory:",
            query="SELECT 1",
            output_file=None,
            workspace=None,
        )

    def test_local_worker_row_limit_is_fail_closed(self, tmp_path):
        db_path = tmp_path / "external.db"
        with sqlite3.connect(db_path) as connection:
            connection.execute("CREATE TABLE rows (value INTEGER)")
            connection.executemany("INSERT INTO rows VALUES (?)", [(1,), (2,), (3,)])

        with pytest.raises(ValueError, match="2 row limit"):
            _execute_sql_query_local(
                "test",
                "SELECT value FROM rows ORDER BY value",
                connection_url=f"sqlite:///{db_path}",
                max_rows=2,
                max_result_bytes=1024,
            )

    def test_local_worker_result_byte_limit_is_fail_closed(self):
        with pytest.raises(ValueError, match="byte limit"):
            _execute_sql_query_local(
                "test",
                "SELECT 'a long value' AS value",
                connection_url="sqlite:///:memory:",
                max_rows=10,
                max_result_bytes=16,
            )

    def test_sqlite_attach_is_rejected_by_authorizer(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")
        attached = tmp_path / "other.db"
        with pytest.raises(DatabaseError, match="not authorized"):
            execute_sql_query("test", f"ATTACH DATABASE '{attached}' AS other")

    def test_sqlite_attach_word_in_string_or_identifier_is_allowed(self, monkeypatch):
        monkeypatch.setenv("XAGENT_EXTERNAL_DB_TEST", "sqlite:///:memory:")
        result = execute_sql_query("test", "SELECT 'attach' AS \"attach\"")
        assert result["success"] is True
        assert result["rows"] == [{"attach": "attach"}]

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
        mock_result.keys.return_value = ["id", "name"]
        mock_row = Mock()
        mock_row._mapping = {"id": 1, "name": "test"}
        mock_result.all.return_value = [mock_row]
        mock_conn.execute.return_value = mock_result

        result = execute_sql_query("test", "SELECT * FROM users")
        assert result["success"] is True
        assert result["row_count"] == 1
        assert result["columns"] == ["id", "name"]
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
        with (
            patch.dict("sys.modules", {"pyarrow": None}),
            pytest.raises(ImportError, match="pyarrow"),
        ):
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
