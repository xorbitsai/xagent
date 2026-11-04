"""Fresh-state contract for the shared API test database."""

import pytest
from sqlalchemy import text

from xagent.web.models.database import get_engine, get_session_local
from xagent.web.models.user import User


@pytest.mark.parametrize("iteration", range(2))
def test_database_fixture_isolates_rows_schema_and_seed(
    _test_db: None, iteration: int
) -> None:
    with get_engine().begin() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one() == "wal"
        assert connection.execute(text("SELECT COUNT(*) FROM users")).scalar_one() == 0
        assert (
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE name = 'fixture_probe'")
            ).first()
            is None
        )
        assert (
            connection.execute(
                text(
                    "SELECT provider_name FROM oauth_providers WHERE provider_name = 'google'"
                )
            ).scalar_one()
            == "google"
        )
        assert (
            connection.execute(
                text(
                    "SELECT is_visible_in_connector FROM public_mcp_apps WHERE app_id = 'chrome-devtools'"
                )
            ).scalar_one()
            == 0
        )

        # Each invocation dirties rows, schema, and seed data. The next one
        # must still start from a complete, independent fresh database.
        connection.execute(text("CREATE TABLE fixture_probe (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text("DELETE FROM oauth_providers WHERE provider_name = 'google'")
        )
        connection.execute(
            text(
                "UPDATE public_mcp_apps SET is_visible_in_connector = 1 WHERE app_id = 'chrome-devtools'"
            )
        )

    with get_session_local()() as session:
        session.add(
            User(username="fixture-probe", password_hash="unused", is_admin=False)
        )
        session.commit()
