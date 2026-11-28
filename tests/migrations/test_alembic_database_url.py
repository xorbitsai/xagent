from sqlalchemy import URL, create_engine, engine_from_config, inspect

from xagent.db.config import create_alembic_config
from xagent.db.migration import try_upgrade_db


def test_alembic_config_preserves_postgresql_password():
    url = URL.create(
        "postgresql+psycopg2",
        username="migration_user",
        password="test-password",
        host="localhost",
        database="xagent",
    )
    engine = create_engine(url)
    try:
        config = create_alembic_config(engine)
        migration_engine = engine_from_config(
            config.get_section(config.config_ini_section, {})
        )
        try:
            assert migration_engine.url.password == url.password
        finally:
            migration_engine.dispose()
    finally:
        engine.dispose()


def test_alembic_config_preserves_encoded_postgresql_url():
    url = URL.create(
        "postgresql+psycopg2",
        username="migration_user",
        password="test%@/password",
        host="localhost",
        database="xagent",
        query={"options": "-csearch_path=isolated_schema"},
    )
    engine = create_engine(url)
    try:
        config = create_alembic_config(engine)
        assert config.get_main_option("sqlalchemy.url") == url.render_as_string(
            hide_password=False
        )
    finally:
        engine.dispose()


def test_upgrade_db_accepts_percent_encoded_sqlite_uri(tmp_path):
    db_path = tmp_path / "migration database.db"
    engine = create_engine(f"sqlite:///{db_path.as_uri()}?mode=rwc&uri=true")
    try:
        try_upgrade_db(engine)
        assert inspect(engine).has_table("alembic_version")
        assert db_path.is_file()
    finally:
        engine.dispose()
