import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from xagent.core.storage import get_default_db_url

# Import all models to ensure they are registered with Base.metadata
# Type checking is disabled for these imports as they are dynamically loaded by Alembic
# flake8: noqa: E402
from xagent.web import models as web_models  # noqa: F401
from xagent.web.models.database import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
#
# This branch only runs when alembic is driven from a config *file*: the
# standalone `alembic upgrade head` CLI and tests/migrations/ (which build
# their Config from alembic.ini). Normal app startup goes through
# xagent.db.config.create_alembic_config(), which builds an in-memory Config
# with no filename, so config.config_file_name is None there and this branch
# is skipped -- production is not affected either way.
# ``disable_existing_loggers`` defaults to True, which would disable every
# already-registered logger not explicitly listed in alembic.ini's
# [loggers] section (root/sqlalchemy/alembic) for the rest of the process.
# That's what was poisoning the test suite: migration tests ran fileConfig
# and disabled loggers that later tests then asserted against.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Use our models' MetaData for autogenerate support
target_metadata = Base.metadata


ALEMBIC_VERSION_TABLE = "alembic_version"
ALEMBIC_VERSION_LENGTH = 255


def ensure_wide_alembic_version_table(
    connection: Connection, *, commit: bool = False
) -> None:
    """Ensure Alembic can store this project's long revision identifiers."""
    changed = False
    inspector = sqlalchemy_inspect(connection)
    tables = inspector.get_table_names()

    if ALEMBIC_VERSION_TABLE not in tables:
        connection.execute(
            text(
                "CREATE TABLE alembic_version "
                "(version_num VARCHAR(255) NOT NULL, "
                "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
            )
        )
        changed = True
        if commit:
            connection.commit()
        return

    columns = {
        column["name"]: column
        for column in inspector.get_columns(ALEMBIC_VERSION_TABLE)
    }
    version_num = columns.get("version_num")
    if version_num is None:
        return

    current_length = getattr(version_num["type"], "length", None)
    if current_length is None or current_length >= ALEMBIC_VERSION_LENGTH:
        return

    if connection.dialect.name == "postgresql":
        connection.execute(
            text(
                "ALTER TABLE alembic_version ALTER COLUMN version_num TYPE varchar(255)"
            )
        )
        changed = True
    elif connection.dialect.name == "sqlite":
        connection.execute(text("PRAGMA foreign_keys=OFF"))
        connection.execute(
            text(
                "CREATE TABLE alembic_version_wide "
                "(version_num VARCHAR(255) NOT NULL, "
                "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO alembic_version_wide (version_num) "
                "SELECT version_num FROM alembic_version"
            )
        )
        connection.execute(text("DROP TABLE alembic_version"))
        connection.execute(
            text("ALTER TABLE alembic_version_wide RENAME TO alembic_version")
        )
        connection.execute(text("PRAGMA foreign_keys=ON"))
        changed = True

    if changed and commit:
        connection.commit()


# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    if url is None:
        # Respect DATABASE_URL environment variable
        url = os.getenv("DATABASE_URL")
        if url is None:
            url = get_default_db_url()

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Check if connection is provided via config.attributes
    connection = config.attributes.get("connection", None)

    if connection is not None:
        # Use provided connection
        ensure_wide_alembic_version_table(connection)
        is_postgresql = connection.dialect.name == "postgresql"
        if is_postgresql and connection.in_transaction():
            # PostgreSQL online index migrations use an Alembic autocommit
            # block.  Reflection above starts an implicit transaction; close
            # it before Alembic takes ownership of per-migration boundaries.
            connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            transaction_per_migration=is_postgresql,
        )
        with context.begin_transaction():
            context.run_migrations()
    else:
        # Fallback: create new connection using URL from config
        configuration = config.get_section(config.config_ini_section, {})
        if configuration.get("sqlalchemy.url") is None:
            # Respect DATABASE_URL environment variable
            url = os.getenv("DATABASE_URL")
            if url is None:
                url = get_default_db_url()
            configuration["sqlalchemy.url"] = url

        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            ensure_wide_alembic_version_table(connection, commit=True)

        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                transaction_per_migration=connection.dialect.name == "postgresql",
            )

            with context.begin_transaction():
                context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
