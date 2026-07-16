import logging
from typing import Any, cast

from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from sqlalchemy import Engine, inspect

from .config import create_alembic_config

logger = logging.getLogger(__name__)


def is_database_empty(engine: Engine) -> bool:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return len(tables) == 0


def get_alembic_revision(engine: Engine) -> str | None:
    """Get the current Alembic revision from the database."""
    with engine.connect() as conn:
        context: Any = MigrationContext.configure(conn)
        return cast(str | None, context.get_current_revision())


def _check_revision_is_known(alembic_cfg: Any, engine: Engine, version: str) -> None:
    """Fail with an actionable message when the DB revision is unknown.

    A database whose ``alembic_version`` points at a revision missing from the
    installed package was almost always created or upgraded by a *newer*
    xagent release (e.g. a source checkout or Docker image sharing the same
    storage root). Alembic cannot downgrade without the newer migration
    scripts, so surface what happened and how to recover instead of letting
    ``command.upgrade`` fail with a bare "Can't locate revision" error.
    """
    script = ScriptDirectory.from_config(alembic_cfg)
    try:
        script.get_revisions(version)
    except CommandError:
        db_url = engine.url.render_as_string(hide_password=True)
        raise RuntimeError(
            f"The database at '{db_url}' is at schema revision '{version}', "
            "which this xagent installation does not know about. It was most "
            "likely created or upgraded by a newer version of xagent (for "
            "example a source checkout or Docker image using the same "
            "storage root). To fix this, upgrade xagent to that version or "
            "newer, or point DATABASE_URL / XAGENT_STORAGE_ROOT at a "
            "different database."
        ) from None


def try_upgrade_db(engine: Engine) -> None:
    """Upgrade database to latest migration (or stamp head for brand-new databases)."""
    try:
        logger.info("Starting database upgrade process")
        alembic_cfg = create_alembic_config(engine)
        version = get_alembic_revision(engine)

        # An empty-string version_num (tampered/corrupted alembic_version)
        # is treated like a missing revision: get_revisions("") dies with a
        # bare AssertionError instead of a catchable error.
        if not version:
            if is_database_empty(engine):
                logger.info("Creating new database, stamping to latest revision.")
                with engine.begin() as conn:
                    alembic_cfg.attributes["connection"] = conn
                    command.stamp(alembic_cfg, "head")
            else:
                raise RuntimeError(
                    "Database exists without alembic revision information. Please initialize the database schema version manually by running: alembic stamp <revision>"
                )
        else:
            _check_revision_is_known(alembic_cfg, engine, version)
            logger.info(f"Current version: {version}, upgrading to head")
            with engine.begin() as conn:
                alembic_cfg.attributes["connection"] = conn
                command.upgrade(alembic_cfg, "head")
    except Exception as e:
        logger.error(f"Automatic database upgrade failed: {e}")
        raise
