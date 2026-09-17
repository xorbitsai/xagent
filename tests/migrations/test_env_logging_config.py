"""Regression test for the alembic ``env.py`` logging configuration.

``env.py`` calls ``logging.config.fileConfig(config.config_file_name)`` when
Alembic is driven from a config *file* (the standalone ``alembic upgrade
head`` CLI, and this test suite, which builds its ``Config`` from
``alembic.ini``). ``fileConfig`` defaults ``disable_existing_loggers`` to
``True``, which disables *every* logger already registered in
``logging.Logger.manager.loggerDict`` that is not one of the names
explicitly configured in ``alembic.ini`` (``root``, ``sqlalchemy``,
``alembic``).

Normal application startup does **not** go through this branch:
``xagent.db.config.create_alembic_config`` builds an in-memory Alembic
``Config`` with no filename, so ``config.config_file_name`` is ``None`` and
``fileConfig`` is never called there. See
``test_env_logging_config_production.py`` for the test pinning that fact.

What this file's test *is* affected by: when this test suite constructs a
file-backed ``Config`` and runs a real migration, the old default
(``disable_existing_loggers=True``) would disable every other already
registered logger for the rest of the test process, contaminating
subsequent tests that assert on log output -- 131 of them, historically.
This test drives a real Alembic upgrade (exercising the actual ``env.py``)
against a throwaway SQLite database and asserts that a logger which existed
before the upgrade is not disabled afterward.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

project_root = Path(__file__).parent.parent.parent


@pytest.fixture
def preexisting_logger() -> Generator[logging.Logger, None, None]:
    """A logger that exists before the migration runs, restored afterward."""
    name = "xagent.test.env_logging_config.preexisting"
    logger = logging.getLogger(name)
    original_disabled = logger.disabled
    try:
        yield logger
    finally:
        logger.disabled = original_disabled


@pytest.fixture
def sqlite_alembic_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """An Alembic config pointed at a throwaway SQLite database."""
    db_path = tmp_path / "env_logging_config.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


@pytest.mark.integration
def test_alembic_env_does_not_disable_preexisting_loggers(
    preexisting_logger: logging.Logger,
    sqlite_alembic_cfg: Config,
) -> None:
    """Running the real migration env.py must not disable existing loggers.

    This exercises the file-backed ``Config`` path (the same one the
    standalone ``alembic upgrade head`` CLI and this test suite use).
    ``preexisting_logger`` stands in for any ``xagent.*`` logger created by
    normal module imports before this migration runs. Alembic only
    explicitly configures ``root``, ``sqlalchemy``, and ``alembic`` in
    ``alembic.ini``; every other pre-existing logger must be left alone.
    """
    # Base tables are created by SQLAlchemy in production before migrations
    # run; mirror that so `command.upgrade` exercises the same path as
    # `tests/migrations/test_migration_integration.py`.
    from sqlalchemy import create_engine

    from xagent.web.models.database import Base

    sqlalchemy_url = sqlite_alembic_cfg.get_main_option("sqlalchemy.url")
    assert sqlalchemy_url is not None
    engine = create_engine(sqlalchemy_url)
    Base.metadata.create_all(bind=engine)
    engine.dispose()

    assert not preexisting_logger.disabled, "sanity check before the upgrade"

    command.upgrade(sqlite_alembic_cfg, "head")

    assert not preexisting_logger.disabled, (
        "alembic's env.py disabled a pre-existing logger; fileConfig() must "
        "be called with disable_existing_loggers=False"
    )
