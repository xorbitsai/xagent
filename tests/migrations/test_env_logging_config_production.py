"""Pin that the production migration path never runs ``fileConfig``.

Companion to ``test_env_logging_config.py``, not a substitute. That file
proves the *fix* works on the path it actually changes: a file-backed Alembic
``Config``, used by the standalone ``alembic upgrade head`` CLI and by this
test suite.

This file pins something narrower. Production startup
(``xagent.web.models.database`` -> ``xagent.db.migration.try_upgrade_db`` ->
``xagent.db.config.create_alembic_config``) never reaches the ``fileConfig``
branch in ``env.py``, because ``create_alembic_config`` builds its ``Config``
in memory with no filename. ``config.config_file_name`` is ``None`` there, so
``if config.config_file_name is not None:`` skips ``fileConfig`` entirely --
with or without this PR's ``disable_existing_loggers=False``.

So this does NOT show the fix changed production behavior; production was
never affected. It guards against ``create_alembic_config`` later being given
a filename, which would route production through ``fileConfig`` and let the
process-wide logging mutation reappear.

Asserting on ``logger.disabled`` alone cannot catch that, and this file
deliberately does not rest on it: the PR sets
``disable_existing_loggers=False``, so even with ``fileConfig`` running no
logger would be disabled, and such an assertion would keep passing while the
guard it claims to provide was gone. The regression would still be real --
``fileConfig`` also replaces the root handler and resets the root level. The
checks below therefore assert the *cause* (no filename, no ``fileConfig``
call) rather than one downstream symptom this PR happens to suppress.
"""

from __future__ import annotations

import logging
import logging.config
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine

from xagent.db.config import create_alembic_config
from xagent.db.migration import try_upgrade_db


@pytest.fixture
def preexisting_logger() -> Generator[logging.Logger, None, None]:
    """A logger that exists before the migration runs, restored afterward."""
    name = "xagent.test.env_logging_config_production.preexisting"
    logger = logging.getLogger(name)
    original_disabled = logger.disabled
    try:
        yield logger
    finally:
        logger.disabled = original_disabled


def test_production_alembic_config_has_no_config_file_name(tmp_path: Path) -> None:
    """``create_alembic_config`` must not produce a file-backed ``Config``.

    This is the root fact the guard rests on: ``env.py`` gates ``fileConfig``
    on ``config.config_file_name is not None``, so a config built without a
    filename can never reach it.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'cfg.db'}")
    try:
        cfg = create_alembic_config(engine)
        assert cfg.config_file_name is None, (
            "create_alembic_config() returned a file-backed Config; env.py "
            "would then run fileConfig() during production startup and mutate "
            "process-wide logging state"
        )
    finally:
        engine.dispose()


@pytest.mark.integration
def test_try_upgrade_db_never_calls_fileconfig(
    preexisting_logger: logging.Logger,
    tmp_path: Path,
) -> None:
    """The production startup path must not invoke ``fileConfig`` at all.

    ``env.py`` re-imports ``fileConfig`` from ``logging.config`` on every run
    (Alembic execs the file fresh), so patching it there observes the real
    call. ``wraps`` keeps the genuine behaviour, so a regression fails on the
    assertion rather than by breaking the migration.
    """
    # An empty database exercises `command.stamp(alembic_cfg, "head")`,
    # `try_upgrade_db`'s path for a brand-new database. `env.py` runs either
    # way -- stamp and upgrade both trigger it -- so this covers the same
    # branch a populated database would.
    engine = create_engine(f"sqlite:///{tmp_path / 'try_upgrade_db_logging.db'}")

    assert not preexisting_logger.disabled, "sanity check before the upgrade"

    with patch.object(
        logging.config, "fileConfig", wraps=logging.config.fileConfig
    ) as spy_file_config:
        try:
            try_upgrade_db(engine)
        finally:
            engine.dispose()

    assert spy_file_config.call_count == 0, (
        "the production migration path called fileConfig() "
        f"{spy_file_config.call_count} time(s); create_alembic_config() must "
        "not pass a config_file_name, or env.py will mutate process-wide "
        "logging state during application startup"
    )

    # Secondary, and deliberately not load-bearing: with
    # disable_existing_loggers=False in place this holds even if fileConfig
    # had run. Kept only to document the symptom the original bug produced.
    assert not preexisting_logger.disabled
