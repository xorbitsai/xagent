"""Tests for ``interaction_requests_table_exists``, the schema-presence gate
the two purge implementations and the checkpoint prune path use to stay safe
on a deployment upgraded to a migration revision before
``task_interaction_requests`` exists.

Named ``_gate`` rather than ``test_task_interaction_schema.py`` to avoid
colliding with that already-existing file (added by #1209): it pins the
model's own CHECK/FK/shape inventory, an unrelated concern to the presence
predicate tested here.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_interaction_schema_shared import (
    tables_excluding_interaction_requests,
)
from xagent.web.models.database import Base
from xagent.web.services.task_interaction_schema import (
    interaction_requests_table_exists,
)


def _postgres_url() -> str | None:
    return os.getenv("XAGENT_TEST_POSTGRES_URL") or os.getenv(
        "POSTGRES_TEST_DATABASE_URL"
    )


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


def test_sqlite_full_create_all_has_the_table(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'full.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        assert interaction_requests_table_exists(session) is True
    finally:
        session.close()
        engine.dispose()


def test_sqlite_filtered_create_all_lacks_the_table(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'filtered.db'}")
    Base.metadata.create_all(
        bind=engine, tables=tables_excluding_interaction_requests()
    )
    session = sessionmaker(bind=engine)()
    try:
        assert interaction_requests_table_exists(session) is False
    finally:
        session.close()
        engine.dispose()


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------


@pytest.fixture()
def postgres_engine():
    url = _postgres_url()
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()


@pytest.mark.postgresql
def test_postgres_full_create_all_has_the_table(postgres_engine) -> None:
    Base.metadata.create_all(bind=postgres_engine)
    session = sessionmaker(bind=postgres_engine)()
    try:
        assert interaction_requests_table_exists(session) is True
    finally:
        session.close()


@pytest.mark.postgresql
def test_postgres_filtered_create_all_lacks_the_table(postgres_engine) -> None:
    Base.metadata.create_all(
        bind=postgres_engine, tables=tables_excluding_interaction_requests()
    )
    session = sessionmaker(bind=postgres_engine)()
    try:
        assert interaction_requests_table_exists(session) is False
    finally:
        session.close()
