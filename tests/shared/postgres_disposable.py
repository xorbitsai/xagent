"""Shared disposable-PostgreSQL-database plumbing for migration test suites.

Extracted from the near-identical ``postgresql_engine_factory`` fixture
duplicated across tests/alembic/test_20260809_add_task_interaction_requests.py
and tests/migrations/test_task_interaction_requests_schema_parity.py: same
mint-via-CREATE-DATABASE/drop-on-teardown body, same ``creator=`` workaround,
same per-database error accounting. Centralizing it here means a fix to the
teardown accounting (or the ``creator=`` workaround) lands once instead of
twice.

Every engine here is built with an explicit ``creator=`` -- a plain function
calling ``psycopg2.connect()`` directly -- instead of letting SQLAlchemy's
psycopg2 dialect open the DBAPI connection itself. Measured against this
module's target server: SQLAlchemy's own connect path intermittently (and,
in one run, for 80 straight seconds of retries) fails to authenticate
against a database created moments earlier, with the server returning a
password-authentication FATAL; a bare ``psycopg2.connect()`` with identical
host/port/user/password/dbname against the same freshly created database
always succeeds. The ``creator=`` callable sidesteps whatever SQLAlchemy-side
difference triggers that, while every other Engine/Connection/Inspector
behavior (create_all, alembic Operations, inspection) stays identical.
"""

from __future__ import annotations

import importlib.util
import os
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator

import psycopg2
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import make_url


def psycopg2_kwargs(base_url: str, dbname: str | None = None) -> dict[str, object]:
    parsed = make_url(base_url)
    return {
        "host": parsed.host,
        "port": parsed.port,
        "user": parsed.username,
        "password": parsed.password,
        "dbname": dbname if dbname is not None else parsed.database,
    }


@contextmanager
def disposable_database_factory(
    prefix: str,
) -> Iterator[Callable[[str], sa.engine.Engine]]:
    """Yield a callable that mints a brand-new, disposable PostgreSQL
    database per call (via CREATE DATABASE against the server
    XAGENT_TEST_POSTGRES_URL names) and drops every database it created on
    exit. Skips the whole test if the env var is unset.

    ``prefix`` becomes the leading segment of every minted database's name
    (``f"{prefix}_{tag}_{uuid4().hex[:10]}"``), so two suites sharing one
    PostgreSQL server never mint colliding names even if they happen to pass
    the same ``tag`` to the returned factory.
    """
    base_url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not base_url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")

    def _admin_connection():
        conn = psycopg2.connect(**psycopg2_kwargs(base_url))
        conn.autocommit = True
        return conn

    minted: list[tuple[str, sa.engine.Engine]] = []

    def _make(tag: str) -> sa.engine.Engine:
        dbname = f"{prefix}_{tag}_{uuid.uuid4().hex[:10]}"
        admin_conn = _admin_connection()
        try:
            admin_conn.cursor().execute(f'CREATE DATABASE "{dbname}"')
        finally:
            admin_conn.close()

        connect_kwargs = psycopg2_kwargs(base_url, dbname)

        def _connect(_kwargs: dict[str, object] = connect_kwargs):
            return psycopg2.connect(**_kwargs)

        engine = sa.create_engine("postgresql://", creator=_connect)
        minted.append((dbname, engine))
        return engine

    try:
        yield _make
    finally:
        for dbname, engine in minted:
            try:
                engine.dispose()
            except Exception as exc:  # noqa: BLE001
                # Never stop here: an engine that will not dispose must not
                # block the drops below, or every database after it leaks.
                warnings.warn(
                    f"could not dispose the engine for {dbname}: {exc!r}",
                    stacklevel=2,
                )

        undropped: list[str] = []
        if minted:
            try:
                admin_conn = _admin_connection()
            except Exception as exc:
                raise RuntimeError(
                    "cannot reach the PostgreSQL server to drop the "
                    "disposable databases this fixture created; drop them "
                    "by hand: " + ", ".join(name for name, _ in minted)
                ) from exc
            try:
                # One autocommit connection serves every drop: a failed
                # DROP DATABASE leaves it usable, so a database still in
                # use costs its own name, not the rest of the list.
                for dbname, _engine in minted:
                    try:
                        admin_conn.cursor().execute(
                            f'DROP DATABASE IF EXISTS "{dbname}"'
                        )
                    except Exception as exc:  # noqa: BLE001
                        undropped.append(f"{dbname} ({exc!r})")
            finally:
                admin_conn.close()

        if undropped:
            raise RuntimeError(
                "disposable databases left behind on the server; drop them "
                "by hand: " + "; ".join(undropped)
            )


def load_migration_module(path: Path, name: str = "migration_under_test") -> ModuleType:
    """Load the Alembic revision module at ``path`` under ``name`` without
    going through the revision chain or requiring it be importable as a
    package (its filename starts with a digit)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
