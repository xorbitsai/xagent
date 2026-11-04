"""A disposable PostgreSQL database for E2E tests.

With ``XAGENT_TEST_POSTGRES_URL`` set, each test gets a fresh database on that
server, dropped afterwards: its role needs CREATEDB, and the server must be
PostgreSQL 13 or newer (``DROP DATABASE ... WITH (FORCE)``). Other suites use
the same variable as their base database; these tests never touch it. Without
it a PostgreSQL 17 container is started per test, which needs a Docker daemon.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from uuid import uuid4

import psycopg2
import pytest
import sqlalchemy as sa

from tests.e2e.minio_harness import (
    _docker_available,
    _docker_client,
    run_container_with_dynamic_ports,
)

POSTGRES_PASSWORD = "xagent_test"
POSTGRES_DATABASE = "xagent_test"
# WHY: the image tag is a floating major, so a drifted pull would otherwise keep
# this suite green while silently testing against PostgreSQL 16.
MIN_SERVER_VERSION_NUM = 170000
# DROP DATABASE ... WITH (FORCE), for a server the caller provides.
MIN_EXTERNAL_SERVER_VERSION_NUM = 130000


def _external_database(server_url: str) -> Iterator[str]:
    server = sa.engine.make_url(server_url).set(drivername="postgresql+psycopg2")
    name = f"xagent_e2e_{uuid4().hex[:12]}"
    admin = sa.create_engine(server, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            version = int(
                connection.execute(sa.text("SHOW server_version_num")).scalar()
            )
            if version < MIN_EXTERNAL_SERVER_VERSION_NUM:
                pytest.fail(
                    "XAGENT_TEST_POSTGRES_URL must name PostgreSQL 13 or newer, "
                    f"got server_version_num={version}"
                )
            try:
                connection.execute(sa.text(f'CREATE DATABASE "{name}"'))
            except sa.exc.ProgrammingError as exc:
                if getattr(exc.orig, "pgcode", None) != "42501":
                    raise
                pytest.fail(
                    "XAGENT_TEST_POSTGRES_URL's role needs CREATEDB: each "
                    f"PostgreSQL E2E test creates its own database ({exc.orig})"
                )
        try:
            yield server.set(database=name).render_as_string(hide_password=False)
        finally:
            with admin.connect() as connection:
                # Hosts that did not shut down cleanly may still hold sessions.
                connection.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    finally:
        admin.dispose()


def _container_database() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("Requires reachable Docker daemon or XAGENT_TEST_POSTGRES_URL")

    client = _docker_client()
    container, host_ports = run_container_with_dynamic_ports(
        client,
        "postgres:17-bookworm",
        name=f"xagent-postgres-e2e-{uuid4().hex[:12]}",
        container_ports=("5432/tcp",),
        tmpfs={"/var/lib/postgresql/data": "rw,size=256m"},
        environment={
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DATABASE,
        },
    )
    host_port = host_ports["5432/tcp"]

    database_url = (
        "postgresql+psycopg2://postgres:"
        f"{POSTGRES_PASSWORD}@127.0.0.1:{host_port}/{POSTGRES_DATABASE}"
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                connection = psycopg2.connect(
                    host="127.0.0.1",
                    port=host_port,
                    user="postgres",
                    password=POSTGRES_PASSWORD,
                    dbname=POSTGRES_DATABASE,
                    connect_timeout=1,
                )
            except psycopg2.OperationalError:
                time.sleep(0.25)
            else:
                with connection.cursor() as cursor:
                    cursor.execute("SHOW server_version_num")
                    server_version_num = int(cursor.fetchone()[0])
                connection.close()
                if server_version_num < MIN_SERVER_VERSION_NUM:
                    raise RuntimeError(
                        "e2e fixture requires PostgreSQL 17 or newer, got "
                        f"server_version_num={server_version_num}"
                    )
                break
        else:
            raise RuntimeError("PostgreSQL did not become ready")

        yield database_url
    finally:
        container.remove(force=True)


@pytest.fixture
def postgres_url() -> Iterator[str]:
    """A ``postgresql+psycopg2`` URL of an empty database, dropped afterwards."""
    server_url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if server_url:
        yield from _external_database(server_url)
    else:
        yield from _container_database()
