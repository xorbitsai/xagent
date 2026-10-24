"""Fair-share selection against PostgreSQL: the listing, and which index each query uses."""

import json
import random
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from tests.alembic.lock_retry_support import driver_engines
from tests.shared.postgres_disposable import disposable_database_factory
from tests.web.services.fair_selection_support import (
    CAP_STATES,
    FakeClock,
    World,
    play,
    scan,
    seed_world,
)
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.services import task_command_transport as transport
from xagent.web.services.task_dispatch_cap import DispatchCapSnapshot, DispatchPolicy

INDEX = "ix_task_commands_claimable_fair"


@pytest.fixture
def pg_factory():
    with disposable_database_factory("xagent_fair_selection") as make:
        yield make


@pytest.fixture(params=["psycopg2", "psycopg"])
def pg(request, pg_factory):
    engines = driver_engines(pg_factory, request.param)
    Base.metadata.create_all(engines["psycopg2"])
    try:
        yield engines[request.param]
    finally:
        if request.param != "psycopg2":
            engines[request.param].dispose()


QUEUED_ROWS = [
    # Sorts before every other key, so a listing that starts from it by mistake shows.
    ("0", "batch", "completed"),
    ("a", "batch", "pending"),
    ("a", "batch", "processing"),
    ("a", "interactive", "pending"),
    ("b", "batch", "completed"),
    ("b", "batch", "failed"),
    ("b", "interactive", "pending"),
    ("c", "batch", "pending"),
    ("user:10", "batch", "pending"),
    ("user:2", "interactive", "processing"),
    ("user:2", "batch", "pending"),
]


@pytest.mark.postgresql
def test_the_listing_gives_the_same_tenants_as_on_sqlite(pg, tmp_path) -> None:
    lite = sa.create_engine(f"sqlite:///{tmp_path / 'lite.db'}")
    Base.metadata.create_all(lite)
    found = {}
    for name, engine in (("sqlite", lite), ("postgresql", pg)):
        world = World(engine)
        for tenant, dispatch_class, status in QUEUED_ROWS:
            world.add(tenant, dispatch_class, status=status)
        # Sorts before every other queued key, so an anchor that counts it shows.
        world.add("0-only-a-pause", kind="pause")
        # And one between them, which only a step that counts it reaches.
        world.add("only-a-pause", kind="cancel")
        with world.sessions() as db:
            found[name] = transport._list_queued_tenants(db)
    lite.dispose()

    assert found["postgresql"] == found["sqlite"]
    # A pause takes no slot, so its key does not make a tenant.
    assert set(found["postgresql"]) == {"a", "b", "c", "user:10", "user:2"}
    assert set(found["postgresql"]["b"]) == {"interactive"}


@pytest.fixture
def busy_table(pg):
    """Fifty thousand queued commands, nine in ten of one tenant, among as many settled."""
    with pg.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO users (id, username, password_hash, is_admin) "
                "VALUES (1, 'u', 'h', false)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO tasks (id, user_id, title, status) "
                "SELECT g, 1, 't', 'PENDING' FROM generate_series(1, 100) g"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO task_execution_commands (task_id, command_id, kind, "
                "payload, status, attempt_count, dispatch_class, fairness_key, "
                "created_at, updated_at) "
                "SELECT (g % 100) + 1, 'h' || g, 'start', '{}', 'completed', 1, "
                "'batch', 'user:' || (g % 7 + 1), now(), now() "
                "FROM generate_series(1, 50000) g"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO task_execution_commands (task_id, command_id, kind, "
                "payload, status, attempt_count, dispatch_class, fairness_key, "
                "created_at, updated_at) "
                "SELECT (g % 100) + 1, 'q' || g, 'start', '{}', 'pending', 0, "
                "'batch', CASE WHEN g % 10 = 0 THEN 'user:' || (g % 7 + 2) "
                "ELSE 'user:1' END, now(), now() "
                "FROM generate_series(1, 50000) g"
            )
        )
    with pg.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(sa.text("VACUUM (ANALYZE) task_execution_commands"))
        connection.execute(sa.text("ANALYZE tasks"))
    return pg


def index_names(plan) -> set[str]:
    """Every index a plan reads, anywhere in its tree."""
    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            if "Index Name" in node:
                found.add(node["Index Name"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(plan) if isinstance(plan, str) else plan)
    return found


def explain(engine, statement: str):
    with engine.connect() as connection:
        return connection.exec_driver_sql(
            "EXPLAIN (FORMAT JSON) " + statement.replace("%", "%%")
        ).scalar_one()


@pytest.mark.postgresql
def test_the_plan_of_the_tenant_listing_reads_the_tenant_index(busy_table) -> None:
    plan = explain(busy_table, str(transport._LIST_QUEUED_TENANTS_POSTGRESQL))

    assert INDEX in index_names(plan)


@pytest.mark.postgresql
def test_the_plan_of_a_tenants_window_query_reads_the_tenant_index(busy_table) -> None:
    cap = DispatchCapSnapshot(
        policy=DispatchPolicy(limit=30, interactive_reserve=3),
        active_ids=frozenset(),
    )
    with sa.orm.Session(busy_table) as db:
        query = transport._in_tenant_window(
            transport._dispatcher_candidate_query(
                db,
                runner_id="me",
                command_db_id=None,
                shared=True,
                cap=cap,
                now=datetime(2026, 10, 23, tzinfo=timezone.utc),
            ),
            "user:5",
            "batch",
        )
        statement = str(
            query.statement.compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            )
        )

    assert "fairness_key >=" in statement and "fairness_key <=" in statement
    assert INDEX in index_names(explain(busy_table, statement))


SWITCH = "XAGENT_DISPATCHER_FAIR_SHARE"


@pytest.fixture
def pg_world(pg, monkeypatch):
    """A world on PostgreSQL with the switch on and a clock the test moves."""
    clock = FakeClock()
    built = World(pg)
    monkeypatch.setattr(transport, "_fair_clock", clock)
    monkeypatch.setattr(transport, "_fair_state", transport._FairState())
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    monkeypatch.setattr(transport, "get_runner_id", lambda: "me")
    monkeypatch.setattr(database_module, "_SessionLocal", built.sessions)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv(SWITCH, "true")
    return built, clock


def statements_sent(world, action):
    seen: list[str] = []

    def record(_conn, _cursor, statement, *_args):
        seen.append(" ".join(statement.split()))

    sa.event.listen(world.engine, "before_cursor_execute", record)
    try:
        return action(), seen
    finally:
        sa.event.remove(world.engine, "before_cursor_execute", record)


@pytest.mark.postgresql
@pytest.mark.parametrize("cap_state", sorted(CAP_STATES))
@pytest.mark.parametrize(
    "raw", [None, "false", "garbage"], ids=["unset", "false", "garbage"]
)
def test_with_the_switch_off_a_scan_sends_what_it_sent_before(
    pg_world, monkeypatch, cap_state, raw
) -> None:
    world, _ = pg_world
    if raw is None:
        monkeypatch.delenv(SWITCH)
    else:
        monkeypatch.setenv(SWITCH, raw)
    first = seed_world(world, "three")
    cap = transport.DispatchCapSnapshot(
        policy=DispatchPolicy(limit=2, interactive_reserve=1),
        active_ids=CAP_STATES[cap_state](first),
    )
    off = transport._new_fair_attempt()
    assert off is None

    chosen, seen = statements_sent(world, lambda: scan(cap, attempt=off))
    plain, plain_seen = statements_sent(world, lambda: scan(cap))

    assert chosen == plain
    assert seen == plain_seen
    assert not any("RECURSIVE" in statement for statement in seen)
    assert transport._fair_state.attempts == 0


@pytest.mark.postgresql
def test_the_dispatcher_follows_the_reference_over_random_worlds(pg_world) -> None:
    world, clock = pg_world
    for seed in range(30):
        play(
            world,
            clock,
            30_000 + seed,
            tenants=random.Random(seed).randint(2, 7),
            reserves=lambda limit: range(limit),
        )
