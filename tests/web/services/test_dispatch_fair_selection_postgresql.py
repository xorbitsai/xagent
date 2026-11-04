"""Fair-share selection against PostgreSQL: the listing, and which index each query uses."""

import json
import random
import re
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from tests.alembic.lock_retry_support import driver_engines
from tests.shared.postgres_disposable import disposable_database_factory
from tests.web.services.fair_selection_support import (
    WORLDS,
    FakeClock,
    World,
    attempt,
    cap_of,
    play,
    scan,
)
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_transport as transport
from xagent.web.services.task_dispatch_cap import DispatchCapSnapshot, DispatchPolicy

INDEX = "ix_task_commands_claimable_fair"
PRIORITY_INDEX = "ix_task_commands_claimable_fair_priority"


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


TABLE = "task_execution_commands"
INDEX_SCANS = {"Index Scan", "Index Only Scan"}


def plan_root(plan) -> dict:
    return (json.loads(plan) if isinstance(plan, str) else plan)[0]["Plan"]


def nodes_under(node) -> list[dict]:
    """``node`` and every node below it."""
    found = [node]
    for child in node.get("Plans", ()):
        found.extend(nodes_under(child))
    return found


def driving_scan(node) -> dict:
    """The scan that feeds the query: down the outer child of each node to a relation."""
    while "Relation Name" not in node:
        outer = [
            child
            for child in node.get("Plans", ())
            if child.get("Parent Relationship") == "Outer"
        ]
        assert outer, f"no outer child under {node['Node Type']}"
        node = outer[0]
    return node


def seq_scans_of_commands(node) -> list[dict]:
    return [
        found
        for found in nodes_under(node)
        if found["Node Type"] == "Seq Scan" and found.get("Relation Name") == TABLE
    ]


def assert_reads_only_through(node, index: str) -> None:
    """Every read of the command table below ``node`` is an index scan of ``index``,
    and there is at least one."""
    reads = [
        found for found in nodes_under(node) if found.get("Relation Name") == TABLE
    ]
    assert reads, f"{node['Node Type']} reads no command table"
    assert [(read["Node Type"], read.get("Index Name")) for read in reads] == [
        (read["Node Type"], index) for read in reads
    ]
    assert all(read["Node Type"] in INDEX_SCANS for read in reads)


def assert_fed_by(root: dict, index: str) -> dict:
    """The scan that drives ``root`` is an index scan of ``index``; returns it."""
    scan_node = driving_scan(root)
    assert scan_node["Node Type"] in INDEX_SCANS
    assert scan_node["Relation Name"] == TABLE
    assert scan_node.get("Index Name") == index
    return scan_node


def explain(engine, statement: str):
    with engine.connect() as connection:
        return connection.exec_driver_sql(
            "EXPLAIN (FORMAT JSON) " + statement.replace("%", "%%")
        ).scalar_one()


def claimable_query(db, cap):
    """The claimable query of a scan, as ``_find_command_candidate`` builds it."""
    return transport._dispatcher_candidate_query(
        db,
        runner_id="me",
        command_db_id=None,
        shared=True,
        cap=cap,
        now=datetime(2026, 10, 23, tzinfo=timezone.utc),
    ).filter(TaskExecutionCommand.task_id.notin_(()))


def literal_sql(query) -> str:
    return str(
        query.statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


@pytest.mark.postgresql
def test_the_plan_of_the_tenant_listing_reads_the_tenant_index(busy_table) -> None:
    root = plan_root(
        explain(busy_table, str(transport._LIST_QUEUED_TENANTS_POSTGRESQL))
    )

    (union,) = [n for n in nodes_under(root) if n["Node Type"] == "Recursive Union"]
    for relationship in ("Outer", "Inner"):
        (branch,) = [
            child
            for child in union["Plans"]
            if child.get("Parent Relationship") == relationship
        ]
        assert_reads_only_through(branch, INDEX)
    # The listing has no anti-join: no step of it scans the table in order.
    assert seq_scans_of_commands(root) == []


@pytest.mark.postgresql
def test_the_plan_of_a_tenants_window_query_reads_the_priority_index(
    busy_table,
) -> None:
    cap = DispatchCapSnapshot(
        policy=DispatchPolicy(limit=30, interactive_reserve=3),
        active_ids=frozenset(),
    )
    with sa.orm.Session(busy_table) as db:
        statement = literal_sql(
            transport._in_tenant_window(claimable_query(db, cap), "user:5", "batch")
        )

    assert "fairness_key >=" in statement and "fairness_key <=" in statement
    root = plan_root(explain(busy_table, statement))
    # The claimable query's check for an earlier unfinished command of the same task
    # is planned as a hash anti-join that reads the command table; the oldest-first
    # lookup without the switch has the same node, so it is not a property of the
    # window.
    feed = assert_fed_by(root, PRIORITY_INDEX)
    limits = [
        n
        for n in nodes_under(root)
        if n["Node Type"] == "Limit" and any(m is feed for m in nodes_under(n))
    ]
    assert limits, "the window's LIMIT is not above its feeding scan"
    window = min(limits, key=lambda n: len(nodes_under(n)))
    assert seq_scans_of_commands(window) == []


@pytest.mark.postgresql
def test_the_plan_of_the_older_control_lookup_reads_the_kind_index(busy_table) -> None:
    cap = DispatchCapSnapshot(
        policy=DispatchPolicy(limit=30, interactive_reserve=3),
        active_ids=frozenset(range(1, 11)),
    )
    with busy_table.connect() as connection:
        newest = connection.execute(
            sa.text("SELECT max(id) FROM task_execution_commands")
        ).scalar_one()
    with sa.orm.Session(busy_table) as db:
        statement = literal_sql(
            transport._older_controls(claimable_query(db, cap), newest).limit(1)
        )

    # The claimable query's check for an earlier unfinished command of the same task
    # is planned as a hash anti-join that reads the command table; the oldest-first
    # lookup without the switch has the same node, so it is not a property of the
    # older-control lookup.
    assert_fed_by(
        plan_root(explain(busy_table, statement)), "ix_task_commands_claimable_kind"
    )


RUNNING = range(1, 11)
UNFINISHED_INDEX = "ix_task_commands_unfinished_task"
ID_INDEX = "ix_task_commands_claimable_id"
TASK_ORDER_INDEX = "ix_task_commands_task_order"
PRIMARY_KEY_INDEX = "task_execution_commands_pkey"


@pytest.fixture
def flood_table(pg):
    """Ten running tasks behind a backlog of another tenant's older commands.

    Task 1..10 each hold five thousand settled commands and one newer pending
    follow-up. Tasks 11..2000 hold fifty thousand pending starts that are all
    older than those follow-ups.
    """
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
                "SELECT g, 1, 't', 'PENDING' FROM generate_series(1, 2000) g"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO task_execution_commands (task_id, command_id, kind, "
                "payload, status, attempt_count, dispatch_class, fairness_key, "
                "created_at, updated_at) "
                "SELECT 11 + (g % 1990), 'q' || g, 'start', '{}', 'pending', 0, "
                "'batch', 'user:2', now(), now() "
                "FROM generate_series(1, 50000) g"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO task_execution_commands (task_id, command_id, kind, "
                "payload, status, attempt_count, dispatch_class, fairness_key, "
                "created_at, updated_at) "
                "SELECT (g % 10) + 1, 'h' || g, 'message', '{}', 'completed', 1, "
                "'batch', 'user:1', now(), now() "
                "FROM generate_series(1, 50000) g"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO task_execution_commands (task_id, command_id, kind, "
                "payload, status, attempt_count, dispatch_class, fairness_key, "
                "created_at, updated_at) "
                "SELECT g, 'f' || g, 'message', '{}', 'pending', 0, "
                "'batch', 'user:1', now(), now() "
                "FROM generate_series(1, 10) g"
            )
        )
    with pg.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(sa.text("VACUUM (ANALYZE) task_execution_commands"))
        connection.execute(sa.text("ANALYZE tasks"))
    return pg


def read_units(root: dict) -> list[dict]:
    """The reads of the command table that ran, one entry per read unit.

    A unit is an index scan or a sequential scan of the table; a Bitmap Heap Scan
    together with the Bitmap Index Scans below it is one unit, whose index names
    and index conditions are those of its bitmap scans. Nodes that never ran
    (zero loops) are left out. ``rows`` counts every row the unit looked at,
    kept or filtered out, over all its loops.
    """
    units = []
    for node in nodes_under(root):
        if node.get("Relation Name") != TABLE or node.get("Actual Loops", 0) <= 0:
            continue
        below = nodes_under(node)
        indexes = [n for n in below if "Index Name" in n]
        units.append(
            {
                "type": node["Node Type"],
                "indexes": {n["Index Name"] for n in indexes},
                "conds": " ".join(n.get("Index Cond", "") for n in indexes),
                "rows": (node["Actual Rows"] + node.get("Rows Removed by Filter", 0))
                * node["Actual Loops"],
            }
        )
    return units


def explain_analyze(engine, statement: str) -> dict:
    """The executed plan of ``statement``, run without parallel workers."""
    with engine.connect() as connection:
        connection.exec_driver_sql("SET max_parallel_workers_per_gather = 0")
        plan = connection.exec_driver_sql(
            "EXPLAIN (ANALYZE, FORMAT JSON) " + statement.replace("%", "%%")
        ).scalar_one()
    return plan_root(plan)


def newest_id(engine) -> int:
    with engine.connect() as connection:
        return connection.execute(
            sa.text("SELECT max(id) FROM task_execution_commands")
        ).scalar_one()


@pytest.mark.postgresql
def test_the_plan_of_the_older_lookup_among_running_tasks_is_bounded_by_their_commands(
    flood_table, monkeypatch
) -> None:
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    cap = DispatchCapSnapshot(
        policy=DispatchPolicy(limit=30, interactive_reserve=3),
        active_ids=frozenset(RUNNING),
    )
    before = newest_id(flood_table) + 1
    with sa.orm.Session(flood_table) as db:
        statement = literal_sql(
            transport._older_of_tasks(
                lambda: claimable_query(db, cap), before, frozenset(RUNNING)
            ).limit(1)
        )

    units = read_units(explain_analyze(flood_table, statement))

    assert any(
        UNFINISHED_INDEX in unit["indexes"] and "running_task.id" in unit["conds"]
        for unit in units
    )
    assert not any(TASK_ORDER_INDEX in unit["indexes"] for unit in units)
    # An ordered scan that is not keyed by task walks the backlog until it meets a
    # row of a running task; the primary key and the claimable-id index are fine
    # only when they are entered at a given id.
    for unit in units:
        if unit["indexes"] & {ID_INDEX, PRIMARY_KEY_INDEX}:
            assert re.search(r"\(id = ", unit["conds"]), unit
    assert not [unit for unit in units if unit["type"] == "Seq Scan"]
    # Bound, with R running tasks and U of their unfinished commands older than
    # the cut-off (here every running task holds one unfinished command, its
    # follow-up, so U = R = 10). Rows actually read: each task's lookup reads its
    # first unfinished command (U rows over R loops); the check for an earlier
    # unfinished command of the same task finds none before it (0 rows); the
    # outer query reads the chosen row by id (1 row) and its check finds none
    # (0 rows). That is U + 1. PostgreSQL reports Actual Rows and Rows Removed by
    # Filter as per-loop averages rounded to an integer, so multiplied back by
    # the loops each of the two counts of a node can overstate it by less than
    # half its loops: under R for each of the two nodes that loop once per task,
    # nothing for the outer nodes. The bound is U + 1 + 2R = 31. A scan of the
    # backlog or of the settled history reads tens of thousands of rows.
    running, unfinished = len(RUNNING), len(RUNNING)
    assert sum(unit["rows"] for unit in units) <= unfinished + 1 + 2 * running


@pytest.mark.postgresql
@pytest.mark.parametrize("shared", ["true", "false"])
def test_the_same_task_check_of_the_oldest_claimable_lookup_reads_the_unfinished_index(
    flood_table, monkeypatch, shared
) -> None:
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", shared)
    cap = DispatchCapSnapshot(
        policy=DispatchPolicy(limit=30, interactive_reserve=3),
        active_ids=frozenset(RUNNING),
    )
    with sa.orm.Session(flood_table) as db:
        statement = literal_sql(claimable_query(db, cap).limit(1))

    units = read_units(explain_analyze(flood_table, statement))

    # The check that a candidate has no earlier unfinished command of its task
    # goes through the index of unfinished commands and not through the index of
    # every command of the task in order.
    assert any(UNFINISHED_INDEX in unit["indexes"] for unit in units)
    assert not any(TASK_ORDER_INDEX in unit["indexes"] for unit in units)


@pytest.fixture
def waiting_start_then_cancel(pg, monkeypatch):
    """One task whose pending start waits for an admission ticket, then a cancel.

    The bucket has room and holds no ticket, no pacing row exists and the start's
    ticket has no owner, so the start is claimable, and the cancel is claimable
    too because a cancel may pass a start that waits for a ticket. Another
    tenant's newer task follows.
    """
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    world = World(pg)
    start = world.add("a", kind="start")
    task = world.task_of(start)
    cancel = world.follow_up(task, "a", kind="cancel")
    world.add("b")
    with world.sessions() as db:
        db.add(TaskAdmissionBucket(key="bucket", capacity=1, max_pending=10))
        db.flush()
        db.add(
            TaskAdmissionTicket(
                command_id=start,
                bucket_key="bucket",
                task_id=task,
                owner_attempt_id=None,
            )
        )
        db.commit()
    return world, task, start, cancel


@pytest.mark.postgresql
def test_of_two_commands_of_a_running_task_that_can_pass_the_older_is_found(
    waiting_start_then_cancel,
) -> None:
    world, task, start, cancel = waiting_start_then_cancel
    cap = DispatchCapSnapshot(
        policy=DispatchPolicy(limit=30, interactive_reserve=3),
        active_ids=frozenset({task}),
    )
    with world.sessions() as db:
        # Both commands pass every filter of the claimable query, so the lookup
        # has two rows of the one task to choose from.
        assert [
            row.id
            for row in claimable_query(db, cap).filter(
                TaskExecutionCommand.task_id == task
            )
        ] == [start, cancel]

        found = transport._older_of_tasks(
            lambda: claimable_query(db, cap), cancel + 10, frozenset({task})
        ).first()

    assert found is not None and found.id == start


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
@pytest.mark.parametrize(
    "raw", [None, "false", "garbage"], ids=["unset", "false", "garbage"]
)
def test_with_the_switch_off_a_scan_lists_no_tenant_and_takes_the_oldest(
    pg_world, monkeypatch, raw
) -> None:
    world, _ = pg_world
    ids = [world.add(*row[:2], kind=row[2]) for row in WORLDS["three"]]
    cap = cap_of(occupancy={"user:1": 1})
    if raw is None:
        monkeypatch.delenv(SWITCH)
    else:
        monkeypatch.setenv(SWITCH, raw)
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    assert transport._new_fair_attempt() is None

    chosen, seen = statements_sent(world, lambda: scan(cap, attempt=None))

    assert chosen == ids[0]
    assert not any("RECURSIVE" in statement for statement in seen)
    assert not any("fairness_key >=" in statement for statement in seen)
    assert transport._fair_state.since_fallback == 0
    # The same world with the switch on is served differently.
    monkeypatch.setenv(SWITCH, "true")
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    assert scan(cap, attempt=attempt()) == ids[1]


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
