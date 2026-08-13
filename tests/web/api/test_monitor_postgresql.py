"""Monitoring endpoints against a real PostgreSQL database.

``monitor.py`` reads fields out of ``TraceEvent.data`` (a ``Column(JSON)``)
through a per-dialect helper. Its PostgreSQL branch is the one branch the
SQLite suite never executes, and it shipped ``~?`` -- an operator PostgreSQL
does not have for ``json``, ``jsonb`` or ``text``. Every query through that
branch therefore failed on PostgreSQL while SQLite CI stayed green, and each
call site swallowed the error into an empty result (#1149).

The two symptoms differ by endpoint, which is why all three are covered here:

- ``/monitor/stats`` raised. Its ``except`` substitutes ``active_models = 0``,
  but the failed statement leaves the transaction aborted, so the *next*
  query in the same session (the ``llm_call_end`` scan) raised
  ``PendingRollbackError`` outside any local handler and the request became a
  500.
- ``/monitor/popular-tools`` and ``/monitor/model-stats`` issue no further
  query after their handler, so they returned HTTP 200 with an empty list --
  a dashboard of zeros with nothing but a log line to show for it.

Since #1248 the column is ``jsonb``, which decodes escapes to native text at
INSERT and therefore rejects the payload shapes that used to poison reads --
the NUL escape and either half of an unpaired UTF-16 surrogate. That moves
what this file can and must pin:

- the hazardous shapes can no longer be seeded at all; a class of tests here
  asserts the INSERT itself fails, which is the invariant the migration
  exists to establish;
- the write-side sanitizer (``web/utils/json_payload_sanitizer.py``) must
  turn each of those shapes into something the column accepts and ``->>``
  reads back;
- the endpoints are exercised over storable payloads only. A *valid*
  surrogate pair is not a hazard and must survive, so a non-BMP payload is
  seeded alongside text that merely looks like an escape.

The read guard in ``get_json_field_expression`` still exists for databases
whose migration has not run. With ``jsonb`` it can never match, so no test
over the model's own table can reach its drop path any more --
``TestReadGuardAgainstNativeJson`` therefore builds a throwaway table with
a native ``json`` column and runs the guard against that, which keeps the
branch's normalize-then-match SQL executing on real PostgreSQL rather than
only in the dialect-independent mirrors in ``tests/web/test_monitor_api.py``.

Fixture pattern copied from
``tests/web/services/test_task_status_storage_postgresql.py`` (skip-if-unset
via ``XAGENT_TEST_POSTGRES_URL``; CI provides it in the PostgreSQL job).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Iterator

import pytest
from sqlalchemy import JSON, Column, Integer, MetaData, Table, select
from sqlalchemy import text as sa_text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from xagent.web.api.monitor import (
    get_json_field_expression,
    get_model_stats,
    get_monitoring_stats,
    get_popular_tools,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.user import User
from xagent.web.utils.json_payload_sanitizer import (
    REPLACEMENT_CHARACTER,
    sanitize_json_payload,
)

# String values a ``json`` column accepted but ``->>`` then refused to convert
# to text, taking the whole query down with it (#1149). ``jsonb`` refuses them
# at INSERT instead (#1248), which is what the rejection tests below pin.
# Built with ``chr`` because an editor will happily turn an escape sequence
# into the character it names, and a lone surrogate character is not encodable
# as UTF-8.
NUL_PAYLOAD = chr(0x0000)  # -> ``unsupported Unicode escape sequence``
LONE_HIGH_SURROGATE = chr(0xD800)  # -> ``invalid input syntax for type json``
LONE_LOW_SURROGATE = chr(0xDC00)  # same, from the other side of the pair

# A non-BMP character, which ``json.dumps`` writes as a *valid* surrogate
# pair. It must NOT be dropped: emoji in an LLM payload are ordinary.
NON_BMP_CHAR = chr(0x1F600)

# Text that merely looks like an escape: the JSON carries a doubled backslash,
# so ``->>`` reads it back without complaint. Dropping it would cost a real
# monitoring row for nothing.
BACKSLASH = chr(92)
LITERAL_ESCAPE_TEXT = BACKSLASH + "u0000"

# The nasty one: literal text shaped like a high surrogate escape, immediately
# followed by a genuinely unpaired low surrogate. Read naively the two look
# like a valid pair, the row slips through, and ``->>`` fails the whole query.
SURROGATE_BEHIND_LITERAL = BACKSLASH + "ud83d" + LONE_LOW_SURROGATE


@pytest.fixture()
def pg_session() -> Iterator[Session]:
    """Session against a real PostgreSQL, where the json operators are real.

    SQLite resolves the helper's other branch, so a dialect-specific operator
    error can only surface here.
    """
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def _seed_admin_with_trace_events(db: Session) -> User:
    """One admin, one task, and the trace events the three endpoints count.

    Every payload here is one ``jsonb`` stores: the hazardous shapes are
    unstorable since #1248 and are covered by ``TestJsonbRejectsHazards``
    instead. The paired-surrogate and literal-escape payloads are the ones
    nothing may drop.
    """
    admin = User(username="monitor-pg-admin", password_hash="hash", is_admin=True)
    db.add(admin)
    db.commit()
    db.refresh(admin)

    task = Task(user_id=admin.id, title="monitor-pg-task")
    db.add(task)
    db.commit()
    db.refresh(task)

    now = datetime.now()
    emoji_model = f"emoji-model {NON_BMP_CHAR}"
    payloads: list[tuple[str, dict[str, Any]]] = [
        ("llm_call_start", {"model_name": "gpt-4o", "step_id": "s1", "attempt": 1}),
        ("llm_call_start", {"model_name": "gpt-4o", "step_id": "s2", "attempt": 1}),
        ("llm_call_start", {"model_name": "claude-opus", "step_id": "s3"}),
        # A valid surrogate pair is not a hazard and must still count.
        ("llm_call_start", {"model_name": emoji_model, "step_id": "s4"}),
        # Text that only looks like an escape extracts fine, so nothing may
        # treat it as a hazard.
        (
            "llm_call_start",
            {"model_name": "literal-escape-model", "note": LITERAL_ESCAPE_TEXT},
        ),
        ("tool_execution_start", {"tool_name": "calculator"}),
        ("tool_execution_start", {"tool_name": "calculator"}),
        ("tool_execution_start", {"tool_name": "web_search"}),
    ]
    for index, (event_type, data) in enumerate(payloads):
        db.add(
            TraceEvent(
                task_id=task.id,
                event_id=f"monitor-pg-{index}",
                event_type=event_type,
                timestamp=now,
                data=data,
            )
        )
    db.commit()
    return admin


@pytest.mark.postgresql
async def test_monitoring_stats_counts_active_models_on_postgresql(
    pg_session: Session,
) -> None:
    """/monitor/stats reports the real model count instead of failing.

    Before the fix this raised a 500: the rejected operator aborted the
    transaction and the next query in the handler could not run.
    """
    admin = _seed_admin_with_trace_events(pg_session)

    stats = await get_monitoring_stats(db=pg_session, current_user=admin)

    # gpt-4o, claude-opus, the emoji model and the literal-escape model: a
    # valid surrogate pair and text that merely looks like an escape both
    # count as ordinary models.
    assert stats["activeModels"] == 4


@pytest.mark.postgresql
async def test_popular_tools_returns_usage_counts_on_postgresql(
    pg_session: Session,
) -> None:
    """/monitor/popular-tools returns real rows instead of an empty list."""
    admin = _seed_admin_with_trace_events(pg_session)

    tools = await get_popular_tools(db=pg_session, current_user=admin)

    assert [(entry["name"], entry["usage_count"]) for entry in tools] == [
        ("calculator", 2),
        ("web_search", 1),
    ]


@pytest.mark.postgresql
async def test_model_stats_returns_per_model_calls_on_postgresql(
    pg_session: Session,
) -> None:
    """/monitor/model-stats returns real rows instead of an empty list."""
    admin = _seed_admin_with_trace_events(pg_session)

    stats = await get_model_stats(db=pg_session, current_user=admin)

    assert {entry["name"]: entry["total_tasks"] for entry in stats} == {
        "gpt-4o": 2,
        "claude-opus": 1,
        f"emoji-model {NON_BMP_CHAR}": 1,
        "literal-escape-model": 1,
    }


@pytest.mark.postgresql
class TestJsonbRejectsHazards:
    """The invariant #1248 establishes: the column can no longer hold a
    payload ``->>`` cannot read back. Each shape that used to fail a whole
    monitoring query now fails its own INSERT instead, which is a local,
    attributable error rather than a silent dashboard of zeros.
    """

    @pytest.mark.parametrize(
        ("label", "payload"),
        [
            ("nul", {"model_name": "nul-model", "note": NUL_PAYLOAD}),
            ("high", {"model_name": "high-model", "note": LONE_HIGH_SURROGATE}),
            ("low", {"model_name": "low-model", "note": LONE_LOW_SURROGATE}),
            # The escape in the extracted field rather than beside it:
            # position must not matter.
            ("in-field", {"model_name": f"tainted-{NUL_PAYLOAD}"}),
            # A real unpaired surrogate hiding behind text that imitates the
            # other half of a pair.
            (
                "hidden",
                {"model_name": "hidden-model", "note": SURROGATE_BEHIND_LITERAL},
            ),
        ],
    )
    def test_unstorable_payload_is_rejected_at_insert(
        self, pg_session: Session, label: str, payload: dict[str, Any]
    ) -> None:
        admin = User(
            username=f"reject-{label}-admin", password_hash="hash", is_admin=True
        )
        pg_session.add(admin)
        pg_session.commit()
        task = Task(user_id=admin.id, title=f"reject-{label}")
        pg_session.add(task)
        pg_session.commit()

        pg_session.add(
            TraceEvent(
                task_id=task.id,
                event_id=f"reject-{label}",
                event_type="llm_call_start",
                timestamp=datetime.now(),
                data=payload,
            )
        )
        with pytest.raises(DBAPIError):
            pg_session.commit()
        pg_session.rollback()

    def test_sanitized_payload_is_storable_and_readable(
        self, pg_session: Session
    ) -> None:
        """The other half of the contract: what the write-side sanitizer
        produces from a hazardous payload must both store and read back --
        otherwise the sanitizer would only be trading one failure for
        another.
        """
        admin = User(username="sanitized-admin", password_hash="hash", is_admin=True)
        pg_session.add(admin)
        pg_session.commit()
        task = Task(user_id=admin.id, title="sanitized")
        pg_session.add(task)
        pg_session.commit()

        hazardous = {
            "model_name": f"m{NUL_PAYLOAD}{LONE_HIGH_SURROGATE}",
            "note": LONE_LOW_SURROGATE,
        }
        pg_session.add(
            TraceEvent(
                task_id=task.id,
                event_id="sanitized-1",
                event_type="llm_call_start",
                timestamp=datetime.now(),
                data=sanitize_json_payload(hazardous),
            )
        )
        pg_session.commit()

        stats = pg_session.execute(
            TraceEvent.__table__.select().where(TraceEvent.event_id == "sanitized-1")
        ).one()
        assert stats.data["model_name"] == f"m{REPLACEMENT_CHARACTER * 2}"


@pytest.mark.postgresql
class TestJsonbRoundTripFidelity:
    """What the checkpoint blob path depends on: a payload written through
    the sanitizer must come back from ``jsonb`` as the *same JSON*, not
    merely the same numbers. ``trace_message_storage`` re-hashes what it
    reads and rejects a mismatch as corruption, so an int/float retype on
    the way through the column would cost a task its checkpoint.
    """

    def _task(self, db: Session, label: str) -> Task:
        admin = User(username=f"{label}-admin", password_hash="hash", is_admin=True)
        db.add(admin)
        db.commit()
        task = Task(user_id=admin.id, title=label)
        db.add(task)
        db.commit()
        return task

    def _round_trip(self, db: Session, task: Task, payload: dict[str, Any]) -> Any:
        task_id = int(task.id)
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=f"roundtrip-{task_id}",
                event_type="llm_call_start",
                timestamp=datetime.now(),
                data=sanitize_json_payload(payload),
            )
        )
        db.commit()
        # Read through a Core select on a fresh transaction, so what comes
        # back is what the column returned and not the dict still held in
        # the session's identity map.
        db.expunge_all()
        row = db.execute(
            TraceEvent.__table__.select().where(
                TraceEvent.__table__.c.task_id == task_id
            )
        ).one()
        return row.data

    def test_large_float_survives_as_the_same_json(self, pg_session: Session) -> None:
        task = self._task(pg_session, "roundtrip-float")
        payload = {"cost": 1e16, "ratio": 0.1, "count": 3}

        stored = self._round_trip(pg_session, task, payload)

        # The sanitizer converted 1e16 up front, so what comes back matches
        # byte for byte under the canonical form the blob hash uses.
        expected = sanitize_json_payload(payload)
        assert json.dumps(stored, sort_keys=True) == json.dumps(
            expected, sort_keys=True
        )

    def test_unsanitized_large_float_would_change_type(
        self, pg_session: Session
    ) -> None:
        """The control: without the sanitizer's normalization the retype is
        real, so the test above is pinning behaviour and not a tautology."""
        task = self._task(pg_session, "roundtrip-control")
        pg_session.add(
            TraceEvent(
                task_id=task.id,
                event_id="roundtrip-control",
                event_type="llm_call_start",
                timestamp=datetime.now(),
                data={"cost": 1e16},
            )
        )
        pg_session.commit()
        pg_session.expunge_all()

        row = pg_session.execute(
            TraceEvent.__table__.select().where(
                TraceEvent.event_id == "roundtrip-control"
            )
        ).one()
        assert isinstance(row.data["cost"], int)

    def test_negative_zero_survives_as_the_same_json(self, pg_session: Session) -> None:
        """The other end of the numeric range: PostgreSQL numeric has no
        signed zero, so an unsanitized -0.0 comes back as 0.0 and breaks
        the hash the same way a large float does."""
        task = self._task(pg_session, "roundtrip-negzero")
        payload = {"delta": -0.0, "ratio": 0.5}

        stored = self._round_trip(pg_session, task, payload)

        expected = sanitize_json_payload(payload)
        assert json.dumps(stored, sort_keys=True) == json.dumps(
            expected, sort_keys=True
        )

    def test_unsanitized_negative_zero_would_lose_its_sign(
        self, pg_session: Session
    ) -> None:
        """Control for the test above: the sign really is dropped by the
        column, so the normalization is load-bearing."""
        task = self._task(pg_session, "roundtrip-negzero-control")
        pg_session.add(
            TraceEvent(
                task_id=task.id,
                event_id="roundtrip-negzero-control",
                event_type="llm_call_start",
                timestamp=datetime.now(),
                data={"delta": -0.0},
            )
        )
        pg_session.commit()
        pg_session.expunge_all()

        row = pg_session.execute(
            TraceEvent.__table__.select().where(
                TraceEvent.event_id == "roundtrip-negzero-control"
            )
        ).one()
        # json.dumps of the *unsanitized* payload writes "-0.0"; what the
        # column hands back no longer does.
        assert json.dumps(row.data["delta"]) == "0.0"

    def test_ordinary_payload_survives_unchanged(self, pg_session: Session) -> None:
        task = self._task(pg_session, "roundtrip-plain")
        payload = {
            "model_name": f"emoji {NON_BMP_CHAR}",
            "nested": {"items": ["a", "b"], "ok": True, "score": 1.5},
            "none": None,
        }

        stored = self._round_trip(pg_session, task, payload)

        assert json.dumps(stored, sort_keys=True) == json.dumps(payload, sort_keys=True)


@pytest.mark.postgresql
class TestReadGuardAgainstNativeJson:
    """The guard's drop path, executed against a real ``json`` column.

    ``get_json_field_expression``'s PostgreSQL branch exists for databases
    whose migration has not run yet. Since #1248 the model's own column is
    ``jsonb``, so no test above can plant a payload the guard has to drop --
    which left the branch's normalize-then-match SQL (the escaped-backslash
    stand-in, the pair strip, the alternation) with no real-PostgreSQL
    execution at all, only Python mirrors of its semantics.

    This class restores that coverage by building a throwaway table whose
    payload column is native ``json``, so the hazardous rows are storable
    again, and running the guard expression against it. It is deliberately
    not the model's table: the point is to exercise the branch on the shape
    it was written for, and the model must stay ``jsonb``.
    """

    TABLE = "monitor_guard_native_json"

    @pytest.fixture()
    def native_json_table(self, pg_session: Session) -> Iterator[Any]:
        pg_session.execute(
            sa_text(
                f"CREATE TABLE {self.TABLE} "
                "(id integer PRIMARY KEY, data json NOT NULL)"
            )
        )
        pg_session.commit()
        table = Table(
            self.TABLE,
            MetaData(),
            Column("id", Integer, primary_key=True),
            Column("data", JSON, nullable=False),
        )
        try:
            yield table
        finally:
            pg_session.rollback()
            pg_session.execute(sa_text(f"DROP TABLE IF EXISTS {self.TABLE}"))
            pg_session.commit()

    def _seed(self, db: Session, rows: list[tuple[int, str]]) -> None:
        for row_id, payload_json_text in rows:
            db.execute(
                sa_text(
                    f"INSERT INTO {self.TABLE} (id, data) "
                    "VALUES (:id, CAST(:payload AS json))"
                ),
                {"id": row_id, "payload": payload_json_text},
            )
        db.commit()

    def test_hazardous_rows_are_dropped_and_benign_rows_survive(
        self, pg_session: Session, native_json_table: Any
    ) -> None:
        """One query over a mix: without the guard this raises and takes
        every row with it, which is exactly #1149."""
        backslash = chr(92)
        self._seed(
            pg_session,
            [
                (1, '{"model_name": "plain"}'),
                # Hazards the guard must null out.
                (2, '{"model_name": "nul' + backslash + 'u0000"}'),
                (3, '{"model_name": "high' + backslash + 'ud800"}'),
                (4, '{"model_name": "low' + backslash + 'udc00"}'),
                # An orphan hiding behind text shaped like the other half of
                # a pair -- the case the escaped-backslash stand-in exists
                # for. Read naively the two look like a valid pair.
                (
                    5,
                    '{"model_name": "hidden'
                    + backslash
                    + backslash
                    + "ud83d"
                    + backslash
                    + 'udc00"}',
                ),
                # Benign and must survive: a valid pair, and text that only
                # looks like an escape.
                (
                    6,
                    '{"model_name": "emoji'
                    + backslash
                    + "ud83d"
                    + backslash
                    + 'ude00"}',
                ),
                (7, '{"model_name": "literal' + backslash + backslash + 'u0000"}'),
            ],
        )

        expression = get_json_field_expression(
            native_json_table.c.data, "model_name", pg_session
        )
        rows = pg_session.execute(
            select(native_json_table.c.id, expression).order_by(native_json_table.c.id)
        ).fetchall()

        assert dict(rows) == {
            1: "plain",
            2: None,
            3: None,
            4: None,
            5: None,
            6: f"emoji{NON_BMP_CHAR}",
            7: "literal" + backslash + "u0000",
        }
