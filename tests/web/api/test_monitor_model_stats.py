"""``/monitor/model-stats`` usage-rate arithmetic.

``get_model_stats`` reported ``usage_rate: 100.0`` for every model no matter
how the calls were actually distributed (#1245). The grand total was summed
into ``total_calls`` and then shadowed by the loop variable of the same name,
so the rate divided a model's own call count by itself -- ``n / n * 100``, a
constant 100 for every row with any traffic at all. The dashboard renders the
field directly as a percentage, so the distribution it exists to show was
replaced by a column of identical numbers.

Nothing here is dialect-specific: the shadowing is plain Python downstream of
the query, so the SQLite path exercises it. The sibling PostgreSQL suite
covers the dialect-dependent JSON extraction feeding it and asserts only
per-model call counts.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from xagent.web.api.monitor import get_model_stats
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.user import User

from .conftest import _direct_db_session

pytestmark = pytest.mark.usefixtures("_test_db")


def _seed_model_calls(
    db: Session, model_names: list[str], *, owner: str, is_admin: bool = True
) -> User:
    """One user with one ``llm_call_start`` event per name in ``model_names``.

    Repeat a name to give that model more calls. ``owner`` must differ between
    calls within a test -- ``User.username`` is ``unique=True``, so reusing one
    raises rather than merely mixing two users' traffic together.
    """
    user = User(username=owner, password_hash="hash", is_admin=is_admin)
    db.add(user)
    db.commit()
    db.refresh(user)

    task = Task(user_id=user.id, title=f"{owner}-task")
    db.add(task)
    db.commit()
    db.refresh(task)

    # Aware UTC, matching what production writes into this tz-aware column
    # (``websocket.py`` persists ``datetime.now(timezone.utc)``). No assertion
    # here depends on it -- ``get_model_stats`` never reads ``timestamp``, and
    # the column is only populated because it is ``nullable=False``.
    now = datetime.now(timezone.utc)
    for index, model_name in enumerate(model_names):
        db.add(
            TraceEvent(
                task_id=task.id,
                event_id=f"{owner}-{index}",
                event_type="llm_call_start",
                timestamp=now,
                data={"model_name": model_name},
            )
        )
    db.commit()
    return user


async def test_usage_rate_is_each_model_share_of_all_calls() -> None:
    """Each rate is the model's share of the total, not a constant 100.

    A lopsided split is the point: with equal counts every candidate formula
    agrees, so only an uneven distribution tells a real share of the total
    apart from the constant the handler used to return.
    """
    db = _direct_db_session()
    try:
        admin = _seed_model_calls(
            db,
            ["gpt-4o", "gpt-4o", "gpt-4o", "claude-opus"],
            owner="lopsided-admin",
        )

        stats = await get_model_stats(db=db, current_user=admin)

        assert {entry["name"]: entry["usage_rate"] for entry in stats} == {
            "gpt-4o": 75.0,
            "claude-opus": 25.0,
        }
        # The call counts the rates are derived from, so a regression that
        # fixed the ratio by miscounting cannot hide here.
        assert {entry["name"]: entry["total_tasks"] for entry in stats} == {
            "gpt-4o": 3,
            "claude-opus": 1,
        }
    finally:
        db.close()


async def test_single_model_uses_the_whole_budget() -> None:
    """One model with all the traffic really is at 100%.

    Kept alongside the lopsided case so the fix is pinned from both sides: the
    old code returned 100.0 here too, and a fix that merely stopped returning
    100.0 would be just as wrong.
    """
    db = _direct_db_session()
    try:
        admin = _seed_model_calls(db, ["gpt-4o"], owner="solo-admin")

        stats = await get_model_stats(db=db, current_user=admin)

        assert [(entry["name"], entry["usage_rate"]) for entry in stats] == [
            ("gpt-4o", 100.0)
        ]
    finally:
        db.close()


async def test_nameless_calls_stay_out_of_the_denominator() -> None:
    """Calls the response omits must not shrink the rates it reports.

    The query filters a NULL model name but not an empty one, so an empty name
    reaches Python and is dropped there. Once the total became a real
    denominator, dropping such a row from the output while still counting it
    would leave the reported rates summing to less than 100.
    """
    db = _direct_db_session()
    try:
        admin = _seed_model_calls(db, ["gpt-4o", "", ""], owner="nameless-calls-admin")

        stats = await get_model_stats(db=db, current_user=admin)

        assert [(entry["name"], entry["usage_rate"]) for entry in stats] == [
            ("gpt-4o", 100.0)
        ]
    finally:
        db.close()


async def test_no_calls_yields_an_empty_list() -> None:
    """No traffic returns ``[]`` rather than raising or reporting a zero row.

    The handler used to end with ``if not result: return []`` ahead of
    ``return result``, which could not be told apart from it -- both arms
    returned an equal empty list. That branch is gone; this pins the behaviour
    it was there to express.
    """
    db = _direct_db_session()
    try:
        member = _seed_model_calls(db, [], owner="no-traffic-user")

        assert await get_model_stats(db=db, current_user=member) == []
    finally:
        db.close()


async def test_non_admin_shares_are_scoped_to_their_own_calls() -> None:
    """A regular user's denominator is their own traffic, not the fleet's.

    Non-admins get a ``task_id`` subquery restricting the scan to their own
    tasks, so both the numerator and the denominator shrink to what they can
    see. Now that the denominator is read, a leak would not just add rows the
    caller should not see -- it would silently restate the rates for the rows
    they should. The other user's traffic is sized so the two readings cannot
    be confused: fleet-wide would be 90.0/10.0, own-traffic 75.0/25.0.
    """
    db = _direct_db_session()
    try:
        _seed_model_calls(db, ["gpt-4o"] * 6, owner="other-tenant")
        member = _seed_model_calls(
            db,
            ["gpt-4o", "gpt-4o", "gpt-4o", "claude-opus"],
            owner="member",
            is_admin=False,
        )

        stats = await get_model_stats(db=db, current_user=member)

        assert {entry["name"]: entry["usage_rate"] for entry in stats} == {
            "gpt-4o": 75.0,
            "claude-opus": 25.0,
        }
        # The counts too, so a leak cannot hide behind a coincidental ratio.
        assert {entry["name"]: entry["total_tasks"] for entry in stats} == {
            "gpt-4o": 3,
            "claude-opus": 1,
        }
    finally:
        db.close()
