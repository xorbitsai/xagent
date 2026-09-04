"""Real-PostgreSQL coverage for the row lock ``update_custom_api`` and
``delete_custom_api`` take on the ``CustomApi`` definition row before
propagating a rename or removing the link row, respectively -- and, for
the edit route, for the payloads that must *not* take it: a PUT that sets
only ``is_active`` writes the caller's own ``UserCustomApi`` link row and
never the definition row, so it reads the definition row without locking
it.

``FOR UPDATE`` is a no-op on SQLite -- every other suite in this repo runs
against SQLite, so nothing there can tell a genuine second-writer block
from a lock statement that silently does nothing. This file is the one
place that runs the real statement against a real server and proves it
actually blocks a second writer: two concurrent edits, an edit and a
concurrent delete both taking the same lock in the same order, and -- for
the edit route and the delete route each -- the companion path where the
row vanishes between the route's first read and this lock. The delete
route's version of that path additionally pins where the lock sits
relative to ``delete_team_connector``: it is taken first, so a vanished
row is refused before the hook is called at all.

Obtains its database through ``tests/shared/postgres_disposable.py``
(``disposable_database_factory``), the same disposable-CREATE-DATABASE
helper the other ``*_postgresql.py`` suites in this repo use, rather than
opening a hand-rolled connection. That helper reads
``XAGENT_TEST_POSTGRES_URL`` and skips the whole module when it is unset.

Also covers the caller's own ``UserCustomApi`` link row being revoked --
deleted, or stripped of its edit/delete flag -- by a second connection
while a locking route holds the definition row, one case per route.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    ConnectorDeleteDecision,
    set_connector_team_hooks,
)

pytestmark = pytest.mark.postgresql


@pytest.fixture()
def session_factory():
    with disposable_database_factory("xagent_custom_api_edit_lock") as make_database:
        engine = make_database("edit_lock")
        Base.metadata.create_all(bind=engine)
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def seeded(session_factory):
    """One owner, one owned Custom API, in their own committed rows."""
    with session_factory() as db:
        owner = User(
            username="custom-api-edit-lock-owner", password_hash="x", is_admin=False
        )
        db.add(owner)
        db.flush()
        api = CustomApi(
            name="edit-lock-target",
            url="https://example.com/api",
            method="GET",
        )
        db.add(api)
        db.flush()
        db.add(
            UserCustomApi(
                user_id=int(owner.id),
                custom_api_id=int(api.id),
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.commit()
        return int(owner.id), int(api.id)


def test_a_second_editor_blocks_until_the_first_editors_transaction_finishes(
    session_factory, seeded
) -> None:
    """Two real connections, barrier-synchronised: the second call's own
    lock statement must not return until the first call's transaction
    commits or rolls back -- the actual behavior ``FOR UPDATE`` exists to
    provide, and the one thing no SQLite-backed test can demonstrate.
    """
    import xagent.web.api.custom_api as custom_api_api
    from xagent.web.api.custom_api import CustomApiUpdate

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    lock_acquired = threading.Event()
    release_lock = threading.Event()
    second_finished = threading.Event()
    first_call_claimed = threading.Event()
    first_call_lock = threading.Lock()

    real_validate = custom_api_api.validate_runtime_config_declaration

    def paced_validate(**kwargs):
        # Both threads run through this same patched function once each
        # gets past its own lock statement. Only the call that gets here
        # *first* pauses: that is the first editor, holding its row lock
        # open via this still-uncommitted transaction. A second call that
        # reaches this point too (rather than staying blocked earlier,
        # inside its own lock statement) is not made to wait a second
        # time here -- pausing it too would prove nothing about the
        # database lock, only about this Python-level barrier.
        with first_call_lock:
            is_first_call = not first_call_claimed.is_set()
            first_call_claimed.set()
        if is_first_call:
            lock_acquired.set()
            assert release_lock.wait(timeout=10), "the first editor was never released"
        return real_validate(**kwargs)

    custom_api_api.validate_runtime_config_declaration = paced_validate
    session_a = session_factory()
    session_b = session_factory()
    try:

        def run_first():
            return custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(name="renamed-by-first-editor"),
                current_user=current_user,
                db=session_a,
            )

        def run_second():
            result = custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(description="edited-by-second-editor"),
                current_user=current_user,
                db=session_b,
            )
            second_finished.set()
            return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(run_first)
            assert lock_acquired.wait(timeout=5), (
                "the first editor never reached the lock"
            )

            second = executor.submit(run_second)
            # The second call's own lock statement should still be blocked
            # on the database at this point. If the lock were not real (or
            # a no-op, as on SQLite), the second call would sail through
            # almost immediately and this would flip to True.
            # Released in ``finally`` before anything is asserted: a
            # failure here means the second editor is still parked on the
            # first editor's lock, and leaving the first editor paused
            # would hang this executor's shutdown instead of failing the
            # test.
            try:
                assert not second_finished.wait(timeout=1.0), (
                    "the second editor finished before the first one "
                    "released the row -- the lock did not actually block it"
                )
            finally:
                release_lock.set()
            first.result(timeout=10)
            second.result(timeout=10)

        assert second_finished.is_set()
    finally:
        custom_api_api.validate_runtime_config_declaration = real_validate
        session_a.close()
        session_b.close()

    # Both writer sessions are closed above, so this reads what actually
    # committed rather than either session's own uncommitted view. The
    # block above proves the second editor waited; without this it would
    # still pass if neither editor's write survived.
    with session_factory() as fresh:
        row = fresh.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert row.name == "renamed-by-first-editor"
        assert row.description == "edited-by-second-editor"
        assert (
            fresh.query(UserCustomApi)
            .filter(
                UserCustomApi.custom_api_id == api_id,
                UserCustomApi.user_id == owner_id,
            )
            .one()
            .is_active
            is True
        )


def test_the_second_editors_rename_reports_the_first_editors_committed_name_as_old(
    session_factory, seeded
) -> None:
    """``rename_team_connector``'s ``old`` argument must be the name this
    transaction's own lock actually holds once acquired, not whatever the
    pre-lock read saw.

    Interleaving under test: the first editor renames the connector and
    commits while the second editor is blocked on the lock. The second
    editor then acquires the lock, refreshed to the first editor's
    committed name, and renames again. If the second editor's ``old``
    argument were captured before its own lock instead, it would report
    the connector's *original* name -- not the name every team agent's
    selector was already rewritten to by the first editor's own call --
    and the second rewrite would search for a name nothing holds anymore,
    leaving the first rewrite's result permanently dangling with no error.
    """
    import xagent.web.api.custom_api as custom_api_api
    from xagent.web.api.custom_api import CustomApiUpdate

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    lock_acquired = threading.Event()
    release_lock = threading.Event()
    second_finished = threading.Event()
    first_call_claimed = threading.Event()
    first_call_lock = threading.Lock()

    renamed_calls: list[tuple[str, str]] = []
    renamed_calls_lock = threading.Lock()

    def spy_renamed_hook(_db, _user_id, _connector_type, _connector_id, old, new):
        with renamed_calls_lock:
            renamed_calls.append((old, new))

    real_validate = custom_api_api.validate_runtime_config_declaration

    def paced_validate(**kwargs):
        with first_call_lock:
            is_first_call = not first_call_claimed.is_set()
            first_call_claimed.set()
        if is_first_call:
            lock_acquired.set()
            assert release_lock.wait(timeout=10), "the first editor was never released"
        return real_validate(**kwargs)

    custom_api_api.validate_runtime_config_declaration = paced_validate
    session_a = session_factory()
    session_b = session_factory()
    set_connector_team_hooks(renamed=spy_renamed_hook)
    try:

        def run_first():
            return custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(name="renamed-by-first-editor"),
                current_user=current_user,
                db=session_a,
            )

        def run_second():
            result = custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(name="renamed-by-second-editor"),
                current_user=current_user,
                db=session_b,
            )
            second_finished.set()
            return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(run_first)
            assert lock_acquired.wait(timeout=5), (
                "the first editor never reached the lock"
            )

            second = executor.submit(run_second)
            # Released in ``finally`` before anything is asserted -- see
            # the comment in
            # test_a_second_editor_blocks_until_the_first_editors_transaction_finishes.
            try:
                assert not second_finished.wait(timeout=1.0), (
                    "the second editor finished before the first one released the row"
                )
            finally:
                release_lock.set()
            first.result(timeout=10)
            second.result(timeout=10)

        assert renamed_calls == [
            ("edit-lock-target", "renamed-by-first-editor"),
            ("renamed-by-first-editor", "renamed-by-second-editor"),
        ]
    finally:
        set_connector_team_hooks()
        custom_api_api.validate_runtime_config_declaration = real_validate
        session_a.close()
        session_b.close()

    # The hook tuples above are in-process call records; this reads what
    # actually committed, so a rename that reported the right pair of names
    # and then failed to persist cannot pass.
    with session_factory() as fresh:
        row = fresh.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert row.name == "renamed-by-second-editor"
        assert (
            fresh.query(UserCustomApi)
            .filter(
                UserCustomApi.custom_api_id == api_id,
                UserCustomApi.user_id == owner_id,
            )
            .one()
            .is_active
            is True
        )


def test_a_row_that_vanishes_after_the_gate_but_before_the_lock_is_a_404_not_a_500(
    session_factory, seeded
) -> None:
    """The route's own access read can find the row and still lose a race
    to a concurrent delete that commits before the lock statement runs. The
    lock statement must see that as an ordinary "row not found" (``None``)
    and let the route's existing 404 handle it, rather than leaving the
    write path below to fail on a row that is no longer there.

    The concurrent delete is fired from a wrapper around this session's own
    ``query``, which lands it strictly between the two reads: the lock is
    the first statement in this route that asks for ``CustomApi`` (the
    access read above asks for ``UserCustomApi``, and reaches the
    definition row through its relationship rather than through
    ``query``), so the wrapper can recognise the lock query and nothing
    else. The recorded entity sequence is asserted rather than assumed, so
    this test cannot quietly pass on a 404 raised by the access gate
    instead of by the lock.
    """
    import xagent.web.api.custom_api as custom_api_api
    from xagent.web.api.custom_api import CustomApiUpdate

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    db = session_factory()
    real_query = db.query
    deleted_already = threading.Event()
    queried_entities: list[tuple] = []

    def delete_the_row_when_the_lock_query_starts(*entities, **kwargs):
        queried_entities.append(entities)
        if entities == (CustomApi,) and not deleted_already.is_set():
            deleted_already.set()
            with session_factory() as other:
                other.execute(
                    sa.delete(UserCustomApi).where(
                        UserCustomApi.custom_api_id == api_id
                    )
                )
                other.execute(sa.delete(CustomApi).where(CustomApi.id == api_id))
                other.commit()
        return real_query(*entities, **kwargs)

    db.query = delete_the_row_when_the_lock_query_starts
    try:
        with pytest.raises(HTTPException) as exc:
            custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(name="renamed-after-vanish"),
                current_user=current_user,
                db=db,
            )
        assert deleted_already.is_set(), "the concurrent delete never ran"
        assert exc.value.status_code == 404
        assert queried_entities[:2] == [(UserCustomApi,), (CustomApi,)], (
            "the concurrent delete must land after the access gate's own "
            "read and before the lock; otherwise the 404 under test could "
            f"be the access gate's rather than the lock's -- saw "
            f"{queried_entities!r}"
        )
    finally:
        db.close()


def test_a_delete_blocks_until_a_concurrent_edits_transaction_finishes(
    session_factory, seeded
) -> None:
    """``delete_custom_api`` takes the same definition-row lock an
    ``update_custom_api`` call that writes the definition row does, in the
    same order (``CustomApi`` first), precisely so that a concurrent
    edit/delete pair cannot deadlock (PostgreSQL 40P01): the edit's
    transaction below must finish -- commit or roll back -- before the
    delete's own lock statement can proceed, the same block
    ``test_a_second_editor_blocks_until_the_first_editors_transaction_
    finishes`` above demonstrates between two edits. Before
    delete_custom_api took this lock, its own child-row-first deletion
    order (see custom_api.py) and the PUT's parent-row-first order let the
    two routes take these same two rows in opposite orders.

    The concurrent editor below sets a definition field, not ``is_active``.
    That is load-bearing: an ``is_active``-only PUT writes the caller's own
    link row and takes no lock on the definition row at all, so it would
    hold nothing for this delete to wait on and this test would prove
    nothing about lock order.
    """
    import xagent.web.api.custom_api as custom_api_api
    from xagent.web.api.custom_api import CustomApiUpdate

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    lock_acquired = threading.Event()
    release_lock = threading.Event()
    second_finished = threading.Event()

    real_validate = custom_api_api.validate_runtime_config_declaration

    def paced_validate(**kwargs):
        # The editor's own lock statement runs earlier in the route, before
        # this patched call -- by the time this pauses, the editor already
        # holds the definition row lock in an uncommitted transaction.
        lock_acquired.set()
        assert release_lock.wait(timeout=10), "the editor was never released"
        return real_validate(**kwargs)

    custom_api_api.validate_runtime_config_declaration = paced_validate
    session_a = session_factory()
    session_b = session_factory()
    try:

        def run_edit():
            return custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(description="edited-by-the-concurrent-editor"),
                current_user=current_user,
                db=session_a,
            )

        def run_delete():
            result = custom_api_api.delete_custom_api(
                api_id,
                current_user=current_user,
                db=session_b,
            )
            second_finished.set()
            return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            editor = executor.submit(run_edit)
            assert lock_acquired.wait(timeout=5), "the editor never reached the lock"

            deleter = executor.submit(run_delete)
            # The delete's own lock statement should still be blocked on
            # the database at this point. If the two routes took this pair
            # of rows in opposite orders (or if either lock were a no-op,
            # as on SQLite), the delete would sail through almost
            # immediately and this would flip to True.
            # Released in ``finally`` before anything is asserted -- see
            # the comment in
            # test_a_second_editor_blocks_until_the_first_editors_transaction_finishes.
            try:
                assert not second_finished.wait(timeout=1.0), (
                    "the delete finished before the concurrent editor "
                    "released the row -- the lock did not actually block it"
                )
            finally:
                release_lock.set()
            editor.result(timeout=10)
            deleter.result(timeout=10)

        assert second_finished.is_set()
    finally:
        custom_api_api.validate_runtime_config_declaration = real_validate
        session_a.close()
        session_b.close()


def test_a_delete_whose_row_vanishes_after_the_gate_but_before_the_lock_is_a_404(
    session_factory, seeded
) -> None:
    """``delete_custom_api`` has the same window ``update_custom_api`` has:
    its access read can find the row and still lose a race to a concurrent
    delete that commits before the lock statement runs. Without the lock's
    own ``None`` guard the route reaches ``db.delete(None)``, which raises
    ``UnmappedInstanceError`` out of the write path -- an HTTP 500 where a
    404 is correct.

    The concurrent delete is fired from a wrapper around this session's own
    ``query``, which lands it strictly between the two reads: the lock is
    the first statement in this route that asks for ``CustomApi`` (the
    access read above asks for ``UserCustomApi``, and reaches the
    definition row through its relationship rather than through ``query``),
    so the wrapper can recognise the lock query and nothing else. The
    recorded entity sequence is asserted rather than assumed, so this test
    cannot quietly pass on a 404 raised by the access gate instead of by
    the lock.

    Two side-effect assertions come with it. The installed delete hook must
    never be called: the lock precedes ``delete_team_connector``, so a
    vanished row is refused before any hook-side mutation is attempted. And
    the route must not commit: the only ``db.commit()`` in this route is
    after the deletion, and the refusal returns before it.
    """
    import xagent.web.api.custom_api as custom_api_api

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    db = session_factory()
    real_query = db.query
    real_commit = db.commit
    deleted_already = threading.Event()
    queried_entities: list[tuple] = []
    commits: list[str] = []
    hook_calls: list[int] = []

    def spy_deleted_hook(_db, _user_id, _connector_type, connector_id):
        hook_calls.append(int(connector_id))
        return ConnectorDeleteDecision()

    def delete_the_row_when_the_lock_query_starts(*entities, **kwargs):
        queried_entities.append(entities)
        if entities == (CustomApi,) and not deleted_already.is_set():
            deleted_already.set()
            with session_factory() as other:
                other.execute(
                    sa.delete(UserCustomApi).where(
                        UserCustomApi.custom_api_id == api_id
                    )
                )
                other.execute(sa.delete(CustomApi).where(CustomApi.id == api_id))
                other.commit()
        return real_query(*entities, **kwargs)

    def record_commit():
        commits.append("commit")
        return real_commit()

    db.query = delete_the_row_when_the_lock_query_starts
    db.commit = record_commit
    set_connector_team_hooks(deleted=spy_deleted_hook)
    try:
        with pytest.raises(HTTPException) as exc:
            custom_api_api.delete_custom_api(
                api_id,
                current_user=current_user,
                db=db,
            )
        assert exc.value.status_code == 404
        assert deleted_already.is_set(), "the concurrent delete never ran"
        assert queried_entities[:2] == [(UserCustomApi,), (CustomApi,)], (
            "the concurrent delete must land after the access gate's own "
            "read and before the lock; otherwise the 404 under test could "
            f"be the access gate's rather than the lock's -- saw "
            f"{queried_entities!r}"
        )
        assert hook_calls == [], (
            "the lock's 404 must precede delete_team_connector, so a "
            "vanished row is refused before any hook-side mutation is "
            "attempted"
        )
        assert commits == [], "the refused delete must commit nothing"
    finally:
        set_connector_team_hooks()
        db.close()


def test_an_activation_only_edit_does_not_wait_on_a_concurrent_definition_edit(
    session_factory, seeded
) -> None:
    """A payload that sets only ``is_active`` writes this caller's own
    ``UserCustomApi`` link row and never the shared ``CustomApi``
    definition row, so it must not queue behind a concurrent editor that
    holds the definition row.

    Same barrier as
    ``test_a_second_editor_blocks_until_the_first_editors_transaction_finishes``
    above, with the two roles kept apart: the holder edits a definition
    field and stops inside the patched validation call with its
    transaction still open, and the activation-only request then has to
    finish while that transaction is still open. When this route took the
    definition row's lock for every payload, this request waited for the
    holder instead.
    """
    import xagent.web.api.custom_api as custom_api_api
    from xagent.web.api.custom_api import CustomApiUpdate

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    lock_acquired = threading.Event()
    release_lock = threading.Event()
    second_finished = threading.Event()
    first_call_claimed = threading.Event()
    first_call_lock = threading.Lock()

    real_validate = custom_api_api.validate_runtime_config_declaration

    def paced_validate(**kwargs):
        with first_call_lock:
            is_first_call = not first_call_claimed.is_set()
            first_call_claimed.set()
        if is_first_call:
            lock_acquired.set()
            assert release_lock.wait(timeout=10), "the holder was never released"
        return real_validate(**kwargs)

    custom_api_api.validate_runtime_config_declaration = paced_validate
    session_a = session_factory()
    session_b = session_factory()
    try:

        def run_definition_edit():
            return custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(description="held-by-the-definition-editor"),
                current_user=current_user,
                db=session_a,
            )

        def run_activation_only():
            result = custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(is_active=False),
                current_user=current_user,
                db=session_b,
            )
            second_finished.set()
            return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            holder = executor.submit(run_definition_edit)
            assert lock_acquired.wait(timeout=5), (
                "the definition editor never reached the lock"
            )

            activator = executor.submit(run_activation_only)
            # Released in ``finally`` before anything is asserted: a
            # failure here means the activation-only call is still parked
            # on the holder's lock, and leaving the holder paused would
            # hang this executor's shutdown instead of failing the test.
            try:
                finished_while_held = second_finished.wait(timeout=5.0)
            finally:
                release_lock.set()
            holder.result(timeout=10)
            activator.result(timeout=10)

        assert finished_while_held, (
            "the activation-only edit did not finish while the definition "
            "editor still held the definition row -- it is waiting on a "
            "lock for a row it does not write"
        )
    finally:
        custom_api_api.validate_runtime_config_declaration = real_validate
        session_a.close()
        session_b.close()

    with session_factory() as fresh:
        assert (
            fresh.query(UserCustomApi)
            .filter(
                UserCustomApi.custom_api_id == api_id,
                UserCustomApi.user_id == owner_id,
            )
            .one()
            .is_active
            is False
        )
        assert (
            fresh.query(CustomApi).filter(CustomApi.id == api_id).one().description
            == "held-by-the-definition-editor"
        )


def test_an_activation_only_edit_whose_row_vanishes_before_its_read_is_a_404(
    session_factory, seeded
) -> None:
    """The activation-only path skips the lock but keeps the same
    vanished-definition handling: its own fresh read of ``CustomApi`` must
    return ``None`` and raise the route's 404, rather than leaving
    ``db.refresh`` to fail on a row that is no longer there.

    Same wrapper as
    ``test_a_row_that_vanishes_after_the_gate_but_before_the_lock_is_a_404_not_a_500``
    above: the concurrent delete fires from this session's own ``query``,
    which lands it strictly between the access read (which asks for
    ``UserCustomApi``) and this route's only ``CustomApi`` query.
    """
    import xagent.web.api.custom_api as custom_api_api
    from xagent.web.api.custom_api import CustomApiUpdate

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    db = session_factory()
    real_query = db.query
    deleted_already = threading.Event()
    queried_entities: list[tuple] = []

    def delete_the_row_when_the_definition_query_starts(*entities, **kwargs):
        queried_entities.append(entities)
        if entities == (CustomApi,) and not deleted_already.is_set():
            deleted_already.set()
            with session_factory() as other:
                other.execute(
                    sa.delete(UserCustomApi).where(
                        UserCustomApi.custom_api_id == api_id
                    )
                )
                other.execute(sa.delete(CustomApi).where(CustomApi.id == api_id))
                other.commit()
        return real_query(*entities, **kwargs)

    db.query = delete_the_row_when_the_definition_query_starts
    try:
        with pytest.raises(HTTPException) as exc:
            custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(is_active=False),
                current_user=current_user,
                db=db,
            )
        assert exc.value.status_code == 404
        assert deleted_already.is_set(), "the concurrent delete never ran"
        assert queried_entities[:2] == [(UserCustomApi,), (CustomApi,)], (
            "the concurrent delete must land after the access gate's own "
            "read and before this route's definition read; otherwise the "
            "404 under test could be the access gate's -- saw "
            f"{queried_entities!r}"
        )
    finally:
        db.close()


def _revoke_link_after_lock(
    db, session_factory, owner_id, api_id, revocation, clear_column
):
    """Revoke the caller's own link row -- via a second, committing
    connection -- on the *second* time this session asks for
    ``(UserCustomApi,)``: the gate's ask is the first, the re-read added
    after the lock is the second, landing inside the window the lock holds.
    """
    real_query = db.query
    revoked_already = threading.Event()
    queried_entities: list[tuple] = []
    occurrences = 0

    def revoke_when_the_recheck_query_starts(*entities, **kwargs):
        nonlocal occurrences
        queried_entities.append(entities)
        if entities == (UserCustomApi,):
            occurrences += 1
            if occurrences == 2 and not revoked_already.is_set():
                revoked_already.set()
                with session_factory() as other:
                    if revocation == "link-deleted":
                        other.execute(
                            sa.delete(UserCustomApi).where(
                                UserCustomApi.user_id == owner_id,
                                UserCustomApi.custom_api_id == api_id,
                            )
                        )
                    else:
                        other.execute(
                            sa.update(UserCustomApi)
                            .where(
                                UserCustomApi.user_id == owner_id,
                                UserCustomApi.custom_api_id == api_id,
                            )
                            .values(**{clear_column: False})
                        )
                    other.commit()
        return real_query(*entities, **kwargs)

    db.query = revoke_when_the_recheck_query_starts
    return revoked_already, queried_entities


@pytest.mark.parametrize(
    ("revocation", "expected_status"),
    [("link-deleted", 404), ("can-edit-cleared", 403)],
)
def test_a_put_whose_association_is_revoked_after_the_lock_is_refused_with_no_shared_write(
    session_factory, seeded, revocation, expected_status
) -> None:
    """A revoked caller (link deleted, or ``can_edit`` cleared) committed
    by a second connection while this route holds the definition row
    locked must get the gate's own 404/403 from the re-read added after
    the lock, with the shared definition row left untouched either way.

    The payload carries a real rename (not just a description edit): the
    unrevoked path would call ``rename_team_connector`` for it, so its
    absence here is only meaningful because the hook had something to fire
    on. Without that, a passing ``hook_calls == []`` would prove nothing --
    the hook only fires when the name actually changes.
    """
    import xagent.web.api.custom_api as custom_api_api
    from xagent.web.api.custom_api import CustomApiUpdate

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    db = session_factory()
    real_commit = db.commit
    commits: list[str] = []
    hook_calls: list[tuple] = []

    def record_commit():
        commits.append("commit")
        return real_commit()

    def spy_renamed_hook(_db, _user_id, _connector_type, _connector_id, old, new):
        hook_calls.append((old, new))

    db.commit = record_commit
    revoked_already, queried_entities = _revoke_link_after_lock(
        db, session_factory, owner_id, api_id, revocation, "can_edit"
    )
    set_connector_team_hooks(renamed=spy_renamed_hook)
    try:
        with pytest.raises(HTTPException) as exc:
            custom_api_api.update_custom_api(
                api_id,
                CustomApiUpdate(
                    description="edited-after-revocation",
                    name="renamed-after-revocation",
                ),
                current_user=current_user,
                db=db,
            )
        assert exc.value.status_code == expected_status
        assert revoked_already.is_set(), "the concurrent revocation never ran"
        assert queried_entities[:3] == [
            (UserCustomApi,),
            (CustomApi,),
            (UserCustomApi,),
        ], (
            "the revocation must land after the gate's own read and after "
            f"the lock, caught by the re-read -- saw {queried_entities!r}"
        )
        assert hook_calls == [], "the re-read must precede rename_team_connector"
        assert commits == [], "the refused edit must commit nothing"
    finally:
        set_connector_team_hooks()
        db.close()

    with session_factory() as fresh:
        row = fresh.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert row.description is None, (
            "the shared definition row must be untouched by a refused edit"
        )
        assert row.name == "edit-lock-target", (
            "the shared definition row's name must be untouched by a refused edit"
        )
        if revocation == "can-edit-cleared":
            link = (
                fresh.query(UserCustomApi)
                .filter(
                    UserCustomApi.custom_api_id == api_id,
                    UserCustomApi.user_id == owner_id,
                )
                .one()
            )
            assert link.can_edit is False
            assert link.is_active is True


@pytest.mark.parametrize(
    ("revocation", "expected_status"),
    [("link-deleted", 404), ("can-delete-cleared", 403)],
)
def test_a_delete_whose_association_is_revoked_after_the_lock_is_refused_before_the_hook(
    session_factory, seeded, revocation, expected_status
) -> None:
    """Same window as the PUT test above, on the delete route: the re-read
    added after the lock, before ``delete_team_connector`` is ever called,
    must catch a revoked link with the route's existing 404/403, touching
    neither the hook nor the shared definition row.
    """
    import xagent.web.api.custom_api as custom_api_api

    owner_id, api_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    db = session_factory()
    real_commit = db.commit
    commits: list[str] = []
    hook_calls: list[int] = []

    def record_commit():
        commits.append("commit")
        return real_commit()

    def spy_deleted_hook(_db, _user_id, _connector_type, connector_id):
        hook_calls.append(int(connector_id))
        return ConnectorDeleteDecision()

    db.commit = record_commit
    revoked_already, queried_entities = _revoke_link_after_lock(
        db, session_factory, owner_id, api_id, revocation, "can_delete"
    )
    set_connector_team_hooks(deleted=spy_deleted_hook)
    try:
        with pytest.raises(HTTPException) as exc:
            custom_api_api.delete_custom_api(
                api_id,
                current_user=current_user,
                db=db,
            )
        assert exc.value.status_code == expected_status
        assert revoked_already.is_set(), "the concurrent revocation never ran"
        assert queried_entities[:3] == [
            (UserCustomApi,),
            (CustomApi,),
            (UserCustomApi,),
        ], (
            "the revocation must land after the gate's own read and after "
            f"the lock, caught by the re-read -- saw {queried_entities!r}"
        )
        assert hook_calls == [], "the re-read must precede delete_team_connector"
        assert commits == [], "the refused delete must commit nothing"
    finally:
        set_connector_team_hooks()
        db.close()

    with session_factory() as fresh:
        assert fresh.query(CustomApi).filter(CustomApi.id == api_id).one(), (
            "the shared definition row must survive a refused delete"
        )
