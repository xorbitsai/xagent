"""The ephemeral per-turn connector secrets store's TTL-based lifetime.

Some settlement paths (task_orchestrator.py's and websocket.py's deferred-
to-TTL-recovery branches: lease lost, DB pool exhaustion, unhealthy
heartbeat at shutdown) can never safely pop a turn's ephemeral secrets
themselves - at the point they bail out they genuinely do not know whether
the task will land on a terminal status or resume again under the same
turn_id, and task_lease_recovery.py's later batch sweep has no way to map a
recovered task_id back to the turn_id its secrets were stored under. Without
a bound, a turn that goes through one of those paths leaks its secrets for
the life of the process (see connector_runtime.py's `_EPHEMERAL_RUNTIME_
VALUES`, an unbounded module-global dict). Expiry is enforced on every
read/pop, not just by the opportunistic reaper another turn's store call may
trigger - a stale entry that nothing ever looks up again still becomes
unreadable the moment it ages past the TTL, not merely "eventually reclaimed
whenever something else happens to store." A turn that instead resumes
again under its same turn_id (WAITING_FOR_USER/PAUSED) renews the TTL, so a
still-active pause's secrets don't expire on the clock of the FIRST pause
that stored them.
"""

from __future__ import annotations

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRef
from xagent.web.services import connector_runtime as connector_runtime_module
from xagent.web.services.connector_runtime import (
    get_ephemeral_runtime_manifest,
    get_ephemeral_runtime_values,
    pop_ephemeral_runtime_values,
    renew_ephemeral_runtime_values,
    store_ephemeral_runtime_values,
)

_VALUES = {ConnectorRef("mcp", 1): {"secrets": {"authorization": "Bearer t"}}}


def _advance_clock(monkeypatch, seconds: float) -> None:
    """Move every future ``time.monotonic()`` read forward by ``seconds``."""

    real_monotonic = connector_runtime_module.time.monotonic
    offset = {"value": seconds}
    monkeypatch.setattr(
        connector_runtime_module.time,
        "monotonic",
        lambda: real_monotonic() + offset["value"],
    )


def test_ephemeral_values_survive_well_within_the_ttl(monkeypatch) -> None:
    turn_id = "ephemeral-ttl-fresh"
    store_ephemeral_runtime_values(turn_id, _VALUES)
    _advance_clock(
        monkeypatch, connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS / 2
    )

    store_ephemeral_runtime_values("ephemeral-ttl-fresh-trigger", _VALUES)

    assert get_ephemeral_runtime_values(turn_id) is not None
    assert pop_ephemeral_runtime_values(turn_id) is not None
    assert pop_ephemeral_runtime_values("ephemeral-ttl-fresh-trigger") is not None


def test_a_stale_entry_is_reclaimed_by_the_next_store_call(monkeypatch) -> None:
    """A stale entry that nothing ever reads again gets swept off the module
    dicts as soon as an unrelated turn's store runs its opportunistic prune -
    the entry does not merely become unreadable, it is actually removed."""
    turn_id = "ephemeral-ttl-stale"
    store_ephemeral_runtime_values(turn_id, _VALUES)
    assert get_ephemeral_runtime_values(turn_id) is not None

    _advance_clock(
        monkeypatch, connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS + 1
    )

    other_turn_id = "ephemeral-ttl-unrelated-store"
    store_ephemeral_runtime_values(other_turn_id, _VALUES)

    assert turn_id not in connector_runtime_module._EPHEMERAL_RUNTIME_VALUES
    assert pop_ephemeral_runtime_values(other_turn_id) is not None


def test_expiry_also_drops_the_manifest(monkeypatch) -> None:
    turn_id = "ephemeral-ttl-manifest"
    store_ephemeral_runtime_values(turn_id, _VALUES)
    assert get_ephemeral_runtime_manifest(turn_id) is not None

    _advance_clock(
        monkeypatch, connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS + 1
    )
    store_ephemeral_runtime_values("ephemeral-ttl-manifest-trigger", _VALUES)

    assert get_ephemeral_runtime_manifest(turn_id) is None
    pop_ephemeral_runtime_values("ephemeral-ttl-manifest-trigger")


def test_reads_treat_an_expired_entry_as_absent_with_no_intervening_store(
    monkeypatch,
) -> None:
    """A quiet process - no other turn ever stores anything after this one
    expires - must still stop returning it. Expiry has to be an observable
    property of get/pop themselves, not only a side effect the NEXT store
    call happens to trigger; otherwise a deployment with infrequent traffic
    could read a secret well past its advertised TTL."""
    turn_id = "ephemeral-ttl-quiet-process"
    store_ephemeral_runtime_values(turn_id, _VALUES)
    assert get_ephemeral_runtime_values(turn_id) is not None

    _advance_clock(
        monkeypatch, connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS + 1
    )

    assert get_ephemeral_runtime_values(turn_id) is None
    assert get_ephemeral_runtime_manifest(turn_id) is None
    assert pop_ephemeral_runtime_values(turn_id) is None


def test_renew_extends_the_ttl_for_a_still_live_entry(monkeypatch) -> None:
    """A second pause under the same turn_id carries its own fresh
    interaction lifetime - its secrets must not expire on the ORIGINAL
    pause's clock."""
    turn_id = "ephemeral-ttl-renewed"
    store_ephemeral_runtime_values(turn_id, _VALUES)

    _advance_clock(
        monkeypatch, connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS - 1
    )
    renew_ephemeral_runtime_values(turn_id)

    # Past the ORIGINAL store's TTL window, but well within the renewed one.
    _advance_clock(
        monkeypatch, connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS - 1
    )
    assert get_ephemeral_runtime_values(turn_id) is not None
    assert pop_ephemeral_runtime_values(turn_id) is not None


def test_renew_does_not_resurrect_an_already_expired_entry(monkeypatch) -> None:
    """Renewal is not a way to un-expire something that already aged out -
    once gone, a late renewal call must stay a no-op, not silently bring the
    secrets back for a turn nothing else remembers is still active."""
    turn_id = "ephemeral-ttl-renew-too-late"
    store_ephemeral_runtime_values(turn_id, _VALUES)

    _advance_clock(
        monkeypatch, connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS + 1
    )
    renew_ephemeral_runtime_values(turn_id)

    assert get_ephemeral_runtime_values(turn_id) is None
    assert turn_id not in connector_runtime_module._EPHEMERAL_RUNTIME_VALUES


def test_renew_is_a_noop_for_a_turn_with_nothing_stored() -> None:
    """A turn that never stored ephemeral secrets (or was already popped) has
    nothing to keep alive - renewing it must not fabricate an entry."""
    turn_id = "ephemeral-ttl-renew-unknown"

    renew_ephemeral_runtime_values(turn_id)

    assert turn_id not in connector_runtime_module._EPHEMERAL_RUNTIME_STORED_AT
    assert get_ephemeral_runtime_values(turn_id) is None
