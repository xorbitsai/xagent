"""The ephemeral per-turn connector secrets store's TTL-based reaper.

Some settlement paths (task_orchestrator.py's and websocket.py's deferred-
to-TTL-recovery branches: lease lost, DB pool exhaustion, unhealthy
heartbeat at shutdown) can never safely pop a turn's ephemeral secrets
themselves - at the point they bail out they genuinely do not know whether
the task will land on a terminal status or resume again under the same
turn_id, and task_lease_recovery.py's later batch sweep has no way to map a
recovered task_id back to the turn_id its secrets were stored under. Without
a bound, a turn that goes through one of those paths leaks its secrets for
the life of the process (see connector_runtime.py's `_EPHEMERAL_RUNTIME_
VALUES`, an unbounded module-global dict). These tests pin the reaper that
turns that into a bounded leak instead.
"""

from __future__ import annotations

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRef
from xagent.web.services import connector_runtime as connector_runtime_module
from xagent.web.services.connector_runtime import (
    get_ephemeral_runtime_values,
    pop_ephemeral_runtime_values,
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
    """Models the leak scenario directly: nothing ever looks this turn_id up
    again (its task went through a deferred-settlement path and was
    abandoned), so only another, unrelated store's opportunistic prune can
    ever reclaim it - there is no dedicated background sweep."""
    turn_id = "ephemeral-ttl-stale"
    store_ephemeral_runtime_values(turn_id, _VALUES)
    assert get_ephemeral_runtime_values(turn_id) is not None

    _advance_clock(
        monkeypatch, connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS + 1
    )

    # An unrelated turn's store is what triggers the sweep - the stale
    # turn_id above is never itself read or stored to again.
    other_turn_id = "ephemeral-ttl-unrelated-store"
    store_ephemeral_runtime_values(other_turn_id, _VALUES)

    assert get_ephemeral_runtime_values(turn_id) is None
    assert pop_ephemeral_runtime_values(turn_id) is None
    assert pop_ephemeral_runtime_values(other_turn_id) is not None


def test_expiry_also_drops_the_manifest(monkeypatch) -> None:
    from xagent.web.services.connector_runtime import get_ephemeral_runtime_manifest

    turn_id = "ephemeral-ttl-manifest"
    store_ephemeral_runtime_values(turn_id, _VALUES)
    assert get_ephemeral_runtime_manifest(turn_id) is not None

    _advance_clock(
        monkeypatch, connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS + 1
    )
    store_ephemeral_runtime_values("ephemeral-ttl-manifest-trigger", _VALUES)

    assert get_ephemeral_runtime_manifest(turn_id) is None
    pop_ephemeral_runtime_values("ephemeral-ttl-manifest-trigger")
