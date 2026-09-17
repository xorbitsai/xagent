"""One acknowledged registration point for tests that need a scope resolver.

``set_execution_scope_resolver`` requires
``acknowledges_snapshot_candidate_contract=True`` so that an *embedding
application* built against the older precedence fails at import instead of
discovering on its first delegated turn that a persisted snapshot is now a
corroborating candidate rather than an override.

A test that registers a resolver is not such an application: it is setting up a
scenario. Repeating the acknowledgement at every one of those call sites would
put the contract's most important signal where nobody reads it, and would make
the next test that forgets the keyword fail with a ``TypeError`` that says
nothing about what the test was doing. Consumer tests therefore register
through here, and the acknowledgement -- with the reason for it -- lives in one
place.

The contract's own unit tests (``tests/core/test_execution_scope.py``) call
``set_execution_scope_resolver`` directly: there the keyword is the subject
under test, not boilerplate.
"""

from __future__ import annotations

from typing import Optional

from xagent.core.execution_scope import (
    ExecutionScopeResolver,
    set_execution_scope_resolver,
)


def register_scope_resolver(resolver: Optional[ExecutionScopeResolver]) -> None:
    """Install ``resolver`` as the authoritative scope resolver, or clear it.

    ``None`` clears the registration, which needs no acknowledgement: without a
    resolver the snapshot resolves the scope, the behavior that predates the
    contract.
    """
    if resolver is None:
        set_execution_scope_resolver(None)
        return
    set_execution_scope_resolver(
        resolver, acknowledges_snapshot_candidate_contract=True
    )
