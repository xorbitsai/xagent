"""The pre-change equivalence table for the three-state active interaction read.

Each row pairs one ``ActiveInteractionRead`` state with the ``int | None``
value ``active_interaction_id_sync`` returned for that same situation
before it became three-state. Every production call site's translation of
the three states back into that value is checked against these rows: the
parameterized order tests in tests/web/api/test_a2a_api.py,
tests/web/api/v1/test_task_reply.py and
tests/web/api/test_websocket_owner_actor.py (the three legacy-resume close
sites) all read the table from here.

Written in its own module rather than in test_task_interaction_close.py so
that those three files do not have to import a name out of another suite's
test module -- the same split task_interaction_schema_shared.py and
task_status_storage_shared.py use for the row builders and constants their
suites share.

The table's own exhaustiveness (a state or an unavailable-reason word
added later must gain a row here) is pinned by
test_task_interaction_close.py::test_the_pre_change_equivalent_table_covers_every_state,
which lives with the rest of that reader's per-situation tests.
"""

from __future__ import annotations

from xagent.web.services.task_interaction_close import (
    ActiveInteractionAbsent,
    ActiveInteractionFound,
    ActiveInteractionRead,
    ActiveInteractionUnavailable,
)

# Annotated with the union itself rather than ``object``: a row carrying
# something that is not an ``ActiveInteractionRead`` is a type error at the
# table, not a mystery failure in whichever call site's parameterized test
# reads it. ``tests/`` is outside this repo's mypy scope, so the annotation
# only pays off for a reader and for an editor's checker -- which is still
# more than ``object`` offers either.
PRE_CHANGE_EQUIVALENT: list[tuple[ActiveInteractionRead, int | None]] = [
    (ActiveInteractionFound(4321), 4321),
    (ActiveInteractionAbsent(), None),
    (ActiveInteractionUnavailable("session_unavailable"), None),
    (ActiveInteractionUnavailable("lookup_failed"), None),
]
