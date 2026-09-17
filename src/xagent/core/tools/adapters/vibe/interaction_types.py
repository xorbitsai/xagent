"""The interaction types the ask_user_question render surface implements,
plus the off-contract aliases it accepts, the subset of those types that is
useless without options, and the free-text field the engine appends when a
suspending message would otherwise carry nothing answerable.

Kept in a module of its own rather than beside ``InteractionArg`` in
``ask_user_tool``, which is where the model that carries the field lives:
importing ``ask_user_tool`` pulls in the whole tool-registration chain and
with it 61 ``xagent.web`` modules, and one of this list's three consumers
(``core/agent/pattern/react/react.py``) imports nothing from ``xagent.web``
today. A list of seven strings must not be what changes that. This module
imports only the standard library, so any consumer can take it.

Ordered, not a set: two of the three consumers render it into text a model
reads -- the ``ask_user_question`` JSON-Schema enum and
``InteractionArg.type``'s own description -- and an unstable order would
change the prompt from run to run.

The three consumers, all of which used to carry their own copy of these
seven names:

* ``InteractionArg.type``'s description (``ask_user_tool.py``)
* the ``ask_user_question`` JSON-Schema enum (``react.py``)
* the write side's admissibility set (``_V1_INTERACTION_TYPES``,
  ``web/services/task_interaction_service.py``)
"""

from collections.abc import Mapping
from types import MappingProxyType

INTERACTION_TYPES: tuple[str, ...] = (
    "select_one",
    "select_multiple",
    "text_input",
    "file_upload",
    "confirm",
    "number_input",
    "action_cards",
)

# Hand-copy of the map in ``normalizeInteractions``
# (``frontend/src/contexts/app-context-chat.tsx``). Applied engine-side too,
# so an alias cannot read as a rendered field there and an unsupported type
# here.
INTERACTION_TYPE_ALIASES: dict[str, str] = {
    "input": "text_input",
    "text": "text_input",
    "textarea": "text_input",
    "string": "text_input",
    "file": "file_upload",
    "upload": "file_upload",
    "number": "number_input",
    "integer": "number_input",
    "boolean": "confirm",
}

# The pick-from-a-list subset, so one of them with no options is a control
# nobody can answer. Shared so the write-side admissibility rule and the
# engine's answerability check cannot drift apart.
TYPES_REQUIRING_OPTIONS: frozenset[str] = frozenset(
    {"select_one", "select_multiple", "action_cards"}
)

# The free-text field the engine appends when a suspending message would
# otherwise carry nothing answerable. Read-only: publishers copy it, so an
# in-place edit cannot leak into every later append in the process.
DEFAULT_WAITING_INTERACTION: Mapping[str, object] = MappingProxyType(
    {
        "type": "text_input",
        "field": "response",
        "label": "Your response",
        "placeholder": "Type your answer",
        "multiline": True,
    }
)
