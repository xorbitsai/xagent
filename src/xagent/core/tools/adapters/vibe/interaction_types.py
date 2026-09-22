"""The interaction types the ask_user_question render surface implements,
plus the off-contract aliases it accepts, the subset of those types that is
useless without options, the free-text field the engine appends when a
suspending message would otherwise carry nothing answerable, and the
per-field helpers that turn one of those useless controls into a free-text
field of its own (xagent#2529).

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

import logging
from collections.abc import Mapping
from types import MappingProxyType

logger = logging.getLogger(__name__)

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


def lacks_required_options(interaction: Mapping[str, object]) -> bool:
    """Whether this is a pick-from-a-list control with nothing to pick.

    The per-field form of the rule ``_is_answerable`` (``react.py``) applies
    to a whole list. True only for a ``TYPES_REQUIRING_OPTIONS`` type whose
    ``options`` is not a non-empty list: a missing key, ``[]`` (including
    one ``_normalize_ask_user_interactions`` filtered down to empty) and the
    non-list shapes that function leaves verbatim all count, because the
    render surface can iterate none of them. A ``type`` that is not a string
    is not a picker: the model is free to send a list there, and a frozenset
    membership test on it would raise ``TypeError``.
    """

    interaction_type = interaction.get("type")
    if not isinstance(interaction_type, str):
        return False
    if interaction_type not in TYPES_REQUIRING_OPTIONS:
        return False
    options = interaction.get("options")
    return not (isinstance(options, list) and bool(options))


def degrade_to_text_input(interaction: Mapping[str, object]) -> dict[str, object]:
    """The free-text field that stands in for a picker with nothing to pick.

    Keeps the three keys that carry the question -- ``field`` (what the
    answer is filed under), a string ``label`` (the question text, of which
    the run has no other copy) and a non-blank string ``placeholder`` -- and
    drops everything else: ``options`` is what was missing, ``default_value``
    named one of those missing options, and the rest are other types' knobs.
    A ``select_multiple`` asked for several values, so its stand-in is
    ``multiline``; the other two asked for one.
    The write side's validator (``validate_v1_write_payload``) refuses a
    ``text_input`` that carries a non-empty ``options``; an emptied list
    would pass it, but the key is dropped rather than emptied so the
    degraded field carries nothing the type does not use. Returns a new
    dict; the input is not touched.
    """

    degraded: dict[str, object] = {
        "type": "text_input",
        "field": interaction.get("field"),
    }
    label = interaction.get("label")
    if isinstance(label, str):
        degraded["label"] = label
    placeholder = interaction.get("placeholder")
    if isinstance(placeholder, str) and placeholder.strip():
        degraded["placeholder"] = placeholder
    if interaction.get("type") == "select_multiple":
        degraded["multiline"] = True
    return degraded


def degrade_options_less_pickers(
    interactions: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    """Replace every picker with nothing to pick by a free-text field.

    The per-field form of the rule ``ReActPattern._send_waiting_message``
    applies to the whole list. That list-level check appends a field only
    when *nothing* is answerable, so a form mixing an options-less
    ``select_one`` with ordinary text inputs passed it untouched and the
    picker -- usually the field the task depends on -- reached the user as
    a label over an empty dropdown (xagent#2529). The write side's
    validator refuses the same shape (``options_required``,
    ``validate_v1_write_payload``), but the engine's own publish path never
    runs it.

    Meant to run after field deduplication, so the names returned are the
    ones answers come back under, and before the list-level check, which
    then sees the degraded field as answerable. Returns the list to publish
    -- untouched entries are the same objects, not copies -- and the
    degraded field names, in order, for the tool result.
    """

    published: list[dict[str, object]] = []
    degraded_fields: list[str] = []
    for item in interactions:
        if lacks_required_options(item):
            item = degrade_to_text_input(item)
            degraded_fields.append(str(item.get("field") or ""))
        published.append(item)
    if degraded_fields:
        # Same payload discipline as the normalizer's warnings
        # (``_normalize_ask_user_interactions``): bounded, integer counts
        # only, never the model-controlled field name.
        logger.warning(
            "ask_user_question degraded %d options-less selection field(s) "
            "to text_input out of %d",
            len(degraded_fields),
            len(published),
            extra={"degraded": len(degraded_fields), "total": len(published)},
        )
    return published, degraded_fields


# The one passage every model-facing surface uses for the options rule: the
# ``ask_user_question`` schema description (``react.py``), the
# ``InteractionArg.options`` field description (``ask_user_tool.py``) and the
# tool-result note below. One copy, so the three cannot drift apart.
OPTIONS_REQUIRED_GUIDANCE = (
    "Every select_one, select_multiple or action_cards field must carry a "
    "non-empty options list; when the choices come from a system of record, "
    "call the tool that lists them first, or use text_input. A selection "
    "field sent without options is shown to the user as a free-text input."
)

# Handed back to the model beside ``degraded_fields``: the echoed list already
# shows ``text_input`` where it wrote ``select_one``, but a model does not diff
# its own call against the echo.
DEGRADED_FIELDS_NOTE = (
    "The fields listed under degraded_fields were sent as a selection type "
    "with no options and were shown to the user as free-text inputs instead. "
    + OPTIONS_REQUIRED_GUIDANCE
)
