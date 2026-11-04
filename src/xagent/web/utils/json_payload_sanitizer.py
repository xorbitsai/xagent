"""Make a payload survive a PostgreSQL ``jsonb`` column unchanged.

Two separate hazards, both introduced by the column type itself.

**Code points jsonb refuses to store.**

The trace payload columns are ``jsonb`` on PostgreSQL (#1248). ``jsonb``
decodes JSON escapes to native text on the way in, so it rejects two shapes
the previous ``json`` columns stored happily and then failed to read back
(#1149):

- the NUL escape ``\\u0000`` -- PostgreSQL text cannot carry NUL; and
- either half of an unpaired UTF-16 surrogate, which is not a Unicode
  scalar value and has no UTF-8 form.

Both reach this process through ordinary LLM or tool output: ``json.loads``
accepts the escapes without complaint and hands back Python strings carrying
the raw code points, and ``json.dumps`` re-emits them. Sanitizing at the
write boundary keeps the rule in one place instead of at every producer.

Offending code points become U+FFFD (REPLACEMENT CHARACTER), the designated
marker for undecodable input -- replacement rather than deletion, so the
mangling stays visible and adjacent text is not silently joined.

A surrogate code point in a Python ``str`` is always garbage here: a *valid*
pair never survives ``json.loads`` (it is combined into one astral
character), and even a high+low adjacency built by hand cannot be encoded
as UTF-8. So every surrogate is replaced, without pair-matching -- unlike
the read-side guard in ``web/api/monitor.py``, which inspects the *text*
form of stored JSON and must strip valid pairs before matching.

**Numbers jsonb stores but hands back as a different type.**

``jsonb`` parses numbers into ``numeric`` and re-renders them in plain
notation, while ``json`` keeps the literal text. A float that ``repr``
writes with an exponent -- ``1e+16`` -- therefore comes back from the
database as the *int* ``10000000000000000``. Value-wise that is the same
number, but it is not the same JSON, and the checkpoint blob path
(``web/services/trace_message_storage.py``) re-hashes payloads it reads
back and compares them against the hash stored at write time: a payload
carrying such a float would be rejected as corrupt on restore, costing
the task its checkpoint.

Normalizing here, at the same boundary and before the hash is computed,
makes the stored form and the read-back form identical. Every float at or
above 1e16 is integral (the mantissa runs out of fractional bits above
2**53), so the conversion loses nothing that ``jsonb`` would not have
taken anyway.

Rows written *before* the jsonb migration are not covered by this and may
still carry such a float; see that migration's docstring for why they are
not rewritten wholesale.
"""

from __future__ import annotations

import math
import re
from itertools import islice
from typing import Any

REPLACEMENT_CHARACTER = "�"

# NUL plus the whole surrogate range. Raw ranges are safe in a character
# class; the pattern never leaves this module in source-escape form.
_UNSTORABLE_CODE_POINTS = re.compile("[\x00\ud800-\udfff]")

# Where repr switches a float to exponent notation. Below it, json and
# jsonb agree on the text; at or above it, jsonb re-renders in plain
# notation and json.loads reads an int back.
_EXPONENT_NOTATION_THRESHOLD = 1e16


# Returned by _sanitize for a value that needs no edit, so a container can
# tell "nothing changed here" from "changed to something falsy" and skip
# copying itself. A module-private sentinel, never part of a payload.
_UNCHANGED = object()


def sanitize_json_payload(value: Any) -> Any:
    """Return ``value`` in the form the jsonb column will hand back.

    Unstorable code points become U+FFFD, and floats jsonb would return as
    ints are converted up front. Walks strings, dicts (keys included),
    lists, and tuples; every other type passes through untouched.

    A payload that needs no change is returned as the *same object*, and no
    copy of it is built along the way: this runs on every trace write and
    almost every payload is clean, so the clean path allocates nothing.
    Containers copy on first change only, backfilling the prefix they had
    already walked.

    Two distinct dict keys can collide after replacement (``"a\\x00"`` and
    ``"a\\ud800"`` both become ``"a\\ufffd"``); the later key wins, matching
    ``json.loads`` duplicate-key behaviour.

    Recursion is unbounded on purpose: a depth cap would have to leave
    whatever sits below it unsanitized, which is the payload shape this
    function exists to keep out of the column. A structure deep enough to
    exhaust the stack therefore raises ``RecursionError`` rather than
    storing something the database cannot read back -- but note that both
    callers treat a raising trace write as best-effort (``websocket.py``
    rolls back and logs; ``trace_handlers.py`` logs and continues), so the
    row is dropped with a log line, not surfaced to the caller. Payloads
    that deep are not a shape the write paths produce -- the staging path's
    own ``serialize_value`` recursion would hit the limit first -- so this
    is a stated boundary, not a guard anything relies on.
    """
    cleaned = _sanitize(value)
    return value if cleaned is _UNCHANGED else cleaned


def _sanitize(value: Any) -> Any:
    """Return the sanitized form of ``value``, or ``_UNCHANGED``."""
    if isinstance(value, str):
        # One pass, not a search followed by a sub: re.sub hands back the
        # original object when nothing matched, so identity answers the
        # "did anything change" question for free.
        cleaned = _UNSTORABLE_CODE_POINTS.sub(REPLACEMENT_CHARACTER, value)
        return _UNCHANGED if cleaned is value else cleaned
    # No bool guard needed: bool subclasses int, not float, so True/False
    # never reach this branch.
    if isinstance(value, float):
        if abs(value) >= _EXPONENT_NOTATION_THRESHOLD and value.is_integer():
            return int(value)
        # PostgreSQL numeric has no signed zero, so jsonb renders -0.0 as
        # 0.0 while json.dumps writes "-0.0". Same class of retype as the
        # exponent case above, at the opposite end of the range. copysign,
        # because -0.0 == 0.0 is True and cannot distinguish them.
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return 0.0
        return _UNCHANGED
    if isinstance(value, dict):
        cleaned_dict: dict[Any, Any] | None = None
        for index, (key, item) in enumerate(value.items()):
            new_key = _sanitize(key)
            new_item = _sanitize(item)
            if new_key is _UNCHANGED and new_item is _UNCHANGED:
                if cleaned_dict is not None:
                    cleaned_dict[key] = item
                continue
            if cleaned_dict is None:
                # First change: take the prefix already walked, which by
                # construction needed no edit, and copy from here on.
                cleaned_dict = dict(islice(value.items(), index))
            cleaned_dict[key if new_key is _UNCHANGED else new_key] = (
                item if new_item is _UNCHANGED else new_item
            )
        return _UNCHANGED if cleaned_dict is None else cleaned_dict
    if isinstance(value, (list, tuple)):
        cleaned_items: list[Any] | None = None
        for index, item in enumerate(value):
            new_item = _sanitize(item)
            if new_item is _UNCHANGED:
                if cleaned_items is not None:
                    cleaned_items.append(item)
                continue
            if cleaned_items is None:
                cleaned_items = list(value[:index])
            cleaned_items.append(new_item)
        if cleaned_items is None:
            return _UNCHANGED
        return tuple(cleaned_items) if isinstance(value, tuple) else cleaned_items
    return _UNCHANGED
