"""Structured output for a finished run: decide whether the answer needs a
structured result and extract it from the final answer.

:func:`structured_output_eligibility` reads the run result and the caller's
format and returns :data:`SKIP`, :data:`EXTRACT` or an outcome that is already
settled. Judging always uses the compiled form of the caller's own schema; the
nullable copy written by :func:`model_facing_schema` is only shown to the
model. This module belongs to the core layer and imports nothing from the web
layer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from .result import NO_OUTPUT_PLACEHOLDER
from .structured_schema import (
    Incomplete,
    Ready,
    TooComplex,
    canonical_json_bytes,
    compile_schema,
    has_forbidden_char,
    judge_empty_answer,
)

# The key of a run result that carries the structured result.
STRUCTURED_OUTPUT_RESULT_KEY = "structured_output"

# The output limit of one extraction call when the model row sets none.
EXTRACTION_MAX_OUTPUT_TOKENS = 8192
# The most seconds the whole extraction of one run may take.
EXTRACTION_TIMEOUT_SECONDS = 120
MAX_EXTRACTION_ATTEMPTS = 3
# The most bytes of canonical JSON the arguments of one reply may take.
MAX_ARGUMENTS_BYTES = 262_144


@dataclass(frozen=True)
class StructuredOutputFormat:
    """The result shape a caller asked for: the ``name`` and ``description`` of
    the result and its JSON ``schema``.

    The schema is a shared reference and is only read. ``INVALID`` stands for a
    stored format that failed its checks and ``READ_FAILED`` for a stored
    format that could not be read; both are compared with ``is``.
    """

    name: str
    description: str | None
    schema: dict[str, Any]

    INVALID: ClassVar[StructuredOutputFormat]
    READ_FAILED: ClassVar[StructuredOutputFormat]


StructuredOutputFormat.INVALID = StructuredOutputFormat("<invalid>", None, {})
StructuredOutputFormat.READ_FAILED = StructuredOutputFormat("<read_failed>", None, {})


class _Sentinel:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


# The run needs no structured result: no key is written.
SKIP: Final = _Sentinel("SKIP")
# The run needs the model to write its structured result.
EXTRACT: Final = _Sentinel("EXTRACT")

_NOT_FINISHED_STATUSES = frozenset({"waiting_for_user", "interrupted", "paused"})


def answer_text(result: dict[str, Any]) -> str:
    """The text of the final answer of ``result``: a string output as it is, a
    dict or list as compact JSON that keeps non-ASCII characters, none as the
    empty string and anything else turned into text."""
    output = result.get("output")
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    if isinstance(output, (dict, list)):
        return json.dumps(
            output, default=str, ensure_ascii=False, separators=(",", ":")
        )
    return str(output)


def _settled(
    status: str,
    code: str | None = None,
    *,
    output: Any = None,
    missing: Any = (),
    unexpressible: Any = (),
) -> dict[str, Any]:
    """An outcome in the shape every read path accepts: a ready result holds
    its output and no reason; the others hold a reason code, with the field
    lists only when they are not empty."""
    if status == "ready":
        return {"status": "ready", "output": output, "reason": None}
    reason: dict[str, Any] = {"code": code}
    if missing:
        reason["missing_fields"] = list(missing)
    if unexpressible:
        reason["unexpressible_fields"] = list(unexpressible)
    return {"status": status, "output": None, "reason": reason}


def _settled_from_verdict(
    verdict: Ready | Incomplete | TooComplex,
) -> dict[str, Any]:
    if isinstance(verdict, Ready):
        return _settled("ready", output=verdict.output)
    if isinstance(verdict, TooComplex):
        return _settled("failed", "result_too_large")
    code = (
        "information_missing"
        if verdict.missing_fields or verdict.missing_count > 0
        else "value_not_expressible"
    )
    return _settled(
        "incomplete",
        code,
        missing=verdict.missing_fields,
        unexpressible=verdict.unexpressible_fields,
    )


def structured_output_eligibility(
    result: dict[str, Any], fmt: StructuredOutputFormat | None
) -> _Sentinel | dict[str, Any]:
    """Decide what ``result`` needs: :data:`SKIP`, :data:`EXTRACT` or an
    outcome that is already settled without a model.

    The rows are taken from the top and the first that matches decides: no
    format, a run that did not finish normally, an invalid or unreadable
    format, a partial or blocked run, and an empty answer.
    """
    if fmt is None:
        return SKIP
    if (
        not result.get("success")
        or result.get("status") in _NOT_FINISHED_STATUSES
        or result.get("injection_outcome_unknown")
    ):
        return SKIP
    if fmt is StructuredOutputFormat.INVALID:
        return _settled("failed", "invalid_stored_format")
    if fmt is StructuredOutputFormat.READ_FAILED:
        return _settled("failed", "internal_error")
    outcome = result.get("completion_outcome")
    if outcome == "partial":
        return _settled("incomplete", "agent_partial")
    if outcome == "blocked":
        return _settled("incomplete", "agent_blocked")
    text = answer_text(result).strip()
    if text == "" or text == NO_OUTPUT_PLACEHOLDER:
        return _settled_from_verdict(judge_empty_answer(compile_schema(fmt.schema)))
    return EXTRACT


@dataclass(frozen=True)
class _ParsedArguments:
    """Arguments that are JSON the judgement can read."""

    value: Any


@dataclass(frozen=True)
class _BadArguments:
    """Arguments that cannot be judged: ``reason`` is one of ``invalid_json``,
    ``non_finite_number``, ``invalid_character`` and ``too_long``;
    ``arguments_bytes`` is the size of the canonical JSON for ``too_long``."""

    reason: str
    arguments_bytes: int = 0


_JSON_SCALARS = (int, bool, type(None))


def _parse_arguments(arguments: Any) -> _ParsedArguments | _BadArguments:
    """Read the arguments of a tool call, in this order: text is parsed, the
    tree is walked, its canonical JSON is written, its size is measured.

    A text and the object it holds give the same outcome. The walk uses an
    explicit stack; a value whose exact type is not one of the seven JSON types
    is reported without being entered, and only a ``str`` key is checked for
    forbidden characters.
    """
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
        except (ValueError, RecursionError):
            return _BadArguments("invalid_json")
    else:
        value = arguments

    forbidden = non_finite = foreign = False
    stack = [value]
    while stack and not forbidden:
        node = stack.pop()
        kind = type(node)
        if kind is str:
            forbidden = has_forbidden_char(node)
        elif kind is float:
            non_finite = non_finite or not math.isfinite(node)
        elif kind in _JSON_SCALARS:
            continue
        elif kind is dict:
            for key, child in node.items():
                if type(key) is str and has_forbidden_char(key):
                    forbidden = True
                stack.append(child)
        elif kind is list:
            stack.extend(node)
        else:
            foreign = True
    if forbidden:
        return _BadArguments("invalid_character")
    if non_finite:
        return _BadArguments("non_finite_number")
    if foreign:
        return _BadArguments("invalid_json")

    try:
        size = canonical_json_bytes(value)
    except (ValueError, RecursionError, TypeError):
        return _BadArguments("invalid_json")
    if size > MAX_ARGUMENTS_BYTES:
        return _BadArguments("too_long", size)
    return _ParsedArguments(value)
