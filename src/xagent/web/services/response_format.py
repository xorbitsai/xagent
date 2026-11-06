"""The ``response_format`` of a task: checking a request value, storing it,
checking the stored value again before a run, and the ``(type, name)`` pair a
response echoes.

``tasks.agent_config["response_format"]`` holds the format of a task. This
module is the only writer of that key. A request value goes through
:func:`normalize_response_format`, which returns one of three results: ``None``
(change nothing), :data:`CLEAR` (remove the stored format) or a
:class:`ResponseFormat`. :func:`store_response_format` applies that result to a
copy of a task's ``agent_config``.

A rejection is a :class:`ResponseFormatError` whose message is
``<rule> at <where>: <text>``. The rule is one of the 22 ids in
:data:`RULE_TEXTS`, ``<where>`` is a rendered path, and ``<text>`` is the fixed
sentence of the rule. Nothing of the rejected value is echoed except through
the path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ...core.agent.structured_output import StructuredOutputFormat
from ...core.agent.structured_schema import (
    MAX_TEXT_CHARS,
    NAME_PATTERN,
    PathSegment,
    check_schema_structure,
    has_forbidden_char,
    json_kind,
    render_error_path,
)

logger = logging.getLogger(__name__)

# The key of ``tasks.agent_config`` that holds the format of a task.
RESPONSE_FORMAT_CONFIG_KEY = "response_format"

RULE_TEXTS: dict[str, str] = {
    "N1": "response_format must be an object or null.",
    "N2": 'type must be "text" or "json_schema".',
    "N3": (
        "A response_format may only have the key type, "
        'plus json_schema when type is "json_schema".'
    ),
    "N4": (
        "json_schema must be an object with a schema key and no keys other "
        "than name, schema, strict and description."
    ),
    "N5": ('name must be 1 to 64 characters, each one of A-Z, a-z, 0-9, "_" or "-".'),
    "N6": (
        "description must be a string of at most 1024 characters "
        "with no NUL or surrogate characters."
    ),
    "N7": (
        "strict, when present, must be true; "
        "non-strict structured output is not supported."
    ),
    "N8": "Structured output is not supported for workforce keys.",
    "S1": (
        "The schema must be JSON of at most 16384 bytes, with no NaN or "
        "Infinity, no NUL or surrogate characters, and no excessively "
        "deep nesting."
    ),
    "S2": (
        "The schema has more than 2000 schema nodes, "
        "or a schema node nested deeper than 32 levels."
    ),
    "S3": (
        "This keyword or keyword value is not supported. Put constraints "
        "such as format, length, range or pattern in description instead."
    ),
    "S4": (
        "Each schema node needs one supported shape: type (optionally with "
        "enum or const), anyOf, or $ref; enum and const values must match "
        "type, and a nullable type with enum must list null."
    ),
    "S5": (
        'An object node needs properties, "additionalProperties": false and '
        "a required list that names every property exactly once; "
        "an array node needs items."
    ),
    "S6": 'The root of the schema must have "type": "object".',
    "S7": (
        "$defs is allowed only at the root; its keys must be 1 to 64 "
        'characters, each one of A-Z, a-z, 0-9, "_" or "-"; '
        "every entry must be referenced from the root."
    ),
    "S8": (
        "$ref must be the only key of its node and must name an existing "
        'entry as "#/$defs/<name>".'
    ),
    "S9": "Recursive references are not supported.",
    "S10": (
        "anyOf is supported only as a nullable value: exactly one "
        '{"type": "null"} branch and one branch with a single non-null '
        "type. Use several nullable fields instead of a union."
    ),
    "S11": (
        "The schema is too large after $ref references are expanded: "
        "more than 2000 nodes, depth above 32, or more than 65536 bytes "
        "as sent to the model."
    ),
    "S12a": (
        "Each enum needs 1 to 500 distinct values; each value must be a "
        "string of at most 256 characters, a finite number, a boolean or "
        'null; strings must not be blank or contain "__NOT_EXPRESSIBLE__".'
    ),
    "S12b": (
        "The schema has more than 5000 enum values in total "
        "after $ref references are expanded."
    ),
    "S13": (
        "Property names must be 1 to 64 characters, each one of A-Z, a-z, "
        '0-9, "_", "." or "-", and must not be "definitions", "-" or '
        "only digits."
    ),
}

_OUTER_KEYS = {
    "text": frozenset({"type"}),
    "json_schema": frozenset({"type", "json_schema"}),
}
_JSON_SCHEMA_KEYS = frozenset({"name", "schema", "strict", "description"})


def violation_message(rule: str, path: Sequence[PathSegment] = ()) -> str:
    """The message of a rejection: ``<rule> at <where>: <text>``.

    ``path`` is rooted at the ``response_format`` object for the ``N`` rules and
    at ``json_schema.schema`` for the ``S`` rules. An empty path is the root
    itself, written ``response_format`` or ``the schema root``.
    """
    where = render_error_path(list(path))
    if not where:
        where = "response_format" if rule.startswith("N") else "the schema root"
    return f"{rule} at {where}: {RULE_TEXTS[rule]}"


class ResponseFormatError(ValueError):
    """A ``response_format`` that breaks a rule.

    The v1 handlers turn it into a 422 ``invalid_input``; a downstream
    embedder's entry point can catch it the same way. ``str(error)`` is the
    whole message.
    """

    def __init__(self, rule: str, path: Sequence[PathSegment] = ()) -> None:
        self.rule = rule
        self.path = tuple(path)
        super().__init__(violation_message(rule, self.path))


@dataclass(frozen=True)
class ResponseFormat:
    """A checked ``json_schema`` format. ``schema`` is the caller's value,
    unchanged."""

    name: str
    schema: dict[str, Any]
    description: str | None


class _Clear(Enum):
    CLEAR = "clear"


# The result of ``{"type": "text"}``: remove the stored format.
CLEAR = _Clear.CLEAR

ResponseFormatChange = ResponseFormat | _Clear | None


def normalize_response_format(raw: Any) -> ResponseFormatChange:
    """Check the ``response_format`` of a request.

    Returns ``None`` when the request carries no format (the stored one stays),
    :data:`CLEAR` for ``{"type": "text"}``, and a :class:`ResponseFormat` for a
    valid ``json_schema`` format. Rules N1 to N7 run in that order, then the
    schema rules S1 to S13 on ``json_schema.schema``; the first violation
    raises :class:`ResponseFormatError`. Every JSON value is judged by its
    exact type, so a bool is never a number and ``1`` is never ``true``.

    Pure CPU work, so an async caller runs it in a worker thread.
    """
    if raw is None:
        return None
    if json_kind(raw) != "object":
        raise ResponseFormatError("N1")

    format_type = raw.get("type")
    if json_kind(format_type) != "string" or format_type not in _OUTER_KEYS:
        raise ResponseFormatError("N2", ["type"])
    allowed = _OUTER_KEYS[format_type]
    for key in raw:
        if key not in allowed:
            raise ResponseFormatError("N3", [key])
    if format_type == "text":
        return CLEAR

    body = raw.get("json_schema")
    if json_kind(body) != "object":
        raise ResponseFormatError("N4", ["json_schema"])
    for key in body:
        if key not in _JSON_SCHEMA_KEYS:
            raise ResponseFormatError("N4", ["json_schema", key])
    if "schema" not in body:
        raise ResponseFormatError("N4", ["json_schema", "schema"])

    name = body.get("name")
    if json_kind(name) != "string" or NAME_PATTERN.fullmatch(name) is None:
        raise ResponseFormatError("N5", ["json_schema", "name"])

    description: str | None = None
    if "description" in body:
        description = body["description"]
        # Measured before it is searched, so a huge value is refused cheaply.
        if (
            json_kind(description) != "string"
            or len(description) > MAX_TEXT_CHARS
            or has_forbidden_char(description)
        ):
            raise ResponseFormatError("N6", ["json_schema", "description"])

    if "strict" in body and body["strict"] is not True:
        raise ResponseFormatError("N7", ["json_schema", "strict"])

    schema = body["schema"]
    violation = check_schema_structure(schema)
    if violation is not None:
        raise ResponseFormatError(violation.rule, violation.path)
    return ResponseFormat(name=name, schema=schema, description=description)


def store_response_format(
    agent_config: Mapping[str, Any], change: ResponseFormatChange
) -> dict[str, Any]:
    """Apply ``change`` to a copy of a task's ``agent_config`` and return it.

    ``None`` leaves the format alone, :data:`CLEAR` removes it, and a
    :class:`ResponseFormat` replaces it with
    ``{"type": "json_schema", "name", "schema", "description"}``. The input is
    never modified. This is the only place that writes the format key.

    Downstream embedders that create or append to tasks themselves must call
    this too.
    """
    new_agent_config = dict(agent_config)
    if change is CLEAR:
        new_agent_config.pop(RESPONSE_FORMAT_CONFIG_KEY, None)
    elif isinstance(change, ResponseFormat):
        new_agent_config[RESPONSE_FORMAT_CONFIG_KEY] = {
            "type": "json_schema",
            "name": change.name,
            "schema": change.schema,
            "description": change.description,
        }
    return new_agent_config


_STORED_KEYS = frozenset({"type", "name", "schema", "description"})
# The order in which a missing key is reported.
_STORED_KEY_ORDER = ("type", "name", "schema", "description")


def _stored_violation(value: Any) -> tuple[str, list[PathSegment]] | None:
    """The first rule the stored ``value`` breaks and where, or ``None``.

    The steps run in order and a failed step ends the check. The path is rooted
    at the stored value for the shape and ``N`` rules and at the schema for the
    ``S`` rules.
    """
    if json_kind(value) != "object":
        return "stored_shape", []
    extra = [key for key in value if key not in _STORED_KEYS]
    if extra:
        return "stored_shape", [extra[0]]
    for key in _STORED_KEY_ORDER:
        if key not in value:
            return "stored_shape", [key]
    stored_type = value["type"]
    if json_kind(stored_type) != "string" or stored_type != "json_schema":
        return "stored_shape", ["type"]
    name = value["name"]
    if json_kind(name) != "string" or NAME_PATTERN.fullmatch(name) is None:
        return "N5", ["name"]
    description = value["description"]
    if description is not None and (
        json_kind(description) != "string"
        or len(description) > MAX_TEXT_CHARS
        or has_forbidden_char(description)
    ):
        return "N6", ["description"]
    violation = check_schema_structure(value["schema"])
    if violation is not None:
        return violation.rule, list(violation.path)
    return None


def parse_stored_response_format(
    value: Any, *, task_id: object = None
) -> StructuredOutputFormat:
    """Check the format stored on a task again and return it for a run.

    ``value`` is what :func:`store_response_format` wrote:
    ``{"type": "json_schema", "name", "schema", "description"}``. The shape of
    those four keys is checked first, then ``name`` and ``description`` as
    rules N5 and N6, then the schema as rules S1 to S13. The first failed step
    logs one WARNING with ``task_id``, the rule and the rendered path (only the
    keys and list indices that lead to the fault), never the name, the
    description or any string or number from the value, and returns
    :data:`StructuredOutputFormat.INVALID`. A
    value that passes is returned as a :class:`StructuredOutputFormat` that
    holds the stored objects themselves, uncopied.

    The request rules N1 to N4, N7 and N8 are not run: they describe the outer
    form of a request, which a stored value no longer has.

    Pure CPU work, so an async caller runs it in a worker thread.
    """
    found = _stored_violation(value)
    if found is None:
        return StructuredOutputFormat(
            name=value["name"],
            description=value["description"],
            schema=value["schema"],
        )
    rule, path = found
    where = render_error_path(path) or (
        "the schema root" if rule.startswith("S") else "response_format"
    )
    logger.warning(
        "task_id=%s stored response_format is not valid: rule=%s at %s",
        task_id,
        rule,
        where,
    )
    return StructuredOutputFormat.INVALID


def echo_for_change(change: ResponseFormatChange) -> tuple[str, str]:
    """The ``(response_format_type, response_format_name)`` of a task that
    starts with ``change`` and no earlier format."""
    if isinstance(change, ResponseFormat):
        return "json_schema", change.name
    return "text", ""


def echo_for_config(agent_config: Any) -> tuple[str, str]:
    """The ``(response_format_type, response_format_name)`` a task's stored
    ``agent_config`` reports.

    A stored value that is a dict with ``type == "json_schema"`` and a string
    ``name`` reports ``("json_schema", name)``; anything else reports
    ``("text", "")``. The schema itself is not checked here.
    """
    if type(agent_config) is not dict:
        return "text", ""
    stored = agent_config.get(RESPONSE_FORMAT_CONFIG_KEY)
    if type(stored) is not dict:
        return "text", ""
    name = stored.get("name")
    if stored.get("type") == "json_schema" and type(name) is str:
        return "json_schema", name
    return "text", ""
