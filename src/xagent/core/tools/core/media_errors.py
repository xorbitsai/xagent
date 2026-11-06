"""What a failed media call may say, and what it must never say.

The media tools call third-party services -- Ark/Seedance, OpenAI, Gemini,
DashScope, ElevenLabs, Xinference -- and every one of them fails with a message
written for the account holder rather than for the person waiting on a video.
An Ark 404 for a model the deployment never activated arrives as::

    Error code: 404 - {'error': {'code': 'ModelNotOpen', 'message': 'Your
    account 2100...4339 has not activated the model doubao-seedance-2-0-fast-
    260128. Please activate the model service in the Ark Console. Request id:
    0217...f8', ...}}, request_id: 2026...zEj

Tool results used to hand that string to the model as ``error``. The model,
being helpful, repeated it: the platform's own account id, the provider request
id, the provider model id and the provider console name all reached an end user
in a chat message. The failure was real and worth reporting; everything that
made it identifiable was not.

So a failed media call now reports three things and nothing else:

* a category from :data:`MEDIA_FAILURE_CATEGORIES` -- stable, greppable, and
  what an alert would fire on;
* one plain sentence carrying no id, model name, path, provider console or
  error class;
* for a rejected parameter, the values the caller can actually change to.

The exception itself still goes to the log, at the level it went before, with
its ids rubbed out (:func:`redact_provider_detail`) -- an operator needs the
provider's own words, an end user does not. The adapters log through the same
function, so an id is masked in their warning lines too, not only in the one
line a tool writes. ``model_not_enabled`` also writes a fixed second line,
``media model not enabled: provider=<p> model=<m>``, so a deployment that never
activated a model is one query away instead of one support thread away.

Classification is duck-typed over the exception's ``__cause__``/``__context__``
chain rather than by provider class: the adapters wrap provider errors in
``RuntimeError`` in some paths and let them through untouched in others, and a
media tool must not have to import six SDKs to name a failure. Anything the
chain does not explain is ``unknown`` -- the honest answer, and the safe one:
``unknown`` promises the caller nothing, while a wrongly-guessed
``invalid_request`` would send them to change a parameter that was never the
problem. The one thing an adapter can state instead of leaving to inference is
:class:`MediaParameterError`, which carries the field the provider rejected and
the values it accepts.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping
from typing import Any, Optional

from ...utils.security import is_credential_key, redact_sensitive_text

MODEL_NOT_ENABLED = "model_not_enabled"
INVALID_REQUEST = "invalid_request"
PROVIDER_UNAVAILABLE = "provider_unavailable"
UNKNOWN = "unknown"

MEDIA_FAILURE_CATEGORIES = frozenset(
    {
        MODEL_NOT_ENABLED,
        INVALID_REQUEST,
        PROVIDER_UNAVAILABLE,
        UNKNOWN,
    }
)

# One sentence per category, and deliberately nothing else: no provider name,
# no model id, no "check your API key", no error class. The model turns this
# into the reply, so anything that reads like an internal diagnostic here is
# one paraphrase away from the user.
_FAILURE_MESSAGES = {
    MODEL_NOT_ENABLED: (
        "This model is not available here right now, and changing the request "
        "will not help."
    ),
    INVALID_REQUEST: (
        "The request was rejected because a parameter is outside what this "
        "model supports."
    ),
    PROVIDER_UNAVAILABLE: (
        "The media service failed to answer. The same request may work later."
    ),
    UNKNOWN: "The request failed, and the reason is not available.",
}

# Provider-side wording for "you may not use this model". Ark reports it as
# HTTP 404 with ``code=ModelNotOpen``; the OpenAI-compatible SDKs answer
# ``model_not_found``; Gemini says the model "is not found for API version".
# Matched case-insensitively against both the structured code and the message.
_MODEL_NOT_ENABLED_MARKERS = (
    "modelnotopen",
    "model_not_open",
    "model not open",
    "modelnotfound",
    "model_not_found",
    "model not found",
    "has not activated",
    "not activated",
    "not subscribed",
    "does not have access to it",
    "no access to the model",
    "model is not enabled",
)

# Provider-side wording for "the request itself is wrong".
_INVALID_REQUEST_MARKERS = (
    "invalidparameter",
    "invalid_parameter",
    "invalid parameter",
    "invalidargument",
    "invalid_argument",
    "validationerror",
    "validation_error",
    "validation error",
    "badrequest",
    "bad_request",
    "bad request",
    "invalidrequest",
    "invalid_request",
    "missingparameter",
    "missing parameter",
    "unprocessable",
    "must be between",
    # A voice the caller named that the account cannot use. ElevenLabs answers
    # 404 ``voice_not_found``; the fix is to list the voices and pick one, so
    # it belongs with the parameter errors rather than with a missing route.
    "voice_not_found",
    "voice not found",
)

# Provider-side wording for "try again later". Transport failures (timeouts,
# refused connections) are matched by class name instead, because their
# message is the transport library's, not the provider's.
_PROVIDER_UNAVAILABLE_MARKERS = (
    "throttl",
    "ratelimit",
    "rate_limit",
    "rate limit",
    "toomanyrequests",
    "too many requests",
    "serviceunavailable",
    "service_unavailable",
    "service unavailable",
    "internalerror",
    "internal_error",
    "internal server error",
    "servererror",
    "overloaded",
    "badgateway",
    "bad gateway",
    "gatewaytimeout",
    "gateway timeout",
    "timed out",
    "timeout",
)

_PROVIDER_UNAVAILABLE_CLASS_NAMES = (
    "apiconnectionerror",
    "apitimeouterror",
    "connecterror",
    "connectionerror",
    "connectionreseterror",
    "readtimeout",
    "connecttimeout",
    "timeout",
    "timeouterror",
    "transport",
)

# HTTP statuses that mean the caller has to change the request.
_INVALID_REQUEST_STATUSES = frozenset({400, 409, 413, 415, 422})
# Transient by definition: 408 request timeout, 425 too early, 429 too many
# requests, and every 5xx. 401/402/403 are deliberately absent: they are a
# deployment's problem with a credential or a bill, retrying will not fix them,
# and calling them "unavailable" would promise a retry that cannot work.
_PROVIDER_UNAVAILABLE_STATUSES = frozenset({408, 425, 429})

_MAX_CAUSE_DEPTH = 8

_REDACTED = "<redacted>"

# A field the provider names, then ``:`` or ``=``, then its value, quoted or
# not. The value ends at its own quote, a comma, a semicolon, a bracket or
# whitespace, so masking it leaves the surrounding JSON or repr intact: the
# earlier version consumed ``\S+`` and ate the quote and the colon instead of
# the value, turning ``{'request_id': 'req_short_id'}`` into
# ``{'request_id=<redacted> 'req_short_id'}`` -- balanced only by accident and
# with the id still there.
_STRUCTURED_FIELD_PATTERN = re.compile(
    r"""(?ix)
    (?<![A-Za-z0-9_-])
    (?P<name_quote>["']?)
    (?P<name>[A-Za-z][A-Za-z0-9 _-]{0,40})
    (?P=name_quote)
    \s*[:=]\s*
    (?P<value>
        "(?:[^"\\\n]|\\.)*"      # one quoted value, a backslash escape and all
      | '(?:[^'\\\n]|\\.)*'
      | \[[^\[\]{}]*\]           # a flat array of them, masked as one value
      | [^\s,;:()[\]{}"']+
    )
    """
)

# A label, whitespace and a value with no punctuation between them at all:
# ``Request id req_short_id``, ``Your account acct_abc has not activated``.
# Only labels that name an identifier are matched, and the value has to look
# like one -- a digit or an underscore in it -- so ordinary prose such as
# ``your account is not activated`` keeps its words.
_PROSE_IDENTIFIER_PATTERN = re.compile(
    r"""(?ix)
    \b
    (?P<label>
        request[ _-]?id
      | (?:account|organisation|organization|org)[ _-]?(?:id|number)
      | account
    )
    \s+
    (?P<value>(?=[A-Za-z0-9_.:-]*[0-9_])[A-Za-z0-9][A-Za-z0-9_.:-]*)
    """
)

# Shapes masked wherever they appear, whatever the field around them. These
# are the fallback for text that names no field at all (``Your account
# 2100...4339 has not activated ...``) and for the short ids the field rules
# cannot see because the provider only wrote the value.
_SHAPE_RULES = (
    # API keys and bearer tokens, which several SDKs echo in their message.
    (re.compile(r"(?i)\b(?:sk|pk|rk)[-_][A-Za-z0-9._-]{8,}"), _REDACTED),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{8,}"), "bearer <redacted>"),
    # UUIDs: provider request ids as often as resource ids.
    (
        re.compile(
            r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
        ),
        _REDACTED,
    ),
    # Long runs mixing letters and digits: request ids and opaque tokens, in
    # every shape the providers use for them.
    (
        re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{20,}\b"),
        _REDACTED,
    ),
    # Long digit runs: account ids and epochs. Seven digits leaves model name
    # suffixes (``260128``) and port numbers readable.
    (re.compile(r"\b\d{7,}\b"), _REDACTED),
)

_MAX_LOGGED_DETAIL = 300


def _normalize_field_name(name: str) -> str:
    """``request id`` / ``requestId`` / ``request-id`` -> ``request_id``."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    return re.sub(r"[\s-]+", "_", spaced).strip("_").lower()


def _is_identifier_field(name: str) -> bool:
    """Whether ``name`` labels an identifier rather than prose about one."""
    normalized = _normalize_field_name(name)
    if not normalized:
        return False
    return normalized == "id" or normalized.endswith(("_id", "_ids", "_uuid", "_guid"))


# Header names whose whole value is a credential, whatever the value's shape.
_AUTH_HEADER_FIELDS = frozenset({"authorization", "proxy_authorization"})


def _is_secret_field(name: str) -> bool:
    """Whether the value of the structured field ``name`` must be masked.

    Credential names reuse :func:`is_credential_key`, the rule the repository
    already applies to ``identifier=value`` assignments, so a field named
    ``api_key`` or ``access_token`` is masked here in the quoted shapes that
    rule does not parse. A header that carries nothing but a credential is
    masked under its own name: ``Authorization`` does not end in a credential
    word, and a short ``Bearer`` value is below every shape rule.
    """
    normalized = _normalize_field_name(name)
    if not normalized:
        return False
    return (
        _is_identifier_field(normalized)
        or is_credential_key(normalized)
        or normalized in _AUTH_HEADER_FIELDS
    )


# A quoted run of text: a JSON value, a repr string. ``"""`` does not occur in
# provider messages, and a value with a newline is split by the caller before
# this runs, so one line at a time is enough. A backslash escape is part of
# the run it sits in, so ``"req_\_after"`` is read as one value rather than
# closing the quote early.
_QUOTED_SECTION_PATTERN = re.compile(
    r"""(?x)
    "(?:[^"\\\n]|\\.)*"
    |
    '(?:[^'\\\n]|\\.)*'
    """
)


def _mask_field_assignments(text: str) -> str:
    """Mask the value of every structured field the rules call secret."""
    parts: list[str] = []
    copied = 0
    for match in _STRUCTURED_FIELD_PATTERN.finditer(text):
        if match.start() < copied or not _is_secret_field(match.group("name")):
            continue
        parts.append(text[copied : match.start("value")])
        parts.append(_REDACTED)
        copied = match.end("value")
    parts.append(text[copied:])
    return "".join(parts)


def _mask_structured_fields(text: str) -> str:
    """Mask labelled ids and credentials, including those inside a message.

    The quoted contents are masked first: a provider that nests its real error
    inside ``"message": "..."`` (every OpenAI-compatible SDK does) would
    otherwise be skipped whole by the outer scan, which matches the outer
    field and moves past its value -- and that value is where the account id
    and the request id live. The last pass reads the labels a provider writes
    as prose, with no ``:`` or ``=`` between the label and the value.
    """

    def _mask_section(match: re.Match[str]) -> str:
        quote = match.group(0)[0]
        return f"{quote}{_mask_field_assignments(match.group(0)[1:-1])}{quote}"

    masked_inside = _QUOTED_SECTION_PATTERN.sub(_mask_section, text)
    masked = _mask_field_assignments(masked_inside)
    return _PROSE_IDENTIFIER_PATTERN.sub(
        lambda match: f"{match.group('label')} {_REDACTED}", masked
    )


def redact_provider_detail(text: Any) -> str:
    """``text`` on one line, with ids, keys and opaque tokens rubbed out.

    For the log only. The result is not client-safe either way -- it is the
    provider's own message -- but an operator reading it should not be looking
    at a customer's account id, and an operator's log search should not be able
    to turn one up.

    The repository's credential rules (:func:`redact_sensitive_text`) run
    first, then the labelled id/credential fields, then the shapes that need
    no label. Both media adapters and media tools go through this one
    function, so an id that is hidden in a tool result is hidden in the
    adapter's own log line too.
    """
    cleaned = " ".join(str(text or "").split())
    cleaned = redact_sensitive_text(cleaned)
    cleaned = _mask_structured_fields(cleaned)
    for pattern, replacement in _SHAPE_RULES:
        cleaned = pattern.sub(replacement, cleaned)
    if len(cleaned) > _MAX_LOGGED_DETAIL:
        cleaned = cleaned[:_MAX_LOGGED_DETAIL].rstrip() + "…"
    return cleaned


class MediaParameterError(ValueError):
    """A request parameter the caller can change, and the values it accepts.

    The adapters check a few parameters before the provider sees them -- an
    output format the client cannot encode, a clip length outside the model's
    range, a prompt over the limit. A bare ``ValueError`` says only that
    something was wrong, which leaves the failure ``unknown`` and the caller
    with nothing to change. This carries the field name and, where the adapter
    knows them, the accepted values.

    Raise it only for a parameter the caller supplied and can change; a bug in
    our own code is not the caller's request error.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str,
        allowed_values: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        # ``None`` rather than an empty dict: no reliable range means the
        # failure says "the request was rejected" and stops there, instead of
        # inventing values the caller would then try and fail with.
        self.allowed_values = dict(allowed_values) if allowed_values else None


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and the causes/contexts underneath it, outermost first."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    depth = 0
    while current is not None and depth < _MAX_CAUSE_DEPTH:
        if id(current) in seen:
            return
        seen.add(id(current))
        yield current
        depth += 1
        current = current.__cause__ or current.__context__


def _status_code(exc: BaseException) -> Optional[int]:
    """The HTTP status behind ``exc``, however its SDK exposes it."""
    for attribute in ("status_code", "http_status", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            if 100 <= value <= 599:
                return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool):
        if 100 <= value <= 599:
            return value
    # Google's SDK keeps the status in ``.code`` as an int.
    value = getattr(exc, "code", None)
    if isinstance(value, int) and not isinstance(value, bool):
        if 100 <= value <= 599:
            return value
    return None


def _code_strings(exc: BaseException) -> list[str]:
    """Every provider-side code-like string ``exc`` carries."""
    codes: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            codes.append(value)

    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            for key in ("code", "type", "status"):
                _add(error.get(key))
            _add(error.get("message"))
        for key in ("code", "type", "message"):
            _add(body.get(key))
    error = getattr(exc, "error", None)
    if isinstance(error, Mapping):
        for key in ("code", "type", "message"):
            _add(error.get(key))
    for attribute in ("provider_code", "error_code", "code", "status", "message"):
        _add(getattr(exc, attribute, None))
    return codes


def _code_and_target_strings(exc: BaseException) -> list[str]:
    """The provider's code and the request target it blames, as strings.

    Only the fields that say *what* was refused, never the message: a message
    mentioning a model does not prove the model was the thing refused.
    """
    values: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            values.append(value)

    for link in _exception_chain(exc):
        containers: list[Any] = [link]
        body = getattr(link, "body", None)
        if isinstance(body, Mapping):
            containers.append(body)
            error = body.get("error")
            if isinstance(error, Mapping):
                containers.append(error)
        for container in containers:
            if isinstance(container, Mapping):
                for key in ("code", "param", "target", "field"):
                    _add(container.get(key))
            else:
                for attribute in ("code", "param", "target", "field"):
                    _add(getattr(container, attribute, None))
    return values


def classify_media_failure(exc: BaseException) -> str:
    """The category for ``exc``, from its chain and the HTTP status it carries."""
    statuses: list[int] = []
    haystack: list[str] = []
    class_names: list[str] = []
    parameter_rejected = False
    for link in _exception_chain(exc):
        if isinstance(link, MediaParameterError):
            parameter_rejected = True
        status = _status_code(link)
        if status is not None:
            statuses.append(status)
        haystack.extend(_code_strings(link))
        haystack.append(str(link))
        class_names.append(type(link).__name__.lower())

    # An adapter that already knows the caller passed something it cannot use
    # says so, in its own type; that beats any guess from a status.
    if parameter_rejected:
        return INVALID_REQUEST

    lowered = " ".join(haystack).lower()

    if any(marker in lowered for marker in _MODEL_NOT_ENABLED_MARKERS):
        return MODEL_NOT_ENABLED
    # A 404 is "this deployment cannot use the model" only when the provider
    # says so: a code or request target that names a model. The same status
    # answers a missing route, a voice that does not exist and a deleted
    # file, and calling those "not enabled, and changing the request will not
    # help" both misleads the caller and fires the not-enabled alert.
    if 404 in statuses and any(
        "model" in value.lower() for value in _code_and_target_strings(exc)
    ):
        return MODEL_NOT_ENABLED
    if any(status in _INVALID_REQUEST_STATUSES for status in statuses):
        return INVALID_REQUEST
    if any(marker in lowered for marker in _INVALID_REQUEST_MARKERS):
        return INVALID_REQUEST
    if any(status in _PROVIDER_UNAVAILABLE_STATUSES for status in statuses):
        return PROVIDER_UNAVAILABLE
    if any(status >= 500 for status in statuses):
        return PROVIDER_UNAVAILABLE
    if any(marker in lowered for marker in _PROVIDER_UNAVAILABLE_MARKERS):
        return PROVIDER_UNAVAILABLE
    if any(
        marker in name
        for name in class_names
        for marker in _PROVIDER_UNAVAILABLE_CLASS_NAMES
    ):
        return PROVIDER_UNAVAILABLE
    return UNKNOWN


def media_error_fields(
    category: str,
    *,
    allowed_values: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The failure fields a media tool result carries for ``category``.

    ``allowed_values`` is attached only for :data:`INVALID_REQUEST`: it is the
    list of things the caller can change, and it is meaningless -- worse,
    misleading -- on a failure that is not theirs to fix. Like every other
    value in these fields it is written for the model to repeat, so it holds
    numbers, sizes and formats, never an id.
    """
    if category not in MEDIA_FAILURE_CATEGORIES:
        category = UNKNOWN
    fields: dict[str, Any] = {
        "error": _FAILURE_MESSAGES[category],
        "error_category": category,
    }
    if allowed_values and category == INVALID_REQUEST:
        fields["allowed_values"] = dict(allowed_values)
    return fields


def _parameter_error(exc: BaseException) -> Optional[MediaParameterError]:
    """The typed parameter error in ``exc``'s chain, if an adapter raised one.

    Its presence is what separates "the adapter rejected this field and knows
    what it accepts" from "a provider rejected the request on the far side":
    only the first can state a range, including stating that there is none.
    """
    for link in _exception_chain(exc):
        if isinstance(link, MediaParameterError):
            return link
    return None


def _model_identity(model: Any) -> tuple[str, str]:
    """``(provider, model name)`` for the not-enabled log marker."""
    for attribute in ("model_provider", "provider_name", "provider"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value.strip():
            provider = value.strip()
            break
    else:
        provider = "unknown"
    for attribute in ("model_name", "model", "model_id"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
    else:
        name = "unknown"
    return provider, name


def media_failure_fields(
    exc: BaseException,
    *,
    logger: logging.Logger,
    action: str,
    model: Any = None,
) -> dict[str, Any]:
    """Log ``exc`` and return the failure fields that replace ``str(exc)``.

    ``action`` is the caller's own phrase ("Video generation failed"), kept so
    the existing per-tool log lines stay recognizable. The exception's class
    and redacted message follow it, and ``model_not_enabled`` adds the marker
    line operators alert on.

    A range comes only from a :class:`MediaParameterError`, because only that
    adapter has named the rejected field. A provider rejection arrives without
    one: the body may name a field, a different field, or none at all, and a
    list for a field it never blamed sends the caller to change the wrong
    thing. Such a failure carries no ``allowed_values``.
    """
    category = classify_media_failure(exc)
    logger.error(
        "%s (category=%s): %s: %s",
        action,
        category,
        type(exc).__name__,
        redact_provider_detail(exc),
    )
    if category == MODEL_NOT_ENABLED:
        provider, name = _model_identity(model)
        logger.error("media model not enabled: provider=%s model=%s", provider, name)
    rejected = _parameter_error(exc)
    return media_error_fields(
        category,
        allowed_values=rejected.allowed_values if rejected is not None else None,
    )
