"""Classify the provider fault behind a failed durable-storage operation.

``ManagedFileRef`` wraps every provider exception into
``DurableStorageOperationError`` -- a static message, with the storage key on
its ``storage_key`` attribute (#1643) -- and #1467 restored the ``__cause__``
chain to the logs. A traceback is readable but
not *queryable*: an operator triaging a burst of 503s wants to know whether
they are looking at one throttle, a thousand throttles, or a rejected
credential, and that question is a field aggregation, not a text search.

This module turns the chain into structured fields. It answers only "what was
it, and is retrying plausible" -- it does not decide HTTP status. Every fault
in this family still answers 503; see ``ProviderFault.retryable``.

**Why the chain must be walked.** s3fs does not surface botocore's
``ClientError``. Its ``translate_boto_error`` maps recognized codes onto
``OSError`` subclasses (``PermissionError`` for ``AccessDenied``, and so on),
falls back to ``OSError(EIO, ...)`` for the rest, and sets ``__cause__`` to the
original ``ClientError``. So the provider code lives two levels below the wrap:

    DurableStorageOperationError("Failed to write durable object")
      __cause__ -> PermissionError("Access Denied")          <- s3fs
        __cause__ -> ClientError(response={"Error": {"Code": "AccessDenied"}})

Only the innermost link carries the code, which is why classification starts at
the outermost exception and keeps descending.

**Why duck typing rather than ``except ClientError``.** botocore is declared
but the storage backend is pluggable (local filesystem, and the alternative
vector/object backends behind extras), so this code must not require botocore
to be importable, and must not silently stop classifying when a deployment
swaps s3fs for another fsspec implementation. Reading ``.response`` off
whatever is in the chain works for anything that follows botocore's shape and
degrades to ``retryable=None`` for anything that does not.
"""

from __future__ import annotations

import errno
from dataclasses import dataclass
from typing import Iterator

# Provider error codes worth another attempt. Sourced from botocore's own
# retry configuration (`botocore/data/_retry.json`, whose throttling list is
# cross-service) plus the S3 error reference for the S3-specific ones -- not
# from the S3 reference alone, which does not say what botocore retries.
_RETRYABLE_CODES = frozenset(
    {
        "BandwidthLimitExceeded",
        "InternalError",
        # AWS documents this as "a conflicting conditional operation is in
        # progress against this resource; try again", so it is transient
        # despite being a 4xx.
        "OperationAborted",
        "PriorRequestNotComplete",
        "RequestLimitExceeded",
        "RequestThrottled",
        "RequestThrottledException",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "SlowDown",
        "ThrottledException",
        "Throttling",
        "ThrottlingException",
        "TooManyRequestsException",
        "TransactionInProgressException",
        "LimitExceededException",
    }
)

# Permanent causes worth naming even without an HTTP status.
#
# Trimmed to credential/configuration faults, on review: every 4xx code here
# would already be called permanent by ``_retryable_from_status``, so the table
# only decides when a response carries no ``ResponseMetadata`` -- which a
# strictly-conforming botocore response always does, but a loosely-conforming
# backend need not. These are the ones whose verdict an operator most needs to
# be right in that case: retrying a rejected credential forever is the failure
# mode #1467 set out to make diagnosable, and it is worth one frozenset to be
# certain of it. Codes whose only information was their own status are gone.
_PERMANENT_CODES = frozenset(
    {
        "AccessDenied",
        "AccountProblem",
        "AllAccessDisabled",
        "CredentialsNotSupported",
        "ExpiredToken",
        "InvalidAccessKeyId",
        "InvalidSecurity",
        "InvalidToken",
        "NoSuchBucket",
        "SignatureDoesNotMatch",
        "TokenRefreshRequired",
    }
)

# Exactly ``s3fs.core.S3_RETRYABLE_ERRORS`` -- the set the installed backend
# itself retries on -- so a fault reaching us wearing one of these names has
# already exhausted the backend's own attempts. Authoritative: it is copied
# from a published constant, and drift is visible by diffing against it.
_S3FS_RETRYABLE_EXC_NAMES = frozenset(
    {
        "ClientPayloadError",
        "FSTimeoutError",
        "HTTPClientError",
        "IncompleteRead",
        "ResponseParserError",
    }
)

# Transport-level botocore/urllib3 names not in that constant. Inferred from
# what each class means, not from a published retry list -- kept separate so
# the distinction survives a reordering, which a single set could not express.
_INFERRED_RETRYABLE_EXC_NAMES = frozenset(
    {
        "ConnectionClosedError",
        "ConnectionError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "IncompleteReadError",
        "ProtocolError",
        "ReadTimeoutError",
        "ResponseStreamingError",
        "TimeoutError",
    }
)

_RETRYABLE_EXC_NAMES = _S3FS_RETRYABLE_EXC_NAMES | _INFERRED_RETRYABLE_EXC_NAMES

# Credential/config faults raised before a request is ever signed.
_PERMANENT_EXC_NAMES = frozenset(
    {
        "EndpointResolutionError",
        "NoCredentialsError",
        "NoRegionError",
        "ParamValidationError",
        "PartialCredentialsError",
        "ProfileNotFound",
        "UnknownServiceError",
    }
)


def _errnos(*names: str) -> frozenset[int]:
    """Resolve ``errno`` names, skipping any the platform does not define.

    Named lookup rather than attribute access because these tables are built at
    import time: an ``AttributeError`` for an errno some platform does not
    define would fail the import of every consumer, which is a far worse
    outcome than losing one entry from a classification table.
    """
    resolved = (getattr(errno, name, None) for name in names)
    return frozenset(value for value in resolved if isinstance(value, int))


# ``errno`` values for backends that fail as plain OS errors -- the local
# filesystem backend, and s3fs's unmapped-code fallback.
_RETRYABLE_ERRNOS = _errnos(
    "EAGAIN",
    "EBUSY",
    "ECONNABORTED",
    "ECONNREFUSED",
    "ECONNRESET",
    "EHOSTUNREACH",
    "ENETDOWN",
    "ENETUNREACH",
    "EPIPE",
    "ETIMEDOUT",
)

# ENOSPC and EDQUOT are permanent on purpose: the write cannot succeed until an
# operator frees space or raises a quota, so retrying only amplifies load.
_PERMANENT_ERRNOS = _errnos(
    "EACCES",
    "EDQUOT",
    "EFBIG",
    "EISDIR",
    "ENAMETOOLONG",
    "ENOSPC",
    "ENOTDIR",
    "EPERM",
    "EROFS",
)

_MAX_CHAIN_DEPTH = 12


def _safe_getattr(exc: BaseException, name: str) -> object | None:
    """Read an attribute that a foreign exception may implement as a property.

    ``getattr(obj, name, default)`` only swallows ``AttributeError``. This
    module deliberately duck-types third-party exception shapes, which is
    precisely the category that can implement ``.response`` or ``.errno`` as a
    property doing real work -- and a raise from one would propagate out of the
    caller's ``except`` block and replace a handled 503 with an unhandled 500.
    """
    try:
        return getattr(exc, name, None)
    except Exception:
        return None


@dataclass(frozen=True)
class ProviderFault:
    """What the object store actually said, as structured data.

    ``retryable`` is ``None`` when the chain could not be classified, which is
    deliberately distinct from ``False``: "we know this is permanent" and "we do
    not recognize this" call for different operator responses, and collapsing
    them would let an unrecognized fault masquerade as a diagnosed one. The
    rendered form spells the unknown case ``retryable=unknown`` rather than
    omitting the field, so a classified-but-unknown line is distinguishable from
    a line written before classification existed.

    It is a *diagnostic hint with no behavioural consumer*: nothing branches on
    it, and every fault in this family still answers the same HTTP status.
    Deliberately unlike ``BackgroundJobHandlerError.retryable``
    (web/jobs/exceptions.py), which web/jobs/tasks.py reads to drive real retry
    routing, and unrelated to ``retry_on`` in core/model/chat/error.py.

    ``code`` and ``errno_name`` are separate fields because they are separate
    vocabularies -- ``SlowDown`` is an S3 error code, ``EBUSY`` is an OS errno,
    and putting both in one field would make it impossible to tell which
    namespace a value came from when aggregating.
    """

    provider_class: str
    code: str | None = None
    errno_name: str | None = None
    http_status: int | None = None
    retryable: bool | None = None

    def as_fields(self) -> dict[str, object]:
        """Return the fault as data, for a caller to render.

        Returns fields rather than a formatted string so that one renderer --
        the caller's, which also handles its own request identifiers -- owns
        escaping and layout for the whole line. A second renderer here would
        have to keep its separator and sanitising rules in sync with that one,
        and log-shape knowledge does not belong in this layer anyway.
        """
        return {
            "provider_class": self.provider_class,
            "provider_code": self.code,
            "provider_errno": self.errno_name,
            "provider_http_status": self.http_status,
            "retryable": "unknown" if self.retryable is None else self.retryable,
        }


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """Walk ``__cause__`` then ``__context__``, outermost first.

    ``__context__`` is followed as well because an exception raised inside an
    ``except`` block without ``from`` still records what it displaced, and
    fsspec implementations are not uniformly careful about explicit chaining.
    ``raise ... from None`` is respected, though: ``__suppress_context__``
    means the author deliberately hid what was displaced, and classifying on it
    anyway would attribute the fault to an exception the raiser judged
    unrelated. This mirrors what a printed traceback shows.

    Depth is bounded and identity is tracked: a chain can be cyclic.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < _MAX_CHAIN_DEPTH:
        if id(current) in seen:
            return
        seen.add(id(current))
        yield current
        depth += 1
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            return
        else:
            current = current.__context__


# Statuses whose retry semantics are guaranteed, as an allowlist rather than
# ranges. A range says "everything in 5xx is transient", which is a claim about
# statuses nobody has looked at: 505 (version not supported) and 508 (loop
# detected) are permanent, and an unrecognised 409 can be either -- a
# concurrent-operation conflict clears itself, a `BucketNotEmpty` never does.
# Reporting those as retryable is a wrong verdict, not a missing one, and a
# wrong verdict is worse than ``unknown`` for something whose entire job is to
# tell an operator whether waiting can help.
#
# A code that *is* recognised still decides on its own and outranks this, which
# is how `OperationAborted` stays retryable at 409 (see ``_RETRYABLE_CODES``).
_RETRYABLE_STATUSES = frozenset(
    {
        408,  # request timeout, server-side
        429,  # too many requests
        500,  # internal error
        502,  # bad gateway
        503,  # service unavailable
        504,  # gateway timeout
        509,  # bandwidth limit exceeded (non-standard, used by some providers)
    }
)

# 4xx statuses whose retry semantics are genuinely ambiguous, so neither the
# default below nor a guess applies. 409 is the case that prompted this: a
# concurrent-operation conflict clears itself, while ``BucketNotEmpty`` at the
# same status never does, and only the provider code can tell them apart.
# 429 is deliberately absent: it is unconditionally retryable and answered by
# _RETRYABLE_STATUSES above, so listing it here would be unreachable.
_AMBIGUOUS_4XX = frozenset({409})

# 5xx statuses that are permanent despite their range, which is the direction a
# range rule gets wrong: it reads the whole 5xx block as transient.
_PERMANENT_5XX = frozenset(
    {
        501,  # not implemented -- this backend cannot do it, ever
        505,  # HTTP version not supported
        508,  # loop detected
    }
)


def _retryable_from_status(status: int) -> bool | None:
    """Classify from an HTTP status, or return ``None`` when it does not say.

    The 4xx and 5xx halves get opposite defaults, because their ranges carry
    opposite information. A 4xx describes the request, so absent a reason to
    think otherwise it cannot be fixed by waiting -- 410 and 412 are permanent
    whether or not anyone thought to list them. A 5xx describes the server, and
    only the specific codes below are known to be permanent; the rest of that
    range says nothing an operator can act on.

    Listing the *exceptions* rather than the permanent members is what keeps
    this honest: an earlier revision enumerated ten permanent 4xx and let the
    other forty fall through to ``None``, which discarded correct verdicts
    while claiming to be conservative.
    """
    if status in _RETRYABLE_STATUSES:
        return True
    if 400 <= status <= 499:
        # Ambiguous ones need the provider code, which has already had its say
        # by the time we get here.
        return None if status in _AMBIGUOUS_4XX else False
    if status in _PERMANENT_5XX:
        return False
    # An unclassified 5xx is exactly what this field must not guess at.
    return None


def _classify_from_response(exc: BaseException) -> ProviderFault | None:
    """Classify from a botocore-shaped ``.response``, the richest source."""
    provider_class = type(exc).__name__

    response = _safe_getattr(exc, "response")
    if isinstance(response, dict):
        error = response.get("Error")
        # Every value read out of ``response`` is untrusted: this module
        # deliberately duck-types the botocore shape so foreign backends still
        # classify, which means a library that follows the shape only loosely
        # can put anything here. Narrowing each one to the type this function
        # actually uses -- rather than guarding at the point of use -- keeps a
        # nested dict or list out of both the set lookups below (unhashable,
        # and a raised TypeError here would replace the caller's 503 with an
        # unhandled 500 precisely while an incident is being diagnosed) and out
        # of the log field, where ``str()`` of a dict is noise, not a code.
        code = error.get("Code") if isinstance(error, dict) else None
        code = code if isinstance(code, str) and code else None
        metadata = response.get("ResponseMetadata")
        status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
        # ``bool`` subclasses ``int``, so a ``True`` here would otherwise
        # render as ``provider_http_status=True``.
        status = (
            status if isinstance(status, int) and not isinstance(status, bool) else None
        )
        if code is not None or status is not None:
            return ProviderFault(
                provider_class=provider_class,
                code=code,
                http_status=status,
                retryable=_retryable_from_code(code, status),
            )

    return None


def _classify_from_shape(exc: BaseException) -> ProviderFault | None:
    """Classify from the exception class or its ``errno`` -- the coarse sources.

    Only consulted once no link in the chain offered a ``.response``, because
    both are lossy. s3fs maps ``SlowDown``, ``ServiceUnavailable`` and
    ``OperationAborted`` onto the *same* ``OSError(EBUSY)``, so an errno-derived
    answer cannot tell a throttle from a service outage from an aborted
    multipart -- exactly the distinction this module exists to make. Preferring
    the deeper ``.response`` recovers it.
    """
    provider_class = type(exc).__name__

    if provider_class in _RETRYABLE_EXC_NAMES:
        return ProviderFault(provider_class=provider_class, retryable=True)
    if provider_class in _PERMANENT_EXC_NAMES:
        return ProviderFault(provider_class=provider_class, retryable=False)

    code_attr = _safe_getattr(exc, "errno")
    # No errno needs excluding as "uninformative" any more: a ``.response``
    # anywhere in the chain already outranks every errno, so a coarse frame can
    # no longer mask a precise one.
    if isinstance(code_attr, int) and not isinstance(code_attr, bool):
        if code_attr in _RETRYABLE_ERRNOS:
            retryable: bool | None = True
        elif code_attr in _PERMANENT_ERRNOS:
            retryable = False
        else:
            return None
        return ProviderFault(
            provider_class=provider_class,
            errno_name=errno.errorcode.get(code_attr, str(code_attr)),
            retryable=retryable,
        )

    return None


def _retryable_from_code(code: str | None, status: int | None) -> bool | None:
    """Decide from the provider code, falling back to the HTTP status.

    ``code`` is already narrowed to ``str | None`` by the caller, which is what
    makes the set lookups below safe rather than a hazard.
    """
    if code in _RETRYABLE_CODES:
        return True
    if code in _PERMANENT_CODES:
        return False
    return None if status is None else _retryable_from_status(status)


def classify_provider_fault(exc: BaseException) -> ProviderFault:
    """Describe the provider fault behind ``exc``. Never raises.

    The guarantee is enforced here rather than assumed of the input. Every
    caller runs inside an ``except`` block that is on its way to returning a
    503, so anything thrown from classification would replace that handled
    answer with an unhandled 500 -- losing the diagnosis at the exact moment it
    is being read. ``_safe_getattr`` covers the attribute reads specifically;
    this wrapper covers whatever else a foreign exception type can do, because
    "never throws" should not depend on having enumerated the ways it could.
    """
    try:
        return _classify_provider_fault(exc)
    except Exception:
        try:
            return ProviderFault(provider_class=type(exc).__name__)
        except Exception:
            return ProviderFault(provider_class="unknown")


def _classify_provider_fault(exc: BaseException) -> ProviderFault:
    """Describe the provider fault behind ``exc``.

    Always returns a value: when nothing in the chain is recognized, the result
    still names the innermost exception class, which is the one worth grouping
    on. ``retryable`` stays ``None`` in that case.
    """
    links = list(_chain(exc))
    # Two passes, not one. A single pass returning the first classifiable link
    # lets a shallow, lossy answer mask a precise one further down -- see
    # ``_classify_from_shape``.
    for link in links:
        classified = _classify_from_response(link)
        if classified is not None:
            return classified
    for link in links:
        classified = _classify_from_shape(link)
        if classified is not None:
            return classified
    return ProviderFault(provider_class=type(links[-1]).__name__)
