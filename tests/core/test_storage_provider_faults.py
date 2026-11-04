"""Classification of the provider fault behind a durable-storage failure.

#1467 put the ``__cause__`` chain into the logs; this pins the structured
fields that make a burst of those logs aggregatable. The cases below are built
to the real shapes the S3 path produces: s3fs never surfaces botocore's
``ClientError`` directly, it maps recognized codes onto ``OSError`` subclasses
and hangs the original off ``__cause__`` (``s3fs.errors.translate_boto_error``),
so every realistic fault has the code two links below the wrap.
"""

from __future__ import annotations

import errno

from xagent.core.file_storage.faults import ProviderFault, classify_provider_fault


class DurableStorageOperationError(RuntimeError):
    """Stands in for the real wrap; the classifier does not special-case it.

    Deliberately local rather than imported from ``web.services`` -- a core-layer
    test should not need a web-layer import to exercise a chain walk, and using
    the real class would imply the classifier recognises it, which it does not.
    """


def _client_error(code: str | None, status: int | None) -> Exception:
    """A botocore-shaped ``ClientError`` without requiring botocore."""
    error: dict[str, object] = {}
    if code is not None:
        error["Code"] = code
        error["Message"] = f"{code} occurred"
    response: dict[str, object] = {"Error": error}
    if status is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": status}

    class ClientError(Exception):
        def __init__(self) -> None:
            super().__init__(f"An error occurred ({code})")
            self.response = response

    return ClientError()


def _s3_chain(
    translated: BaseException, code: str | None, status: int | None
) -> DurableStorageOperationError:
    """Rebuild the real chain: wrap -> s3fs translation -> ClientError."""
    translated.__cause__ = _client_error(code, status)
    fault = DurableStorageOperationError(
        "Failed to write durable object: users/7/uploads/abc/report.txt"
    )
    fault.__cause__ = translated
    return fault


def test_throttle_two_links_down_is_found_and_marked_retryable() -> None:
    """The incident shape: SlowDown behind an s3fs OSError behind the wrap."""
    exc = _s3_chain(OSError(errno.EIO, "SlowDown occurred"), "SlowDown", 503)

    fault = classify_provider_fault(exc)

    assert fault.code == "SlowDown"
    assert fault.http_status == 503
    assert fault.retryable is True


def test_rejected_credential_is_marked_permanent() -> None:
    """AccessDenied arrives as a PermissionError; retrying can never help."""
    exc = _s3_chain(PermissionError("Access Denied"), "AccessDenied", 403)

    fault = classify_provider_fault(exc)

    assert fault.code == "AccessDenied"
    assert fault.retryable is False


def test_eio_alone_does_not_end_the_walk() -> None:
    """s3fs uses EIO for every code it cannot map, so EIO must not classify.

    Treating it as an answer would stop one link short of the ``ClientError``
    that actually names the fault -- here an unlisted code whose 503 still
    decides retryability.
    """
    exc = _s3_chain(OSError(errno.EIO, "unmapped"), "SomeNewS3Code", 503)

    fault = classify_provider_fault(exc)

    assert fault.code == "SomeNewS3Code"
    assert fault.retryable is True


def test_unlisted_code_without_a_status_stays_unclassified() -> None:
    """``None`` is not ``False``: unrecognized must not read as diagnosed."""
    exc = _s3_chain(OSError(errno.EIO, "mystery"), "SomeNewS3Code", None)

    fault = classify_provider_fault(exc)

    assert fault.code == "SomeNewS3Code"
    assert fault.retryable is None


def test_transport_failure_is_classified_by_class_name() -> None:
    """Pre-response botocore faults carry no ``.response`` to read."""

    class EndpointConnectionError(Exception):
        pass

    wrap = DurableStorageOperationError("Failed to sign durable object URL: k")
    wrap.__cause__ = EndpointConnectionError("Could not connect to the endpoint")

    fault = classify_provider_fault(wrap)

    assert fault.provider_class == "EndpointConnectionError"
    assert fault.retryable is True


def test_missing_credentials_is_permanent() -> None:
    class NoCredentialsError(Exception):
        pass

    wrap = DurableStorageOperationError("Failed to write durable object: k")
    wrap.__cause__ = NoCredentialsError("Unable to locate credentials")

    fault = classify_provider_fault(wrap)

    assert fault.retryable is False


def test_local_backend_errnos_split_by_whether_waiting_helps() -> None:
    """The filesystem backend fails as a plain OSError, with no provider code."""
    reset = DurableStorageOperationError("Failed to write durable object: k")
    reset.__cause__ = OSError(errno.ECONNRESET, "Connection reset by peer")
    assert classify_provider_fault(reset).retryable is True

    full = DurableStorageOperationError("Failed to write durable object: k")
    full.__cause__ = OSError(errno.ENOSPC, "No space left on device")
    # Permanent on purpose: retrying a full disk only amplifies load.
    assert classify_provider_fault(full).retryable is False
    # An OS errno lands in its own field, never mixed into the provider-code
    # vocabulary: "ENOSPC" and "SlowDown" come from different namespaces.
    assert classify_provider_fault(full).errno_name == "ENOSPC"
    assert classify_provider_fault(full).code is None


def test_an_ambiguous_status_says_unknown_rather_than_guessing() -> None:
    """A wrong verdict is worse than no verdict for this field.

    The old rule was "5xx except 501 is retryable, 4xx is not", which is a claim
    about statuses nobody has looked at. 505 and 508 are permanent despite being
    5xx, and an unrecognised 409 can be either -- a concurrent-operation
    conflict clears itself, ``BucketNotEmpty`` never does. Since the field
    exists to tell an operator whether waiting can help, guessing is the one
    thing it must not do.
    """
    for status in (409, 507, 599):
        exc = _s3_chain(OSError(errno.EIO, "x"), "SomeUnlistedCode", status)
        fault = classify_provider_fault(exc)
        assert fault.retryable is None, f"status {status} should not be guessed"
        assert fault.http_status == status

    for status in (505, 508):
        exc = _s3_chain(OSError(errno.EIO, "x"), "SomeUnlistedCode", status)
        assert classify_provider_fault(exc).retryable is False, status

    # Unambiguous 4xx keep their verdict -- dropping them would trade a wrong
    # answer for a missing one, which is not the trade being made here. The
    # list is deliberately wide: an earlier revision enumerated ten permanent
    # 4xx and let the other forty fall through to ``None``, discarding correct
    # verdicts while describing itself as conservative. 410 and 412 are the
    # ones that caught it -- 412 is S3's ``PreconditionFailed``.
    for status in (401, 402, 403, 404, 406, 410, 412, 413, 422, 428, 431, 451):
        exc = _s3_chain(OSError(errno.EIO, "x"), "SomeUnlistedCode", status)
        assert classify_provider_fault(exc).retryable is False, status

    # 409 is the sole ambiguous 4xx, and 429 must not be swept in with it:
    # it is unconditionally retryable, answered before the 4xx default.
    exc = _s3_chain(OSError(errno.EIO, "x"), "SomeUnlistedCode", 409)
    assert classify_provider_fault(exc).retryable is None
    exc = _s3_chain(OSError(errno.EIO, "x"), "SomeUnlistedCode", 429)
    assert classify_provider_fault(exc).retryable is True


def test_a_recognized_code_outranks_an_ambiguous_status() -> None:
    """``OperationAborted`` is a 409 that AWS documents as retryable."""
    exc = _s3_chain(OSError(errno.EIO, "x"), "OperationAborted", 409)
    assert classify_provider_fault(exc).retryable is True

    # ...while an unrecognised code at the same status stays unknown.
    exc = _s3_chain(OSError(errno.EIO, "x"), "BucketNotEmpty", 409)
    assert classify_provider_fault(exc).retryable is None


def test_http_status_decides_when_no_code_is_recognized() -> None:
    for status, expected in ((429, True), (500, True), (503, True), (501, False)):
        exc = _s3_chain(OSError(errno.EIO, "x"), None, status)
        fault = classify_provider_fault(exc)
        assert fault.retryable is expected, f"status {status}"
        assert fault.http_status == status


def test_an_unrecognizable_chain_still_names_the_innermost_class() -> None:
    """Always return something groupable, with retryability left unknown."""

    class WeirdBackendError(Exception):
        pass

    wrap = DurableStorageOperationError("Failed to write durable object: k")
    wrap.__cause__ = WeirdBackendError("no idea")

    fault = classify_provider_fault(wrap)

    assert fault == ProviderFault(provider_class="WeirdBackendError")
    assert fault.retryable is None


def test_a_cyclic_chain_terminates() -> None:
    """A chain can loop; the walk is bounded so classification cannot hang."""
    first = DurableStorageOperationError("first")
    second = DurableStorageOperationError("second")
    first.__cause__ = second
    second.__cause__ = first

    fault = classify_provider_fault(first)

    assert fault.retryable is None


def test_context_is_followed_when_from_was_omitted() -> None:
    """Not every fsspec implementation chains explicitly with ``from``."""
    wrap = DurableStorageOperationError("Failed to write durable object: k")
    try:
        try:
            raise OSError(errno.ECONNREFUSED, "Connection refused")
        except OSError:
            raise wrap
    except DurableStorageOperationError as raised:
        fault = classify_provider_fault(raised)

    assert fault.retryable is True
    assert fault.errno_name == "ECONNREFUSED"


def test_unknown_retryability_is_stated_rather_than_omitted() -> None:
    """``retryable=unknown`` must be visible, not silently absent.

    Dropping the field would make an unclassified fault indistinguishable from
    a log line written before classification existed -- and the whole point of
    the tri-state is that "unrecognised" reads differently from "permanent".
    """
    fields = ProviderFault(provider_class="WeirdBackendError").as_fields()

    assert fields["provider_class"] == "WeirdBackendError"
    assert fields["retryable"] == "unknown"
    # Unknown sub-fields stay None for the renderer to drop.
    assert fields["provider_code"] is None
    assert fields["provider_errno"] is None


def test_fields_keep_provider_and_os_vocabularies_apart() -> None:
    fields = ProviderFault(
        provider_class="ClientError", code="SlowDown", http_status=503, retryable=True
    ).as_fields()

    assert fields["provider_code"] == "SlowDown"
    assert fields["provider_errno"] is None
    assert fields["provider_http_status"] == 503
    assert fields["retryable"] is True


def test_a_suppressed_context_is_not_classified() -> None:
    """``raise ... from None`` hides the displaced exception; honour that.

    Classifying on a context the raiser deliberately suppressed would attribute
    the fault to an exception they judged unrelated -- and could report a
    confident ``retryable`` for something the storage layer never saw.
    """
    try:
        try:
            raise OSError(errno.ECONNREFUSED, "unrelated bookkeeping failure")
        except OSError:
            raise DurableStorageOperationError(
                "Failed to write durable object: k"
            ) from None
    except DurableStorageOperationError as raised:
        fault = classify_provider_fault(raised)

    assert fault.retryable is None
    assert fault.code is None
    assert fault.provider_class == "DurableStorageOperationError"


def test_a_non_dict_response_attribute_is_ignored() -> None:
    """``.response`` is not unique to botocore; requests-style objects have one.

    Reading it blindly would either crash on a non-subscriptable object or
    invent a code from an unrelated library's response.
    """

    class HttpLibError(Exception):
        def __init__(self) -> None:
            super().__init__("boom")
            self.response = object()

    wrap = DurableStorageOperationError("Failed to write durable object: k")
    inner = HttpLibError()
    inner.__cause__ = OSError(errno.ECONNRESET, "Connection reset by peer")
    wrap.__cause__ = inner

    fault = classify_provider_fault(wrap)

    # Fell through the unusable ``.response`` and kept descending.
    assert fault.errno_name == "ECONNRESET"
    assert fault.retryable is True


def test_a_non_string_code_cannot_crash_the_diagnostic_path() -> None:
    """An unhashable ``Code`` must not turn a handled 503 into an unhandled 500.

    This module duck-types botocore's shape so foreign backends still classify,
    which means a library following that shape loosely can put a dict or list
    under ``Code``. Feeding one to a ``frozenset`` lookup raises
    ``TypeError: unhashable type``, and because classification runs *inside* the
    caller's ``except`` block while it builds the 503, that TypeError would
    escape and replace the response -- losing the diagnosis exactly when an
    incident is being investigated.
    """
    for bad_code in ({"nested": "value"}, ["SlowDown"], object(), 503):
        exc = _s3_chain(OSError(errno.EIO, "x"), None, None)
        # Reach past the helper to plant a value botocore would never send.
        client_error = exc.__cause__.__cause__  # type: ignore[union-attr]
        client_error.response["Error"]["Code"] = bad_code  # type: ignore[attr-defined]

        fault = classify_provider_fault(exc)

        # Unusable code is dropped rather than stringified into the log field.
        assert fault.code is None, bad_code
        assert fault.retryable is None, bad_code


def test_a_non_string_code_still_defers_to_the_http_status() -> None:
    """Dropping an unusable code must not discard the status beside it."""
    exc = _s3_chain(OSError(errno.EIO, "x"), None, 503)
    client_error = exc.__cause__.__cause__  # type: ignore[union-attr]
    client_error.response["Error"]["Code"] = {"nested": "value"}  # type: ignore[attr-defined]

    fault = classify_provider_fault(exc)

    assert fault.code is None
    assert fault.http_status == 503
    assert fault.retryable is True


def test_an_empty_code_is_treated_as_absent() -> None:
    exc = _s3_chain(OSError(errno.EIO, "x"), "", 500)

    fault = classify_provider_fault(exc)

    assert fault.code is None
    assert fault.retryable is True


def test_response_outranks_a_coarser_errno_higher_in_the_chain() -> None:
    """s3fs collapses three distinct S3 codes onto one errno; recover them.

    ``translate_boto_error`` maps ``SlowDown``, ``ServiceUnavailable`` and
    ``OperationAborted`` all to ``OSError(EBUSY)``. Stopping at the first
    classifiable link therefore rendered them byte-for-byte identically as
    ``provider_errno=EBUSY`` -- destroying exactly the throttle-vs-outage
    distinction this module exists to draw. The deeper ``.response`` has to win
    regardless of position.

    Driven through the installed s3fs so the test tracks the backend's real
    mapping rather than a guess about it.
    """
    from s3fs.errors import translate_boto_error

    seen = {}
    for code, status in (
        ("SlowDown", 503),
        ("ServiceUnavailable", 503),
        ("OperationAborted", 409),
    ):
        wrap = DurableStorageOperationError(
            "Failed to write durable object: users/7/uploads/abc/report.txt"
        )
        wrap.__cause__ = translate_boto_error(_client_error(code, status))
        fault = classify_provider_fault(wrap)
        assert fault.code == code, f"{code} was reported as {fault.code}"
        assert fault.http_status == status
        seen[code] = fault.as_fields()

    # The whole point: three inputs, three distinguishable outputs.
    rendered = [tuple(sorted(fields.items())) for fields in seen.values()]
    assert len(set(rendered)) == 3


def test_errno_is_still_used_when_no_response_exists_anywhere() -> None:
    """Dropping errno precedence must not drop errno classification."""
    wrap = DurableStorageOperationError("Failed to write durable object: k")
    wrap.__cause__ = OSError(errno.ECONNRESET, "Connection reset by peer")

    fault = classify_provider_fault(wrap)

    assert fault.errno_name == "ECONNRESET"
    assert fault.retryable is True


def test_classification_never_raises_out_of_a_diagnostic_path() -> None:
    """A foreign exception must not be able to crash the classifier.

    Callers are inside an ``except`` block on their way to a 503, so anything
    thrown here replaces a handled answer with an unhandled 500 -- losing the
    diagnosis precisely when it is being read.
    """

    class HostileResponse(Exception):
        @property
        def response(self) -> object:
            raise RuntimeError("response property blew up")

    class HostileErrno(Exception):
        @property
        def errno(self) -> object:
            raise RuntimeError("errno property blew up")

    for hostile in (HostileResponse("x"), HostileErrno("x")):
        wrap = DurableStorageOperationError("Failed to write durable object: k")
        wrap.__cause__ = hostile

        fault = classify_provider_fault(wrap)

        assert fault.provider_class == type(hostile).__name__
        assert fault.retryable is None


def test_a_boolean_http_status_is_rejected() -> None:
    """``bool`` subclasses ``int``; it must not render as a status."""
    exc = _s3_chain(OSError(errno.EIO, "x"), "SomeNewS3Code", None)
    client_error = exc.__cause__.__cause__  # type: ignore[union-attr]
    client_error.response["ResponseMetadata"] = {"HTTPStatusCode": True}  # type: ignore[attr-defined]

    fault = classify_provider_fault(exc)

    assert fault.http_status is None


def test_clock_skew_is_permanent_and_conflict_is_retryable() -> None:
    """Two corrections to the code tables, pinned so they do not drift back.

    ``RequestTimeTooSkewed`` cannot resolve by retrying -- the clock has to be
    fixed -- while ``OperationAborted`` is documented by AWS as a conflicting
    concurrent operation to try again.
    """
    skewed = _s3_chain(OSError(errno.EIO, "x"), "RequestTimeTooSkewed", 403)
    assert classify_provider_fault(skewed).retryable is False

    aborted = _s3_chain(OSError(errno.EIO, "x"), "OperationAborted", 409)
    assert classify_provider_fault(aborted).retryable is True
