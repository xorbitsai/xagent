"""Every durable-storage fault must reach the log with its provider cause.

The envelopes these paths return are deliberately detail-free, and neither
FastAPI's ``HTTPException`` handler nor the ``/v1/*`` error handler logs a
traceback for the exception it translates -- so the log line is the *only*
record of what actually failed. ``ManagedFileRef`` wraps provider faults into
``DurableStorageOperationError`` carrying just the storage key, which means a
log line without ``exc_info`` leaves an operator unable to tell a throttle from
a timeout from rejected credentials. That is the gap that blocked an incident
investigation in #1467.
"""

from __future__ import annotations

import ast
import io
import logging
from pathlib import Path
from typing import Any, Iterator, cast

import pytest
from fastapi import HTTPException
from fastapi.datastructures import UploadFile

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.api import files as files_api
from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.api.v1.errors import V1ApiError
from xagent.web.models.user import User
from xagent.web.services.managed_file_ref import (
    _MAX_LOG_VALUE_LENGTH,
    DURABLE_FAULT_LOG_PREFIX,
    DurableStorageOperationError,
    ManagedFileRef,
    log_durable_storage_fault,
)

from .conftest import _direct_db_session, _setup_admin

# Not module-wide: only the end-to-end upload test touches the database. The
# other tests drive the helper directly and would pay a schema create/drop for
# nothing.


@pytest.fixture()
def isolated_upload_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Path]:
    """Point staging and the object store at ``tmp_path``.

    Kept local rather than shared with ``test_upload_connection_boundary``:
    this suite's conftest deliberately exposes helpers by explicit import and
    keeps fixtures out of it, and no fixture-sharing module exists for the
    upload paths yet. The durable write is patched to fail before it reaches
    the object store, but the store is still redirected so a regression that
    lets the write through cannot touch a real backend.
    """
    upload_root = tmp_path / "uploads"
    object_root = tmp_path / "objects"
    upload_root.mkdir()
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()
    monkeypatch.setattr(files_api, "get_uploads_dir", lambda: upload_root)
    try:
        yield upload_root
    finally:
        get_unscoped_file_storage.cache_clear()


def _stage_under(root: Path, user_id: int):
    """A deterministic staging-path chooser scoped to ``root``."""

    def get_upload_path(
        filename: str,
        task_id: str | None,
        folder: str | None,
        requested_user_id: int,
    ) -> Path:
        del task_id, folder
        assert requested_user_id == user_id
        path = root / f"user_{user_id}" / Path(filename).name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    return get_upload_path


_FILENAME = "quarterly-report.txt"
_STORAGE_KEY = f"users/7/uploads/8ac1f2/{_FILENAME}"
_PROVIDER_MESSAGE = "SlowDown: Please reduce your request rate (status 503)"
_UNAVAILABLE_DETAIL = "Durable storage is temporarily unavailable"


class _ProviderThrottled(RuntimeError):
    """Stand-in for the boto/S3 error class the wrap discards."""


def _wrapped_fault() -> DurableStorageOperationError:
    """Build a fault shaped exactly like ``ManagedFileRef``'s wraps.

    The key rides on ``storage_key``, not in the message, because ``str(exc)``
    escapes to places the raise site does not control. Keeping this replica in
    the production shape is the point: with the key in the message it would test
    a shape nothing raises, and the log assertions would pass for the wrong
    reason -- from the message text rather than from the rendered field.

    Everything an operator needs to classify the failure lives in ``__cause__``.
    Assigning it directly is what ``raise ... from exc`` does, and it survives a
    later bare ``raise`` of this object.
    """
    fault = DurableStorageOperationError(
        "Failed to write durable object", storage_key=_STORAGE_KEY
    )
    fault.__cause__ = _ProviderThrottled(_PROVIDER_MESSAGE)
    return fault


def _rendered(record: logging.LogRecord) -> str:
    """Render a record the way a handler would, ``exc_info`` included."""
    return logging.Formatter("%(message)s").format(record)


def _warnings(caplog: pytest.LogCaptureFixture, logger_name: str) -> list[str]:
    return [
        _rendered(record)
        for record in caplog.records
        if record.name == logger_name and record.levelno == logging.WARNING
    ]


def _sole_warning(caplog: pytest.LogCaptureFixture, logger_name: str) -> str:
    """The one warning a direct helper call may emit."""
    rendered = _warnings(caplog, logger_name)
    assert len(rendered) == 1, f"expected exactly one warning, got {rendered}"
    return rendered[0]


def _warning_matching(
    caplog: pytest.LogCaptureFixture, logger_name: str, needle: str
) -> str:
    """Pick the fault line out of an endpoint's warnings.

    Not ``_sole_warning``: the request paths under test also run best-effort
    cleanup that logs to this same logger when it cannot remove a staged file
    (``_delete_staged_upload``), and an unrelated second warning must not read
    as a failure of the line this test is about.
    """
    matches = [line for line in _warnings(caplog, logger_name) if needle in line]
    assert len(matches) == 1, f"expected one warning matching {needle!r}: {matches}"
    return matches[0]


def _assert_cause_chain_recorded(rendered: str, *, wrap_key: bool = True) -> None:
    """The provider fault -- class and message -- must be in the log text.

    ``wrap_key=False`` where the wrap's own storage key should not appear: the
    delete path hands over a raw provider exception that has none, and a caller
    passing an explicit ``storage_key`` field deliberately overrides it.
    """
    assert _ProviderThrottled.__name__ in rendered
    assert _PROVIDER_MESSAGE in rendered
    if wrap_key:
        # The key is the anchor an operator greps for. It is no longer in the
        # message -- ``str(exc)`` escapes to clients -- so this asserts the
        # helper renders it from the exception instead of losing it.
        assert f"storage_key={_STORAGE_KEY}" in rendered


def test_durable_storage_unavailable_logs_cause_and_keeps_body_detail_free(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The shared 503 helper is where all nine file-API sites get their log.

    Asserting on the helper rather than on each endpoint is deliberate: the
    exception is a required positional argument and the helper is ``NoReturn``,
    so a call site can neither skip the cause nor log without raising. That
    leaves the helper's own body as the only place the chain can be dropped.
    """
    fault = _wrapped_fault()

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException) as raised:
            files_api._raise_durable_storage_unavailable(fault, "download")

    assert raised.value.status_code == 503
    # Scope segments in the key can encode end-user identity, so the body
    # stays the fixed message -- the detail is server-side only.
    assert raised.value.detail == _UNAVAILABLE_DETAIL
    assert _STORAGE_KEY not in str(raised.value.detail)
    assert _PROVIDER_MESSAGE not in str(raised.value.detail)
    assert raised.value.__cause__ is fault

    rendered = _sole_warning(caplog, files_api.logger.name)
    assert "Durable storage unavailable during download" in rendered
    _assert_cause_chain_recorded(rendered)


def test_durable_storage_unavailable_accepts_an_unwrapped_provider_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The delete path hands over the raw provider exception, not a wrap.

    ``delete_file`` catches ``Exception`` around the durable cleanup rather
    than a ``DurableStorageOperationError``, so the helper has to log something
    that was never wrapped by ``ManagedFileRef``. Before #1467 that site logged
    the storage key and discarded the exception entirely.
    """
    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException) as raised:
            files_api._raise_durable_storage_unavailable(
                _ProviderThrottled(_PROVIDER_MESSAGE),
                "durable cleanup before row delete",
                storage_key=_STORAGE_KEY,
            )

    assert raised.value.status_code == 503
    rendered = _sole_warning(caplog, files_api.logger.name)
    assert "durable cleanup before row delete" in rendered
    # The key is a named field, not part of the bounded operation label.
    assert f"storage_key={_STORAGE_KEY}" in rendered
    _assert_cause_chain_recorded(rendered, wrap_key=False)


def test_one_wrap_yields_one_record_however_many_arms_report_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A fault crossing several handlers is recorded once.

    Two arms legitimately report the same fault -- the request-scoped one that
    answers the client, and the endpoint-scoped one that catches whatever
    escaped -- and neither can know whether the other ran. So the wrap carries
    the fact that it has been logged. Without this, sustained-outage logs
    duplicate every fault, and the duplicate looks like a second failure.
    """
    fault = _wrapped_fault()

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        log_durable_storage_fault(
            files_api.logger, "turn preparation", fault, task_id=42
        )
        log_durable_storage_fault(
            files_api.logger, "endpoint recovery", fault, task_id=42
        )

    # The first arm to report wins, which is the innermost -- the one that knows
    # what it was doing. The coarser endpoint label is the one dropped.
    rendered = _sole_warning(caplog, files_api.logger.name)
    assert "during turn preparation" in rendered
    _assert_cause_chain_recorded(rendered)


def test_an_unwrapped_provider_error_is_not_marked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dedup is scoped to this module's wraps, and only they carry the mark.

    Setting a private attribute on a foreign exception is not safe in principle
    -- it may reject the write, and reading it back is not guaranteed either --
    and it buys nothing: an unwrapped provider error reaches exactly one
    reporting site, so there is nothing to deduplicate.
    """
    raw = _ProviderThrottled(_PROVIDER_MESSAGE)

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        log_durable_storage_fault(files_api.logger, "download", raw)
        log_durable_storage_fault(files_api.logger, "preview", raw)

    assert len(_warnings(caplog, files_api.logger.name)) == 2
    assert not hasattr(raw, "_durable_fault_logged")


@pytest.mark.asyncio
@pytest.mark.usefixtures("_test_db")
async def test_upload_durable_write_failure_logs_the_provider_cause(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    isolated_upload_storage: Path,
) -> None:
    """End to end on the incident path: a failed durable write, one 503, one log.

    ``store_uploaded_files`` backs ``/api/files/upload`` and both public-chat
    upload entry points (share and widget), so this covers every route where
    the reported 503 bursts were observed.
    """
    upload_root = isolated_upload_storage
    # Side effect only: lays down the admin row the upload is attributed to.
    _setup_admin()
    db = _direct_db_session()
    try:
        user_id = int(db.query(User.id).filter(User.username == "admin").scalar())
    finally:
        db.close()
    monkeypatch.setattr(
        files_api, "get_upload_path", _stage_under(upload_root, user_id)
    )

    def fail_sync(_self: ManagedFileRef, *_args: Any, **_kwargs: Any) -> None:
        raise _wrapped_fault()

    monkeypatch.setattr(ManagedFileRef, "sync_to_durable", fail_sync)
    upload = UploadFile(
        filename=_FILENAME,
        file=io.BytesIO(b"payload"),
        headers={"content-type": "text/plain"},
    )

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException) as raised:
            await files_api.store_uploaded_files(
                upload_items=[upload],
                task_type="general",
                task_id=None,
                folder=None,
                user_id=user_id,
                single_file_mode=True,
            )

    assert raised.value.status_code == 503
    assert raised.value.detail == _UNAVAILABLE_DETAIL
    # The cause chain still reaches the client-facing exception too.
    assert isinstance(raised.value.__cause__, DurableStorageOperationError)

    rendered = _warning_matching(
        caplog, files_api.logger.name, "Durable storage unavailable during upload"
    )
    _assert_cause_chain_recorded(rendered)


def test_v1_turn_attachment_durable_fault_logs_the_provider_cause(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The SDK path returns its own 503 envelope and needs its own log line."""

    def fail_resolve(**_kwargs: Any) -> None:
        raise _wrapped_fault()

    monkeypatch.setattr(v1_tasks, "resolve_turn_file_infos", fail_resolve)

    with caplog.at_level(logging.WARNING, logger=v1_tasks.logger.name):
        with pytest.raises(V1ApiError) as raised:
            v1_tasks._resolve_turn_files_or_400(
                file_ids=["8ac1f2"],
                owner_user_id=7,
                db=cast(Any, None),
                task_id=42,
            )

    assert raised.value.http_status == 503
    assert _STORAGE_KEY not in raised.value.message
    assert _PROVIDER_MESSAGE not in raised.value.message

    rendered = _warning_matching(
        caplog, v1_tasks.logger.name, "during turn attachment resolution"
    )
    assert "task_id=42" in rendered
    # The create path has task_id=None, so these carry identification there.
    assert "owner_user_id=7" in rendered
    assert "file_ids=8ac1f2" in rendered
    _assert_cause_chain_recorded(rendered)


def test_v1_turn_attachment_integrity_fault_is_not_reported_as_an_outage(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The integrity subclass through the real cascade, not just the parent.

    The test above injects only ``DurableStorageOperationError``, so it would
    pass with the two arms swapped -- the parent would catch the subclass and
    this site would report permanent corruption as a retryable outage. That is
    the safety property the ordering exists for, and injecting the subclass is
    what actually exercises it.
    """
    from xagent.web.services.managed_file_ref import (
        FILE_INTEGRITY_REUPLOAD_MESSAGE,
        DurableObjectIntegrityError,
    )

    def fail_resolve(**_kwargs: Any) -> None:
        raise DurableObjectIntegrityError(
            FILE_INTEGRITY_REUPLOAD_MESSAGE,
            storage_key="users/7/uploads/8ac1f2/corrupt.txt",
        )

    monkeypatch.setattr(v1_tasks, "resolve_turn_file_infos", fail_resolve)

    with caplog.at_level(logging.WARNING, logger=v1_tasks.logger.name):
        with pytest.raises(V1ApiError) as raised:
            v1_tasks._resolve_turn_files_or_400(
                file_ids=["8ac1f2"],
                owner_user_id=7,
                db=cast(Any, None),
                task_id=42,
            )

    assert raised.value.http_status == 503
    # The envelope is deliberately unchanged; what must not happen is a second,
    # contradicting record calling permanent corruption a transient outage.
    assert not [
        line
        for line in _warnings(caplog, v1_tasks.logger.name)
        if DURABLE_FAULT_LOG_PREFIX in line
    ], "an integrity fault emitted an outage warning -- the arms are misordered"


# Every ``_raise_durable_storage_unavailable`` call site in files.py, with the
# fields it is expected to carry. N2 in review -- the signed-redirect site
# shipping with no identifier -- was invisible because only two sites had
# assertions; this sweep is what makes the set itself the contract.
_FAULT_SITES = (
    ("signed durable redirect", ("file_id",)),
    ("upload", ("user_id", "task_id")),
    ("download", ("file_id",)),
    ("preview", ("file_id",)),
    ("pptx preview", ("file_id",)),
    ("public download", ("file_id",)),
    ("public preview", ("file_id",)),
    ("public preview task asset", ("file_id",)),
    ("durable cleanup before row delete", ("file_id", "storage_key")),
)


# --- shared AST primitives -------------------------------------------------
#
# Several contracts in this file can only be checked against source: which
# call sites exist, what they bind, which arms come first, and whether a
# handler re-raises. Each started as its own hand-rolled walk; these are the
# pieces they share, and the checks themselves stay separate because they
# assert different things.


def _module_ast(module: Any) -> ast.Module:
    """Parse an imported module's own source."""
    source_path = Path(module.__file__)
    return ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))


def _callee_name(call: ast.Call) -> str | None:
    """The function name a call targets, bare or qualified."""
    return _exception_name(call.func)


def _operation_label_node(call: ast.Call) -> ast.expr | None:
    """The expression a call passes as ``operation``, positional or keyword."""
    if len(call.args) > 1:
        return call.args[1]
    return next((kw.value for kw in call.keywords if kw.arg == "operation"), None)


def _operation_label_of(call: ast.Call, where: str) -> str:
    """The call's operation label, which must be a string literal.

    Positional or ``operation=`` keyword -- a site refactored to keyword form
    must fail with "not a literal", never silently drop out of a drift
    comparison or die on an IndexError.
    """
    node = _operation_label_node(call)
    assert isinstance(node, ast.Constant) and isinstance(node.value, str), (
        f"{where}:{call.lineno}: the operation label must be a string "
        f"literal (positional or operation=); got {ast.unparse(call)[:90]!r}"
    )
    return node.value


def _package_modules(package_root: Path) -> list[Path]:
    """Every module the repo-wide sweeps below walk, in a stable order."""
    return sorted(package_root.rglob("*.py"))


def _parse_module_file(path: Path) -> ast.Module:
    """Parse a source file, naming it if it cannot be read or parsed.

    The sweeps walk the whole package, so an unrelated unparsable or
    non-UTF-8 file would otherwise fail them with a traceback that names
    neither the file nor why this test touched it.
    """
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise AssertionError(
            f"{path} could not be parsed while sweeping the package for "
            f"durable-fault sites ({exc.__class__.__name__}: {exc}). This is "
            "not a durable-storage failure -- fix or exclude that file."
        ) from exc


def _exception_name(node: ast.expr) -> str | None:
    """The class name an ``except`` operand refers to, bare or qualified.

    ``ast.Attribute`` counts: a module refactored to ``except mfr.Durable...``
    would otherwise drop out of every sweep below with nothing failing.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _handler_types(handler: ast.ExceptHandler) -> list[str]:
    """The exception class names an ``except`` arm names, tuple or not."""
    if handler.type is None:
        return []
    if isinstance(handler.type, ast.Tuple):
        return [
            name
            for name in (_exception_name(element) for element in handler.type.elts)
            if name is not None
        ]
    name = _exception_name(handler.type)
    return [name] if name is not None else []


def _durable_arm_pairs(tree: ast.Module) -> list[ast.Try]:
    """Every ``try`` that handles both the integrity subclass and its parent."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            "DurableObjectIntegrityError" in _handler_types(h) for h in node.handlers
        )
        and any(
            "DurableStorageOperationError" in _handler_types(h) for h in node.handlers
        )
    ]


def _identifiers(expression: ast.expr) -> set[str]:
    """Every name and attribute the expression reads.

    ``file_ref.record.file_id`` yields ``file_ref``, ``record`` and ``file_id``,
    so a field may be bound to a bare variable or reached through attributes.
    """
    found: set[str] = set()
    for node in ast.walk(expression):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return found


def _binds_a_matching_name(field: str, expression: ast.expr) -> bool:
    """Whether the expression reads an identifier plausibly holding ``field``.

    A qualifier prefix is allowed -- ``task_id=parsed_task_id`` and
    ``user_id=owner_user_id`` name the thing they carry -- while an unrelated
    name is rejected, which is what catches ``file_id=storage_key``.

    The prefix must end at an underscore. A bare suffix test would admit
    ``file_id=profile_id``, since ``"profile_id".endswith("file_id")`` -- an
    unrelated identifier passing on a coincidental substring, which is the
    opposite of the point.

    **This checks spelling, not referent, and that is a real limit.** The
    "public preview task asset" site shipped with ``file_id=file_id`` where the
    failing object was ``asset_record`` -- a different row from the route's
    ``file_id``, so the one line meant to identify the failure named an object
    that was fine. This check passed it, because the spelling was right. Only a
    test that drives the endpoint distinguishes those, which is #1522; do not
    read a green run here as "every site logs the right object".
    """
    return any(
        identifier == field or identifier.endswith(f"_{field}")
        for identifier in _identifiers(expression)
    )


@pytest.mark.parametrize(
    ("field", "source", "accepted"),
    [
        ("file_id", "file_id", True),
        ("file_id", "file_ref.record.file_id", True),
        ("task_id", "parsed_task_id", True),
        ("user_id", "owner_user_id", True),
        ("file_id", "storage_key", False),
        # A coincidental suffix, not a qualified name: the underscore boundary
        # is the whole difference, and without it this passes.
        ("file_id", "profile_id", False),
    ],
)
def test_the_binding_check_accepts_qualified_names_and_rejects_unrelated_ones(
    field: str, source: str, accepted: bool
) -> None:
    """The rule the site sweep leans on, pinned on its own.

    Asserted here rather than only through the sweep because the sweep can only
    fail on the sites that exist: a rule too loose to reject anything would
    still pass it. ``profile_id`` is the case that made this necessary.
    """
    expression = ast.parse(source, mode="eval").body
    assert _binds_a_matching_name(field, expression) is accepted


def test_every_fault_site_label_is_bounded_and_reaches_the_log() -> None:
    """The nine labels are a closed set of bounded, aggregatable values.

    ``upload`` carries no ``file_id`` by design -- it is a batch-registration
    path, and any file in the batch may be the one that failed -- but it does
    carry tenant and task, which is what correlates a 503 burst. Every other
    site identifies its subject.
    """
    tree = _module_ast(files_api)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_raise_durable_storage_unavailable"
    ]
    # Parsed rather than grepped: a substring count also matches the ``def``
    # line, and a label is only a contract if the count is exact.
    assert len(calls) == len(_FAULT_SITES), (
        f"{len(calls)} call sites but {len(_FAULT_SITES)} labels declared -- "
        "add the new site to _FAULT_SITES with the fields it should carry"
    )

    declared = {label for label, _ in _FAULT_SITES}

    # Every label is a plain literal, so it stays a bounded, aggregatable value
    # -- an f-string here is how the storage key first became part of a label.
    passed = {_operation_label_of(node, "files.py") for node in calls}
    assert passed == declared, f"labels drifted: {passed ^ declared}"

    for node in calls:
        label = _operation_label_of(node, "files.py")
        expected = dict(_FAULT_SITES)[label]
        field_keywords = [kw for kw in node.keywords if kw.arg != "operation"]
        # Set comparison: the field names are the contract, their order is not.
        assert {kw.arg for kw in field_keywords} == set(expected), (
            f"site {label!r} passes "
            f"{sorted(str(kw.arg) for kw in field_keywords)}, "
            f"expected {sorted(expected)}"
        )
        # Names alone are not the contract. ``file_id=storage_key`` declares the
        # right field and renders the wrong value, which is invisible both here
        # and to the sweep below -- that one supplies its own placeholder values
        # and never reads the call site. Requiring the bound expression to read
        # something of the same name is what ties the label to the variable.
        for keyword in field_keywords:
            assert keyword.arg is not None, f"site {label!r} uses **kwargs"
            assert _binds_a_matching_name(keyword.arg, keyword.value), (
                f"site {label!r} binds {keyword.arg}= to "
                f"{ast.unparse(keyword.value)!r}, which never reads "
                f"{keyword.arg} -- the field name and the value must agree"
            )


# The counts are part of the contract: the walker can only check the pairs it
# finds, so a deleted integrity arm removes its pair from sight rather than
# failing the ordering assert. Pinning how many pairs each module carries is
# what turns that silent disappearance into a failure. (A ``try`` with only
# the parent arm is legitimately not a pair -- ``store_uploaded_files`` is one,
# where the subclass is unreachable -- which is why the count must be declared,
# not derived.)
_MODULES_WITH_DURABLE_ARM_PAIRS = (
    ("web/api/files.py", 7, lambda: files_api),
    ("web/api/v1/tasks.py", 1, lambda: v1_tasks),
    (
        "core/tools/adapters/vibe/file_ingestion_tool.py",
        1,
        lambda: _file_ingestion_tool(),
    ),
)


# Every direct ``log_durable_storage_fault`` call in the tree, by module and
# bounded label. "Direct" means not through files.py's
# ``_raise_durable_storage_unavailable`` wrapper, whose own forwarding call
# passes its caller's label along as a parameter -- the one site whose label
# is legitimately not a literal, pinned by the files.py sweep instead.
_DIRECT_HELPER_SITES = frozenset(
    {
        ("web/api/files.py", "upload compensation"),
        ("web/api/v1/tasks.py", "turn attachment resolution"),
        (
            "core/tools/adapters/vibe/file_ingestion_tool.py",
            "knowledge-base file restore",
        ),
    }
)

# Labels that are legitimately not string literals, by module and the name
# they are passed as, each with what keeps the value bounded. A label must be
# a literal *or* appear here -- "not a literal" alone is not a licence, or a
# future site passing an unbounded variable would be waved through as if it
# were one of these.
_BOUNDED_NON_LITERAL_LABELS = {
    # The wrapper forwarding its caller's label; the literals live at its
    # callers, where the files.py sweep above checks them.
    ("web/api/files.py", "operation"),
}


def test_every_direct_helper_site_is_declared_with_a_bounded_label() -> None:
    """Discovered, not declared: a new direct reporting site cannot go unswept.

    The files.py sweep only sees calls through the ``NoReturn`` wrapper, so a
    site calling the shared helper directly is outside it. Previously this was
    a hardcoded pair of modules, which could only check the sites someone
    remembered to list -- a third would have gone unscanned. Now the tree is
    walked, and every direct call must carry a string-literal label declared
    above, so an f-string label (the leak-and-cardinality class this exists to
    prevent) fails wherever it appears.
    """
    package_root = Path(files_api.__file__).parents[2]
    found: set[tuple[str, str]] = set()
    forwarding: list[str] = []
    for path in _package_modules(package_root):
        module_label = path.relative_to(package_root).as_posix()
        tree = _parse_module_file(path)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and _callee_name(node) == "log_durable_storage_fault"
            ):
                continue
            label_node = _operation_label_node(node)
            if isinstance(label_node, ast.Constant) and isinstance(
                label_node.value, str
            ):
                # Only the reporting sites: the wrapper below forwards its
                # caller's ``**fields`` by design, and its callers' field names
                # are what the files.py sweep checks.
                assert all(kw.arg is not None for kw in node.keywords), (
                    f"{module_label}:{node.lineno} passes **kwargs to the "
                    "helper, so its field names are not checkable here"
                )
                found.add((module_label, label_node.value))
            elif (
                isinstance(label_node, ast.Name)
                and (module_label, label_node.id) in _BOUNDED_NON_LITERAL_LABELS
            ):
                # Declared above with the mechanism that bounds it.
                forwarding.append(f"{module_label}:{label_node.id}")
            else:
                raise AssertionError(
                    f"{module_label}:{node.lineno} passes a label that is "
                    "neither a string literal nor a declared bounded variable: "
                    f"{ast.unparse(node)[:90]!r}. A bounded literal keeps the "
                    "label aggregatable; add it to _DIRECT_HELPER_SITES, or -- "
                    "if it is a variable something else bounds -- to "
                    "_BOUNDED_NON_LITERAL_LABELS with that reason."
                )

    assert found == set(_DIRECT_HELPER_SITES), (
        f"direct helper sites changed: {found ^ set(_DIRECT_HELPER_SITES)} -- "
        "declare the new site with its bounded label, or fix a label that "
        "stopped being a string literal"
    )
    assert set(forwarding) == {
        f"{module}:{name}" for module, name in _BOUNDED_NON_LITERAL_LABELS
    }, (
        "the declared bounded-variable labels and the ones actually passed "
        f"disagree: {sorted(forwarding)} vs {sorted(_BOUNDED_NON_LITERAL_LABELS)}"
    )


def test_every_module_with_an_arm_pair_is_in_the_table() -> None:
    """The table above is discovered, not merely declared.

    A closed list can only check the modules someone remembered to add: a new
    module introducing a misordered pair elsewhere would be invisible to the
    per-module sweep. This walks every module under ``src/xagent`` and requires
    the set of modules carrying an integrity/parent pair to equal the table, so
    a new pair anywhere fails here with directions instead of going unchecked.
    """
    package_root = Path(files_api.__file__).parents[2]
    found = set()
    for path in _package_modules(package_root):
        if _durable_arm_pairs(_parse_module_file(path)):
            found.add(path.relative_to(package_root).as_posix())

    declared = {label for label, _, _ in _MODULES_WITH_DURABLE_ARM_PAIRS}
    assert found == declared, (
        f"modules with integrity/parent arm pairs changed: {found ^ declared} "
        "-- add the module to _MODULES_WITH_DURABLE_ARM_PAIRS with its pair "
        "count so its ordering is swept, or remove the stale row"
    )


def _file_ingestion_tool() -> Any:
    from xagent.core.tools.adapters.vibe import file_ingestion_tool

    return file_ingestion_tool


@pytest.mark.parametrize(
    ("label", "expected", "loader"), _MODULES_WITH_DURABLE_ARM_PAIRS
)
def test_the_integrity_arm_precedes_its_parent_at_every_site(
    label: str, expected: int, loader: Any
) -> None:
    """Checked at all nine pairs, because three of them have no other guard.

    Measured, not estimated -- an earlier version of this docstring claimed
    seven of the nine were unguarded, which was more than twice the truth.
    Neutralising each integrity arm in turn and running the suite shows six
    pairs are caught behaviourally: ``_durable_redirect_response``,
    ``download_file``, ``preview_file`` and the first ``public_preview_file``
    pair (by the five ``*_checksum_mismatch_asks_user_to_reupload`` tests),
    plus ``v1/tasks.py`` and ``file_ingestion_tool`` (by the real-path
    injections in this file and ``test_kb_creation_tools``).

    The three where a swapped or deleted arm changes behaviour with no
    behavioural test failing are ``preview_pptx_as_pdf``,
    ``public_download_file``, and the second ``public_preview_file`` pair --
    the task-asset one. Those are what this check is load-bearing for; giving
    them real end-to-end coverage is #1522.

    This does not replace the behavioural tests: it proves ordering, not that
    each arm answers correctly. What it adds is that the one regression which
    breaks the property at every site at once cannot land anywhere.
    """
    tree = _module_ast(loader())
    pairs = _durable_arm_pairs(tree)
    assert len(pairs) == expected, (
        f"{label}: expected {expected} integrity/parent pair(s), found "
        f"{len(pairs)} at lines {sorted(p.lineno for p in pairs)} -- a pair was "
        "added or removed. Update _MODULES_WITH_DURABLE_ARM_PAIRS and give the "
        "changed site its coverage story; a deleted integrity arm is invisible "
        "to the ordering check below, which is what this count exists to catch."
    )

    for node in pairs:
        integrity_at = min(
            handler.lineno
            for handler in node.handlers
            if "DurableObjectIntegrityError" in _handler_types(handler)
        )
        parent_at = min(
            handler.lineno
            for handler in node.handlers
            if "DurableStorageOperationError" in _handler_types(handler)
        )
        assert integrity_at < parent_at, (
            f"{label}: the try at line {node.lineno} catches "
            f"DurableStorageOperationError (line {parent_at}) before its "
            f"subclass DurableObjectIntegrityError (line {integrity_at}), which "
            "makes the integrity arm dead and reports corruption as an outage"
        )


@pytest.mark.parametrize(("label", "expected_fields"), _FAULT_SITES)
def test_fault_site_renders_its_label_and_fields(
    caplog: pytest.LogCaptureFixture,
    label: str,
    expected_fields: tuple[str, ...],
) -> None:
    """The helper renders each declared label and field set into the line.

    Helper rendering only, with synthetic values: this never executes a real
    ``files.py`` call site. What ties each site to its declared label, fields,
    and bindings is the AST sweep above; what this pins is that the declared
    shape, once passed, survives into the rendered record.
    """
    fields = {name: f"value-for-{name}" for name in expected_fields}

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException) as raised:
            files_api._raise_durable_storage_unavailable(
                _wrapped_fault(), label, **fields
            )

    assert raised.value.status_code == 503
    rendered = _sole_warning(caplog, files_api.logger.name)
    assert f"during {label}" in rendered
    for name in expected_fields:
        assert f"{name}=value-for-{name}" in rendered
    # A site that names its own storage_key overrides the wrap's, so the wrap's
    # key is legitimately absent there -- that precedence is the point.
    _assert_cause_chain_recorded(
        rendered, wrap_key="storage_key" not in expected_fields
    )


def test_field_values_cannot_forge_a_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A newline in a client-supplied field must not start a new record.

    ``/v1/*`` renders the request's ``files`` list, an unvalidated ``list[str]``,
    so a caller cannot be relied on to have checked (CWE-117).
    """
    forged = "ok,\n2026-01-01 00:00:00 ERROR    xagent.web - FABRICATED entry"

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException):
            files_api._raise_durable_storage_unavailable(
                _wrapped_fault(), "upload", file_ids=forged
            )

    rendered = _sole_warning(caplog, files_api.logger.name)
    message_line = rendered.splitlines()[0]
    assert "FABRICATED" in message_line, "the value must survive, escaped"
    assert "\\n" in message_line
    assert not any(line.startswith("2026-01-01") for line in rendered.splitlines()), (
        "the injected text must not stand as its own record"
    )


def test_a_field_value_cannot_forge_a_second_field_on_the_same_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Whitespace, not a newline, is the cheaper forgery -- close that too.

    Escaping line breaks stops a value fabricating a whole *record*, and
    nothing stopped it fabricating a *field*: rendered with its spaces intact,
    a client-supplied ``abc user_id=999`` put two more ``key=value`` pairs on
    the line, and an operator -- or anything parsing it -- would attribute the
    fault to another tenant and another file.

    Asserted through the tokenizer a consumer actually uses, not against the
    escaping this file happens to do. The first attempt at this quoted the
    value instead, and asserted by stripping the quoted substring before
    looking -- which proved the quoting existed, not that anything downstream
    was safe. It was not: a whitespace tokenizer splits ``file_ids="abc`` from
    ``user_id=999`` and reads the forgery as real, because nothing obliges a
    log reader to honour quotes.
    """
    forged = "abc user_id=999 file_id=someone-elses"

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException):
            files_api._raise_durable_storage_unavailable(
                _wrapped_fault(), "upload", file_ids=forged
            )

    message_line = _sole_warning(caplog, files_api.logger.name).splitlines()[0]

    # The consumer: split on whitespace, then on the first ``=``. This is what
    # awk, a kv filter, and a grep pipeline all do.
    fields = dict(token.split("=", 1) for token in message_line.split() if "=" in token)
    assert set(fields) == {"file_ids", "storage_key"}, fields
    assert "user_id" not in fields and "file_id" not in fields
    # The value survives whole in the one field it belongs to, escaped.
    assert fields["file_ids"].replace("\\x20", " ") == forged
    # Fields the helper renders itself stay untouched and greppable.
    assert fields["storage_key"] == _STORAGE_KEY


def test_the_exported_prefix_is_the_one_the_helper_emits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The constant the other suites match on must track the real message.

    Four negative assertions elsewhere prove an integrity fault emitted *no*
    outage line by looking for this text. They previously inlined it, so a
    reworded message would have left them passing against text that no longer
    exists. Importing the constant only moves that risk unless the constant
    itself is pinned to what the helper emits -- which is what this does.
    """
    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        log_durable_storage_fault(files_api.logger, "download", _wrapped_fault())

    rendered = _sole_warning(caplog, files_api.logger.name).splitlines()[0]
    assert rendered.startswith(f"{DURABLE_FAULT_LOG_PREFIX} download"), rendered


@pytest.mark.parametrize("length", [_MAX_LOG_VALUE_LENGTH - 1, _MAX_LOG_VALUE_LENGTH])
def test_a_field_value_at_or_under_the_bound_is_kept_whole(
    caplog: pytest.LogCaptureFixture, length: int
) -> None:
    """The bound is inclusive, so neither of these may be marked truncated."""
    value = "v" * length

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException):
            files_api._raise_durable_storage_unavailable(
                _wrapped_fault(), "upload", file_ids=value
            )

    rendered = _sole_warning(caplog, files_api.logger.name)
    assert f"file_ids={value} " in f"{rendered.splitlines()[0]} "
    assert "truncated" not in rendered.splitlines()[0]


def test_an_overlong_field_value_is_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One field must not be able to crowd out the rest of the line.

    Pins the boundary rather than only the marker: with the length unasserted
    this passed for any bound at all, so a change from 256 to 4096 would have
    kept a green run while one field again took over the line.
    """
    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException):
            files_api._raise_durable_storage_unavailable(
                _wrapped_fault(), "upload", file_ids="x" * 5000
            )

    message_line = _sole_warning(caplog, files_api.logger.name).splitlines()[0]
    # The retained part is the intact leading prefix, at exactly the bound --
    # one character more is the off-by-one this pins.
    assert f"file_ids={'x' * _MAX_LOG_VALUE_LENGTH}...[truncated]" in message_line
    assert "x" * (_MAX_LOG_VALUE_LENGTH + 1) not in message_line
    assert len(message_line) < 1000
    # The magnitude is part of the contract too, not just the mechanics: the
    # cap exists so one field cannot crowd out the rest of the line, and the
    # assertions above are all written against the constant, so they would stay
    # green if it were raised to a value that defeats that. A ceiling rather
    # than an equality, so the number stays tunable within its purpose.
    assert _MAX_LOG_VALUE_LENGTH <= 512
