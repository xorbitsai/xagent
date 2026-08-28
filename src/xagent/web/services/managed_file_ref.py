from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import mimetypes
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, NoReturn, Protocol, cast

from ...core.execution_scope import (
    EXECUTION_SCOPE_NOT_PROVIDED,
    ExecutionScope,
    ExecutionScopeAuthorityError,
    ExecutionScopeInput,
    ExecutionScopeResolverContractError,
    get_execution_scope,
    resolve_execution_scope,
    resolve_execution_scope_off_turn,
)
from ...core.file_storage import (
    FsspecFileStorage,
    ScopedFileStorage,
    StorageKeyScopeError,
    StoredObject,
    get_user_file_storage,
)
from ...core.file_storage.faults import classify_provider_fault
from ...core.file_storage.keys import (
    build_task_output_storage_key as build_task_output_storage_key,
)
from ...core.file_storage.keys import (
    build_upload_storage_key as build_upload_storage_key,
)
from ...core.file_storage.keys import safe_storage_filename as safe_storage_filename
from ..models.uploaded_file import UploadedFile

logger = logging.getLogger(__name__)
FILE_INTEGRITY_REUPLOAD_MESSAGE = (
    "File integrity verification failed. Please re-upload this file."
)


class DurableObjectMissingError(FileNotFoundError):
    """Raised when a registered file has no local copy or durable object."""


# Field values are rendered into a plain-text ``key=value`` line, so a value
# must stay one token: whitespace in it would let a client-supplied string put
# what reads as further fields on the same record (CWE-117). Quoting is not
# enough -- it only protects consumers that honour quotes, and the deployed
# logs are read by whitespace tokenizers -- so whitespace is escaped instead.
# Some of these values are client-supplied (``/v1/*`` renders the request's
# ``files`` list, an unvalidated ``list[str]``), so a caller cannot be relied
# on to have checked.
_LOG_VALUE_TRANSLATION = str.maketrans({"\n": "\\n", "\r": "\\r", "\t": "\\t"})
_LOG_VALUE_WHITESPACE = re.compile(r"\s")

# Long enough for a UUID list of realistic size, short enough that one field
# cannot crowd out the rest of the line.
_MAX_LOG_VALUE_LENGTH = 256

# The stable prefix every durable-fault line starts with, and the anchor the
# operational note tells alerts to key on. Public because tests assert on its
# *absence* to prove an integrity fault was not reported as an outage, and an
# inlined copy there would pass vacuously once this text changed.
DURABLE_FAULT_LOG_PREFIX = "Durable storage unavailable during"


def _escape_whitespace(match: re.Match[str]) -> str:
    """Render one whitespace character as an unambiguous escape."""
    code = ord(match.group())
    return f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}"


def _sanitize_log_value(value: object) -> str:
    """Flatten a field value to one bounded, single-token string.

    Callers filter ``None`` out before rendering; a ``None`` field is dropped
    from the line entirely (whether it should render instead is #1642).

    The escape character goes first so the result stays decodable: ``\\\\``
    is a literal backslash, ``\\n`` a newline, ``\\x20`` a space. What this
    does not do is make the whole *record* unforgeable -- ``exc_info`` text is
    rendered verbatim by the formatter (#1516), and ESC is not whitespace so it
    still passes through (#1519).
    """
    text = str(value).replace("\\", "\\\\")
    text = text.translate(_LOG_VALUE_TRANSLATION)
    text = _LOG_VALUE_WHITESPACE.sub(_escape_whitespace, text)
    if len(text) > _MAX_LOG_VALUE_LENGTH:
        return f"{text[:_MAX_LOG_VALUE_LENGTH]}...[truncated]"
    return text


def log_durable_storage_fault(
    target_logger: logging.Logger,
    operation: str,
    exc: Exception,
    **fields: object,
) -> None:
    """Record a durable-storage fault with its full cause chain.

    The single implementation for every site that reports a durable-storage
    fault *as such*. That is narrower than every consumer of the class: it is
    a ``RuntimeError``, so broad ``except Exception``/``except RuntimeError``
    arms (chat.py, kb.py, core/workspace.py) absorb it without naming it, and
    each keeps its own ``exc_info`` instead of the fields rendered here.

    The provider class, HTTP status, and throttle-vs-timeout-vs-credentials
    distinction live only in ``__cause__``, and none of the envelopes these
    faults become -- ``HTTPException``, the ``/v1/*`` body, a tool result --
    carries a traceback, so ``exc_info`` is what gets them into a log (#1467).

    ``target_logger`` is the caller's, so a line stays attributed to the
    endpoint or tool that produced it. ``operation`` must be a bounded label,
    a small fixed set of values, so it stays aggregatable; per-request
    identifiers belong in ``fields``.

    Fields are rendered as ``key=value`` *in the message*, not via ``extra=``:
    the deployed formatter (``web/logging_config.py``) renders only
    ``%(message)s``, so anything passed as ``extra`` is invisible in exactly
    the logs an incident is read from. Values are escaped and bounded because
    some are client-supplied; that bounds the fields, not the whole record,
    whose ``exc_info`` text the formatter still renders verbatim (#1516).

    The classified provider fault is appended so a burst of these can be
    aggregated by cause -- one throttle reads very differently from a thousand,
    and both differ from a rejected credential. ``retryable=False`` there is a
    diagnostic claim, not a routing one: every fault in this family still
    answers the same status.
    """
    # One record per fault instance: a wrap can cross several arms that each
    # legitimately report it, and the mark is what makes them safe to write
    # independently. Only this module's wraps are marked, and the attribute is
    # declared on the class, so neither the read nor the write needs a guard.
    if isinstance(exc, DurableStorageOperationError):
        if exc._durable_fault_logged:
            return
        # The key rides on the exception, not in its message (#1643), so it
        # reaches this line from every site -- including ones that never pass
        # it as a field -- while ``str(exc)`` stays safe wherever it escapes.
        # An explicit field wins, so a caller can still name a different key.
        if exc.storage_key and "storage_key" not in fields:
            fields = {**fields, "storage_key": exc.storage_key}

    # One renderer for both the caller's identifiers and the classified fault,
    # so escaping and layout are decided in a single place. Values from the
    # classifier are sanitised on the same terms as request identifiers: a
    # provider ``Code`` is read out of a duck-typed response dict, so a
    # loosely-conforming backend could put a space or newline in it.
    # Classified fields last: they are derived from the exception itself, so
    # where a caller passes a key of the same name the exception's own answer
    # is the one to keep. No caller does today; the precedence is stated so a
    # future collision is a decision rather than a surprise.
    merged = {**fields, **classify_provider_fault(exc).as_fields()}
    rendered = "".join(
        f" {name}={_sanitize_log_value(value)}"
        for name, value in merged.items()
        if value is not None
    )
    # The prefix stays a literal in the template, not an argument: this repo
    # reads ``record.msg`` as the event's identity, and ``"%s %s%s"`` identifies
    # nothing. ``DURABLE_FAULT_LOG_PREFIX`` is the same text for callers and
    # tests to match on, pinned against this line by
    # ``test_the_exported_prefix_is_the_one_the_helper_emits``.
    target_logger.warning(
        "Durable storage unavailable during %s%s", operation, rendered, exc_info=exc
    )
    # Marked only after the record is emitted: if emission raised, the fault
    # would stay unmarked for a later arm to report, instead of being
    # permanently recorded as logged by an attempt that never wrote.
    if isinstance(exc, DurableStorageOperationError):
        exc._durable_fault_logged = True


class DurableStorageOperationError(RuntimeError):
    """Raised when durable object storage is unavailable for an operation.

    **The storage key belongs in ``storage_key``, never in the message.** Its
    scope segments encode the owning user's id, and ``str(exc)`` on this class
    reaches places this module does not control: a bare ``raise`` from a
    WebSocket fault arm carries it into a task-wide broadcast and into a
    persisted command row, and broad ``except RuntimeError`` arms interpolate
    it straight into client-facing text (#1497). Redacting those egresses one
    at a time is how the leak kept returning under new names; with the key off
    the message at every construction in this module -- enforced by the purity
    test in ``tests/web/services/test_managed_file_ref.py`` -- no ``str(exc)``
    egress can reintroduce it from here. The guarantee is module-scoped: that
    scan does not cover other modules, and ``uploaded_file_store.py`` still
    carries other identifiers (it has no storage key) in its own message
    (#1642).

    ``log_durable_storage_fault`` renders the attribute as a field, so every
    line that reports the fault *as such* still carries the key. The broad
    absorbers that render ``str(e)`` -- chat.py, kb.py, core/workspace.py --
    lose it from their lines until they read the attribute, which is #1515's
    scope.

    ``storage_key`` is required (still nullable) so an omission is a type
    error at the call site rather than a silently ``None`` attribute --
    pass ``storage_key=None`` only where there genuinely is no key yet.
    """

    # Set by ``log_durable_storage_fault`` so one fault yields one record even
    # when several arms legitimately report it. Declared here rather than
    # attached dynamically so it is typed, discoverable, and always present to
    # read -- the reason that function needs no guard around the mark.
    _durable_fault_logged: bool = False

    def __init__(self, message: str, *, storage_key: str | None) -> None:
        super().__init__(message)
        self.storage_key = storage_key


class DurableObjectIntegrityError(DurableStorageOperationError):
    """Raised when restored durable bytes do not match the DB checksum."""


# Namespace-authority faults that must never be folded into
# ``DurableStorageOperationError`` by the storage-call wraps below. That
# fallback exists for *backend* faults -- an unreachable object store, a lost
# metadata acknowledgement, a read that times out -- which callers surface as a
# retryable "durable storage is temporarily unavailable". A containment
# violation is not one of those: ``StorageKeyScopeError`` means the key falls
# outside the prefix this handle is bound to (the key encodes the wrong
# namespace, or the handle was bound to the wrong one), and
# ``ExecutionScopeAuthorityError`` means the resolver and the persisted
# snapshot disagree about which namespace the task owns, and
# ``ExecutionScopeResolverContractError`` means a resolver broke its return
# contract or the persisted snapshot cannot be decoded at all -- reachable
# here because binding the handle settles a deferred per-task resolution (see
# ``_bound_storage``). All three are permanent configuration/authority faults
# that no retry can clear, so they propagate as themselves and are classified
# once, at the application boundary (see the handlers registered in
# ``web/app.py``), instead of being reported as an outage an operator would
# retry.
NAMESPACE_AUTHORITY_ERRORS: tuple[type[BaseException], ...] = (
    StorageKeyScopeError,
    ExecutionScopeAuthorityError,
    ExecutionScopeResolverContractError,
)


class UploadedFileLocalPathRecord(Protocol):
    """Fields needed to restore or locate an uploaded file without an ORM session."""

    @property
    def user_id(self) -> Any: ...

    @property
    def file_id(self) -> Any: ...

    @property
    def filename(self) -> Any: ...

    @property
    def storage_path(self) -> Any: ...

    @property
    def storage_key(self) -> Any: ...

    @property
    def storage_status(self) -> Any: ...

    @property
    def checksum(self) -> Any: ...


def guess_media_type(filename: str) -> str:
    media_type, _ = mimetypes.guess_type(filename)
    return media_type or "application/octet-stream"


def iter_file_handle(handle: Any) -> Any:
    try:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        handle.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_to_sha256_hex(checksum: str) -> str | None:
    normalized = checksum.strip()
    if normalized.lower().startswith("sha256:"):
        normalized = normalized.split(":", 1)[1]

    try:
        bytes.fromhex(normalized)
    except ValueError:
        pass
    else:
        if len(normalized) == 64:
            return normalized.lower()

    try:
        decoded = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError):
        return None
    return decoded.hex() if len(decoded) == 32 else None


def _checksum_matches(expected_checksum: str, actual_checksum: str) -> bool:
    expected_sha256 = _checksum_to_sha256_hex(expected_checksum)
    actual_sha256 = _checksum_to_sha256_hex(actual_checksum)
    return (
        expected_sha256 is not None
        and actual_sha256 is not None
        and expected_sha256 == actual_sha256
    )


@dataclass
class ManagedFileRef:
    """Registered file handle with local-first durable fallback semantics.

    A record with no durable key yet is ambiguous at construction time: it
    is either a fresh upload about to be written, or a ``storage_status=
    "pending"`` record a read-only endpoint is merely trying to serve from
    local storage (see ``__post_init__``). Rather than asking every caller
    to declare its intent, this class settles the namespace per operation:
    an operation that addresses bytes that already exist resolves off-turn,
    so a resolver-authoritative mismatch downgrades instead of turning a
    servable read into an error, while ``sync_to_durable`` -- the operation
    that composes a key from the resolved namespace -- resolves fail-closed
    before it does so. ``_pending_scope_task_id`` records that the namespace
    still has to be recovered per task, as opposed to having come from an
    explicit ``execution_scope`` argument or the ambient turn contextvar.

    Off-turn resolution is not the same as never failing: it downgrades a
    mismatch only where the resolver produced an authoritative answer to
    downgrade to. An abstaining resolver that disagrees with the snapshot, a
    resolver that violates its own return contract, and a snapshot loader
    that raises all propagate rather than downgrade. Construction therefore
    does not perform that resolution at all: it records the task to recover
    from and resolves on first use by an operation that needs a namespace.
    A read of a keyless record needs none -- ``ensure_local`` and
    ``materialize`` refuse a record with no durable object before they touch
    storage -- so a dispute cannot turn a servable read into an error.

    A key supplied by the caller bypasses the re-check below, because the
    namespace it encodes was chosen wherever that key was composed. Callers
    that compose a key themselves therefore own that decision; this class
    only covers the keys it composes.
    """

    record: UploadedFileLocalPathRecord
    storage: FsspecFileStorage | ScopedFileStorage = field(default=None)  # type: ignore[assignment]
    execution_scope: ExecutionScopeInput = EXECUTION_SCOPE_NOT_PROVIDED
    _scope_segments: tuple[str, ...] = field(default=(), init=False, repr=False)
    # Not None while a per-task scope recovery is owed but has not run yet.
    # Cleared by the recovery, so two operations on one ref cannot bind to
    # two different namespaces.
    _pending_scope_task_id: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Settle the active scope once, so the bound handle and any key
        # written through it (see ``sync_to_durable``) agree even if the
        # ambient scope changes later or the ref outlives the turn. The
        # ambient contextvar is sampled here, at construction, for exactly
        # that reason; only the per-task resolver/snapshot recovery is
        # deferred, because that is the one step that can raise and the one
        # whose answer a read of a keyless record never consumes.
        #
        # Precedence mirrors ``AgentServiceManager.get_agent_for_task``:
        # explicit arg -> ambient turn contextvar -> the per-task
        # resolver/snapshot keyed on the record's ``task_id``. The last step
        # matters on off-turn paths (bot/builder-chat handlers never enter
        # an ``ExecutionScopeContext``, so the contextvar is None): a workforce
        # sub-task carries its scope only in the persisted snapshot, and
        # without recovering it here ``sync_to_durable`` would write a new
        # file under the owner root instead of the sub-task's scoped subtree.
        # Explicit ``None`` means owner-root. Only an omitted argument may
        # inherit the ambient turn scope or consult per-task resolution.
        scope_input = self.execution_scope
        if scope_input is EXECUTION_SCOPE_NOT_PROVIDED:
            scope = get_execution_scope()
            if scope is None:
                task_id = getattr(self.record, "task_id", None)
                if task_id is not None and not self.storage_key:
                    # No durable key yet means either a fresh object is
                    # about to be placed somewhere under this scope's
                    # subtree, or the record is simply routine for reads (a
                    # freshly-uploaded record sits at storage_status=
                    # "pending" with no key, and list/download/preview
                    # endpoints construct a ref for exactly those records).
                    # Presence of a key cannot tell write intent from read
                    # intent apart, so construction picks neither: it records
                    # the task and resolves nothing. Whichever operation
                    # first needs a namespace supplies the policy --
                    # ``sync_to_durable`` fail-closed before it composes a
                    # key, everything else off-turn, downgrading a
                    # resolver-authoritative mismatch to the resolver's
                    # answer plus a warning. A read of a keyless record asks
                    # for no namespace at all, so the paths that can raise
                    # (an abstaining resolver disagreeing with the snapshot,
                    # a resolver breaking its return contract, a loader that
                    # raises) cannot turn a servable file into a 500.
                    #
                    # A record that already has a durable key is the read
                    # (or rewrite-in-place) path: that key already fixes the
                    # object's location, off-turn re-resolution is
                    # unnecessary, and -- on a resolver/snapshot drift --
                    # would make an already-legitimate key look foreign to a
                    # narrower re-derived scope (see ``storage`` binding
                    # below).
                    #
                    # Skipping this branch (or a downgrade landing on
                    # ``None``) leaves ``scope`` at ``None``, so
                    # ``self._scope_segments`` below is ``()`` and
                    # ``self.storage`` binds to the owner root
                    # (``get_user_file_storage(user_id, scope_segments=())``)
                    # rather than the narrower scoped subtree. Every
                    # operation on this ref -- ``ensure_local``/
                    # ``materialize`` (read), ``signed_access_url`` (sign),
                    # ``delete_durable`` (delete),
                    # ``_verify_durable_checksum_for_direct_access``
                    # (integrity check) -- goes through that same
                    # ``self.storage`` with the fixed ``self.storage_key``,
                    # so none of them get ``ScopedFileStorage``'s
                    # prefix-containment narrowing to the scope subtree; they
                    # are only constrained to the owner's key space. That is
                    # safe here because ``storage_key`` is an absolute key
                    # already chosen under the writing scope -- these
                    # operations address a fixed object rather than
                    # constructing a new key relative to a prefix, so a
                    # containment check against a re-derived (and possibly
                    # drifted) scope would add no protection, only risk
                    # rejecting an already-legitimate key. This is a real
                    # loosening relative to the in-turn binding for the same
                    # read (which narrows through the ambient contextvar
                    # scope and can raise ``StorageKeyScopeError``), not just
                    # a neutral no-op -- it is deliberate for the reason
                    # above, not an oversight.
                    self._pending_scope_task_id = task_id
                    return
        else:
            scope = cast(ExecutionScope | None, scope_input)
        self._scope_segments = (
            scope.durable_storage_segments if scope is not None else ()
        )
        self._bind_storage()

    def _bind_storage(self) -> None:
        """Bind the handle to the settled namespace, unless one was passed in."""
        if self.storage is not None:
            return
        user_id = self.record.user_id
        if user_id is None:
            raise ValueError(
                "Record user_id is required to bind user-scoped storage; "
                "pass an explicit storage handle for records without an owner"
            )
        self.storage = get_user_file_storage(
            int(user_id), scope_segments=self._scope_segments
        )

    def _settle_pending_scope(
        self, resolve: Callable[[Any], ExecutionScope | None]
    ) -> None:
        """Run the deferred per-task recovery, then bind.

        ``resolve`` is the caller's policy: the off-turn form for operations
        that address bytes that already exist, and the fail-closed form where
        a namespace is being chosen for new bytes. Whichever runs first wins
        for the life of this ref, which is why the only caller that passes
        the fail-closed form does so before composing a key.
        """
        task_id = self._pending_scope_task_id
        if task_id is not None:
            scope = resolve(task_id)
            self._scope_segments = (
                scope.durable_storage_segments if scope is not None else ()
            )
            self._pending_scope_task_id = None
        self._bind_storage()

    def _bound_storage(self) -> FsspecFileStorage | ScopedFileStorage:
        """The storage handle, settling a deferred scope recovery on first use."""
        if self.storage is None:
            self._settle_pending_scope(resolve_execution_scope_off_turn)
        return self.storage

    @property
    def local_path(self) -> Path:
        return Path(str(self.record.storage_path))

    @property
    def filename(self) -> str:
        return str(self.record.filename)

    @property
    def storage_key(self) -> str:
        return str(self.record.storage_key or "")

    @property
    def has_durable_object(self) -> bool:
        return bool(self.storage_key and self.record.storage_status == "available")

    def ensure_local(self) -> Path:
        path = self.local_path
        if path.exists() and path.is_file():
            return path

        if not self.has_durable_object:
            raise DurableObjectMissingError(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        try:
            self._bound_storage().copy_to_path(self.storage_key, temp_path)
            self._verify_content_checksum(temp_path)
            temp_path.replace(path)
            return path
        except DurableObjectIntegrityError:
            temp_path.unlink(missing_ok=True)
            raise
        except NAMESPACE_AUTHORITY_ERRORS:
            temp_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise DurableStorageOperationError(
                "Failed to restore durable object",
                storage_key=self.storage_key,
            ) from exc

    def materialize(self, *, allow_existing_local: bool = True) -> Path:
        path = self.local_path
        if allow_existing_local and path.exists() and path.is_file():
            return path

        if not self.has_durable_object:
            raise DurableObjectMissingError(path)

        last_integrity_error: DurableObjectIntegrityError | None = None
        for _attempt in range(2):
            materialized_path: Path | None = None
            try:
                materialized_path = self._bound_storage().materialize(
                    self.storage_key, self.filename
                )
                self._verify_content_checksum(materialized_path)
                return materialized_path
            except DurableObjectIntegrityError as exc:
                last_integrity_error = exc
                if materialized_path is not None:
                    materialized_path.unlink(missing_ok=True)
            except NAMESPACE_AUTHORITY_ERRORS:
                raise
            except Exception as exc:
                raise DurableStorageOperationError(
                    "Failed to materialize durable object",
                    storage_key=self.storage_key,
                ) from exc

        if last_integrity_error is not None:
            raise last_integrity_error
        raise DurableStorageOperationError(
            "Failed to materialize durable object",
            storage_key=self.storage_key,
        )

    def open_read(self) -> BinaryIO:
        return self.ensure_local().open("rb")

    def signed_access_url(
        self,
        *,
        expires: int,
        content_type: str | None = None,
        content_disposition: str | None = None,
    ) -> str | None:
        if not self.has_durable_object:
            return None
        if not self._verify_durable_checksum_for_direct_access():
            return None
        try:
            return self._bound_storage().signed_url(
                self.storage_key,
                expires=expires,
                content_type=content_type,
                content_disposition=content_disposition,
            )
        except NAMESPACE_AUTHORITY_ERRORS:
            raise
        except Exception as exc:
            raise DurableStorageOperationError(
                "Failed to sign durable object URL",
                storage_key=self.storage_key,
            ) from exc

    def sync_to_durable(
        self,
        *,
        storage_key: str | None = None,
        mime_type: str | None = None,
    ) -> StoredObject:
        path = self.local_path
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        resolved_key = storage_key or self.storage_key
        if not resolved_key:
            # Composing a fresh key selects the namespace new bytes land
            # under -- getting it wrong silently and durably writes one
            # tenant's bytes under another tenant's subtree. Construction
            # (see ``__post_init__``) resolves off-turn instead, because a
            # keyless ref might just as well be serving a read, so the
            # fail-closed check belongs here, where the namespace is actually
            # baked into a key. This covers the keys composed here only:
            # every caller that hands ``sync_to_durable`` a ``storage_key``
            # skips this branch and owns the namespace in that key itself.
            #
            # When ``_scope_segments`` came from off-turn per-task recovery,
            # re-resolve it here, fail-closed, right before it is baked into
            # the key. This second resolution cannot disagree with the one
            # already baked into ``_scope_segments`` in the case that
            # matters: a disagreement is exactly what makes this fail-closed
            # call raise, and it raises before ``resolved_key`` is computed
            # or anything is written -- so this is a genuine re-check, not
            # redundant work re-deriving the same answer.
            self._settle_pending_scope(resolve_execution_scope)
            scope_segments = self._scope_segments
            resolved_key = build_upload_storage_key(
                int(self.record.user_id),
                str(self.record.file_id),
                self.filename or path.name,
                scope_segments=scope_segments,
            )
        try:
            stored_object = self._bound_storage().put_file(
                path,
                resolved_key,
                mime_type or getattr(self.record, "mime_type", None),
            )
        except NAMESPACE_AUTHORITY_ERRORS:
            raise
        except Exception as exc:
            raise DurableStorageOperationError(
                "Failed to write durable object",
                storage_key=resolved_key,
            ) from exc
        self.apply_stored_object(stored_object)
        setattr(self.record, "file_size", path.stat().st_size)
        return stored_object

    def adopt_existing_object(
        self, expected_key: str
    ) -> Literal["adopted", "uploaded", "missing"]:
        local_path = self.local_path
        local_exists = local_path.exists() and local_path.is_file()
        try:
            stored_object = self._bound_storage().stat(expected_key)
        except FileNotFoundError:
            return "missing"
        except NAMESPACE_AUTHORITY_ERRORS:
            raise
        except Exception as exc:
            raise DurableStorageOperationError(
                "Failed to inspect durable object metadata",
                storage_key=expected_key,
            ) from exc

        checksum = stored_object.checksum
        if local_exists:
            local_size = local_path.stat().st_size
            remote_size_raw = getattr(stored_object, "size", None)
            remote_size = None if remote_size_raw is None else int(remote_size_raw)
            if remote_size is not None and remote_size != local_size:
                self.sync_to_durable(
                    storage_key=expected_key,
                    mime_type=getattr(self.record, "mime_type", None),
                )
                return "uploaded"
            if not checksum:
                try:
                    checksum = self._bound_storage().content_hash(expected_key)
                except NAMESPACE_AUTHORITY_ERRORS:
                    # The rule holds at every site in this module, not only
                    # where a downstream call happens to raise the same class
                    # again: relying on that would make this a silent swallow
                    # the moment the fallback path changes.
                    raise
                except Exception:
                    self.sync_to_durable(
                        storage_key=expected_key,
                        mime_type=getattr(self.record, "mime_type", None),
                    )
                    return "uploaded"

            if checksum != _sha256_file(local_path):
                self.sync_to_durable(
                    storage_key=expected_key,
                    mime_type=getattr(self.record, "mime_type", None),
                )
                return "uploaded"

        if not checksum:
            try:
                checksum = self._bound_storage().content_hash(expected_key)
            except NAMESPACE_AUTHORITY_ERRORS:
                raise
            except Exception as exc:
                raise DurableStorageOperationError(
                    "Failed to inspect durable object metadata",
                    storage_key=expected_key,
                ) from exc

        self.apply_stored_object(
            StoredObject(
                backend=stored_object.backend,
                key=stored_object.key,
                uri=stored_object.uri,
                size=stored_object.size,
                checksum=checksum,
                etag=stored_object.etag,
            )
        )
        return "adopted"

    def apply_stored_object(self, stored_object: StoredObject) -> None:
        if not stored_object.checksum:
            raise ValueError(
                f"Cannot mark durable object available without checksum: {stored_object.key}"
            )
        setattr(self.record, "storage_backend", stored_object.backend)
        setattr(self.record, "storage_key", stored_object.key)
        setattr(self.record, "storage_uri", stored_object.uri)
        setattr(self.record, "checksum", stored_object.checksum)
        setattr(self.record, "etag", stored_object.etag)
        setattr(self.record, "storage_status", "available")

    def delete_durable(self) -> None:
        if self.has_durable_object:
            self._bound_storage().delete(self.storage_key)

    def _verify_content_checksum(self, path: Path) -> None:
        expected_checksum = getattr(self.record, "checksum", None)
        if not expected_checksum:
            return

        actual_checksum = _sha256_file(path)
        if _checksum_matches(str(expected_checksum), actual_checksum):
            return

        self._raise_integrity_error(str(expected_checksum), actual_checksum)

    def _verify_durable_checksum_for_direct_access(self) -> bool:
        expected_checksum = getattr(self.record, "checksum", None)
        if not expected_checksum:
            logger.warning(
                "Skipping direct durable access without DB checksum: "
                "file_id=%s storage_key=%s",
                getattr(self.record, "file_id", None),
                self.storage_key,
            )
            return False

        try:
            actual_checksum = self._bound_storage().content_hash(self.storage_key)
        except NAMESPACE_AUTHORITY_ERRORS:
            # A permanent authority fault must not be reported as "checksum
            # unavailable", which reads as a transient reason to fall back to
            # backend-mediated access.
            raise
        except Exception as exc:
            # ``exc_info`` for the same reason as everywhere else on this
            # path: this fault is swallowed -- the caller only sees ``False``
            # and silently falls back to backend-mediated delivery -- so this
            # line is the only record of it, and ``error=%s`` alone drops the
            # exception class and the cause chain (#1467).
            logger.warning(
                "Falling back to backend-mediated durable access because content "
                "hash is unavailable: file_id=%s storage_key=%s error=%s",
                getattr(self.record, "file_id", None),
                self.storage_key,
                exc,
                exc_info=exc,
            )
            return False

        if _checksum_matches(str(expected_checksum), str(actual_checksum)):
            return True

        self._raise_integrity_error(str(expected_checksum), str(actual_checksum))

    def _raise_integrity_error(
        self, expected_checksum: str, actual_checksum: str
    ) -> NoReturn:
        logger.error(
            "Durable object integrity check failed: file_id=%s storage_key=%s "
            "expected_checksum=%s actual_checksum=%s",
            getattr(self.record, "file_id", None),
            self.storage_key,
            expected_checksum,
            actual_checksum,
        )
        raise DurableObjectIntegrityError(
            FILE_INTEGRITY_REUPLOAD_MESSAGE, storage_key=self.storage_key
        )


def managed_file_from_record(file_record: UploadedFile) -> ManagedFileRef:
    return ManagedFileRef(file_record)


def ensure_uploaded_file_local_path(
    file_record: UploadedFileLocalPathRecord,
    *,
    execution_scope: ExecutionScopeInput = EXECUTION_SCOPE_NOT_PROVIDED,
) -> Path:
    try:
        return ManagedFileRef(
            file_record,
            execution_scope=execution_scope,
        ).ensure_local()
    except DurableObjectMissingError:
        return Path(str(file_record.storage_path))


def create_uploaded_file_from_local_path(
    *,
    local_path: Path,
    user_id: int,
    filename: str | None = None,
    file_id: str | None = None,
    task_id: int | None = None,
    mime_type: str | None = None,
    storage_key: str | None = None,
    workspace_relative_path: str | None = None,
    workspace_category: str | None = None,
) -> UploadedFile:
    from .uploaded_file_store import create_unbound_uploaded_file_from_local_path

    return create_unbound_uploaded_file_from_local_path(
        local_path=local_path,
        user_id=user_id,
        filename=filename,
        file_id=file_id,
        task_id=task_id,
        mime_type=mime_type,
        storage_key=storage_key,
        workspace_relative_path=workspace_relative_path,
        workspace_category=workspace_category,
    )
