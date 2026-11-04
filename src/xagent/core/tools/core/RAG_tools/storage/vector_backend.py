"""KB engine selection and the per-deployment engine lock.

``XAGENT_VECTOR_BACKEND`` names this deployment's KB engine; the
:class:`~.contracts.VectorIndexStore` is always the LanceDB ledger. The engine is
recorded in the LanceDB data directory at first start, and startup is refused
whenever the setting differs from the record.

Test isolation: pytest should call ``reset_rag_storage_for_tests`` in
``storage.factory`` instead of importing a specific provider.
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from filelock import FileLock, Timeout

from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Primary env var (namespaced to avoid collisions with other libs).
VECTOR_BACKEND_ENV: Final[str] = "XAGENT_VECTOR_BACKEND"

# Backward-compatible alias used in some deployments / docs.
VECTOR_BACKEND_ENV_LEGACY: Final[str] = "VECTOR_STORE_BACKEND"

KB_ENGINE_RECORD: Final[str] = ".kb-engine"
_KB_DATA_TABLES: Final = ("documents", "collection_config", "collection_metadata")
_LOCK_TIMEOUT_SECONDS: Final = 120


class KBStorageBackend(StrEnum):
    """KB engine of a deployment; collection bindings use the same values."""

    LANCEDB = "lancedb"
    MILVUS = "milvus"
    QDRANT = "qdrant"


VectorBackend = KBStorageBackend


def _parse_backend(raw: str) -> VectorBackend:
    """Parse and validate backend string."""
    key = raw.strip().lower()
    if not key:
        return VectorBackend.LANCEDB
    try:
        return VectorBackend(key)
    except ValueError as exc:
        allowed = ", ".join(sorted(b.value for b in VectorBackend))
        raise ConfigurationError(
            f"Invalid {VECTOR_BACKEND_ENV}={raw!r}. Choose one of: {allowed}."
        ) from exc


def get_configured_vector_backend() -> VectorBackend:
    """Read configured vector backend from the environment.

    Precedence: ``XAGENT_VECTOR_BACKEND``, then ``VECTOR_STORE_BACKEND``,
    then default ``lancedb``.

    Returns:
        Selected :class:`VectorBackend`.

    Raises:
        ConfigurationError: If the value is not a known backend name.
    """
    raw = os.environ.get(VECTOR_BACKEND_ENV)
    if raw is None or raw.strip() == "":
        raw = os.environ.get(VECTOR_BACKEND_ENV_LEGACY, "")
    return _parse_backend(raw)


def require_implemented_vector_backend(backend: VectorBackend) -> None:
    """Refuse a KB engine that is not implemented yet; Qdrant is a reserved value.

    Args:
        backend: Resolved backend.

    Raises:
        ConfigurationError: If the backend is known but not implemented yet.
    """
    if backend in (VectorBackend.LANCEDB, VectorBackend.MILVUS):
        return
    raise ConfigurationError(
        f"KB engine {backend.value!r} is not implemented yet. "
        f"Set {VECTOR_BACKEND_ENV} to lancedb (default) or milvus."
    )


def _tables_with_rows(conn: Any, names: list[str]) -> list[str]:
    from ..LanceDB.schema_manager import _safe_close_table

    blocking = []
    for name in names:
        table = None
        try:
            table = conn.open_table(name)
            if table.count_rows() > 0:
                blocking.append(name)
        except Exception:  # noqa: BLE001 - an unreadable table counts as data
            blocking.append(name)
        finally:
            _safe_close_table(table)
    return blocking


def _detect_engine(
    db_dir: str, configured: KBStorageBackend
) -> tuple[KBStorageBackend, list[str]] | None:
    import lancedb

    from ..LanceDB.schema_manager import KB_IDS_TABLE, _safe_close_table
    from ..utils.lancedb_query_utils import list_table_names

    conn = None
    try:
        # Not the cached connection: the finally below closes this one.
        conn = lancedb.connect(db_dir)
        names = list_table_names(conn)
    except Exception as exc:  # noqa: BLE001 - unlistable KB data is unreachable
        _safe_close_table(conn)
        logger.warning("Cannot detect the KB engine in %s: %s", db_dir, exc)
        return None
    try:
        blocking = _tables_with_rows(
            conn, [KB_IDS_TABLE] if KB_IDS_TABLE in names else []
        )
        if blocking:
            return KBStorageBackend.MILVUS, blocking
        blocking = _tables_with_rows(
            conn,
            [
                name
                for name in names
                if name in _KB_DATA_TABLES or name.startswith("embeddings_")
            ],
        )
        return (KBStorageBackend.LANCEDB if blocking else configured), blocking
    finally:
        _safe_close_table(conn)


def _unreachable(db_dir: str) -> bool:
    try:
        mode = os.stat(db_dir).st_mode
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return not stat.S_ISDIR(mode) or not os.access(db_dir, os.R_OK | os.X_OK)


def _read_record(path: Path) -> KBStorageBackend | None:
    try:
        return KBStorageBackend(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read the KB engine record {path}: {exc}"
        ) from exc
    except ValueError as exc:
        raise ConfigurationError(
            f"KB engine record {path} does not name a known engine ({exc}); "
            "delete the record file and restart to detect the engine again."
        ) from exc


def _lock_record(path: Path) -> FileLock | None:
    lock_path = path.with_name(f"{path.name}.lock")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_SECONDS)
        lock.acquire()
    except Timeout:
        logger.warning(
            "Timed out after %ss waiting for %s, held by another process, or the "
            "filesystem does not support flock; the KB engine is not recorded",
            _LOCK_TIMEOUT_SECONDS,
            lock_path,
        )
        return None
    except (OSError, NotImplementedError) as exc:
        logger.warning("Cannot record the KB engine in %s: %s", path, exc)
        return None
    return lock


def _write_record(path: Path, engine: KBStorageBackend) -> bool:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(f"{engine.value}\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as exc:
        logger.warning("Cannot record the KB engine in %s: %s", path, exc)
        return False
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    logger.info("Recorded KB engine %s in %s", engine.value, path)
    return True


def lock_deployment_kb_engine() -> KBStorageBackend:
    """Record the KB engine at first start and refuse a setting that differs.

    Without a record, rows in ``kb_ids`` mean Milvus and rows in the KB data
    tables mean an upgraded LanceDB deployment; otherwise the setting is
    recorded. Detection and the write share one file lock. When the lock or
    the write fails, the detected engine is compared without a record. When
    the LanceDB directory cannot be resolved, reached or listed, it holds no
    reachable KB data, so only a warning is logged.

    Raises:
        ConfigurationError: If the setting is not implemented, the record cannot
            be read or does not name a known engine, or the setting differs from
            the recorded or detected engine.
    """
    from ......providers.vector_store.lancedb import LanceDBConnectionManager

    configured = get_configured_vector_backend()
    require_implemented_vector_backend(configured)
    try:
        db_dir = LanceDBConnectionManager().resolve_dir_from_env()
    except ValueError:
        logger.warning(
            "LANCEDB_DIR is empty; the KB engine is not checked and KB access will fail"
        )
        return configured
    except OSError as exc:
        # Only a default directory that cannot be created or listed raises here,
        # and no KB data is reachable there.
        logger.warning("Cannot create the LanceDB directory for the KB engine: %s", exc)
        return configured
    if _unreachable(db_dir):
        logger.warning(
            "Cannot reach the LanceDB directory %s; the KB engine is not checked",
            db_dir,
        )
        return configured
    path = Path(db_dir) / KB_ENGINE_RECORD
    blocking: list[str] = []
    written = True
    # os.replace publishes the record whole, so reading it needs no lock.
    recorded = _read_record(path)
    if recorded is None:
        lock = _lock_record(path)
        try:
            if lock is not None:
                recorded = _read_record(path)
            if recorded is None:
                detected = _detect_engine(db_dir, configured)
                if detected is None:
                    return configured
                recorded, blocking = detected
                written = lock is not None and _write_record(path, recorded)
        finally:
            if lock is not None:
                lock.release()
    if recorded is not configured:
        held = f" ({', '.join(blocking)} hold data)" if blocking else ""
        if written:
            source = f"recorded in {path}"
            fix = (
                "To change the engine of an empty deployment, "
                "delete the record file and restart."
            )
        else:
            source = f"detected because {path} could not be written"
            fix = (
                f"Make {path.parent} writable (with flock support), "
                f"or set {VECTOR_BACKEND_ENV} to {recorded.value}."
            )
        raise ConfigurationError(
            f"This deployment's KB engine is {recorded.value}{held}, {source}, "
            f"but {VECTOR_BACKEND_ENV} is {configured.value}. {fix}"
        )
    return recorded
