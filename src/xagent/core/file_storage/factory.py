from __future__ import annotations

from functools import lru_cache
from typing import Sequence
from urllib.parse import urlparse

import fsspec

from ...config import (
    get_file_materialize_dir,
    get_file_storage_options,
    get_file_storage_uri,
)
from .keys import build_user_key_prefix
from .scoped import ScopedFileStorage
from .storage import FsspecFileStorage

# Use modest standard retries for transient failures because s3fs adds retry
# layers around writes/range reads; timeouts are per attempt, not a total deadline.
_DEFAULT_S3_CONFIG_KWARGS = {
    "connect_timeout": 3,
    "read_timeout": 10,
    "retries": {
        "mode": "standard",
        "total_max_attempts": 3,
    },
}


def get_file_storage_backend() -> str:
    """Backend scheme of the configured durable storage ("s3", "file", ...)."""
    return get_unscoped_file_storage().backend


def get_user_file_storage(
    user_id: int, *, scope_segments: Sequence[str] = ()
) -> ScopedFileStorage:
    """Get a storage view scoped to a user's key prefix.

    This is the primary storage factory: every operation through the returned
    handle is confined to ``users/{user_id}``. Reach for
    ``get_unscoped_file_storage`` only in infrastructure code inside this
    package.

    When ``scope_segments`` is given, the handle is confined to the deeper
    subtree ``users/{user_id}/{segment}/...`` instead. That prefix is a strict
    extension of the user root, so existing owner-level keys still validate
    while keys belonging to a sibling scope are rejected — the durable-storage
    counterpart to the sandbox filesystem allowlist. Pass the segments from
    :attr:`ExecutionScope.durable_storage_segments` so the narrowing follows
    ``isolate_external_dirs``.
    """
    return ScopedFileStorage(
        storage=get_unscoped_file_storage(),
        prefix=build_user_key_prefix(int(user_id), scope_segments),
    )


@lru_cache
def get_unscoped_file_storage() -> FsspecFileStorage:
    """Build the configured durable file storage backend (no prefix scope)."""
    uri = get_file_storage_uri()
    options = get_file_storage_options()
    parsed = urlparse(uri)
    backend = parsed.scheme or "file"
    if backend == "s3":
        options = _with_default_s3_config_kwargs(options)

    try:
        fs, root = fsspec.core.url_to_fs(uri, **options)
    except ImportError as exc:
        if backend == "s3":
            raise RuntimeError(
                "XAGENT_FILE_STORAGE_URI uses s3:// but s3fs is not installed"
            ) from exc
        raise

    return FsspecFileStorage(
        fs=fs,
        root=str(root),
        backend=backend,
        base_uri=uri,
        materialize_dir=get_file_materialize_dir(),
    )


def _with_default_s3_config_kwargs(options: dict) -> dict:
    config_kwargs = options.get("config_kwargs")
    if config_kwargs is None:
        return {**options, "config_kwargs": dict(_DEFAULT_S3_CONFIG_KWARGS)}
    if not isinstance(config_kwargs, dict):
        return options
    return {
        **options,
        "config_kwargs": {
            **_DEFAULT_S3_CONFIG_KWARGS,
            **config_kwargs,
        },
    }
