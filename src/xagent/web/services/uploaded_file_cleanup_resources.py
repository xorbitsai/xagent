"""Resource snapshots and safe disposal for the claimed-upload lifecycle."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote, urlsplit
from uuid import uuid4

from ...config import (
    get_external_upload_dirs,
    get_file_materialize_dir,
    get_file_storage_options,
    get_storage_root,
    get_uploads_dir,
)
from ...core.file_storage import get_user_file_storage
from ...core.file_storage.storage import (
    atomic_copy_temp_prefix,
    materialized_file_path,
    materialized_key_directory,
    normalize_storage_key,
)
from ...core.workspace import scoped_user_root
from .managed_file_ref import _checksum_to_sha256_hex
from .uploaded_file_cleanup_disposal import (
    CleanupResourceUncertain,
    CleanupWorkBudget,
    dispose_directory,
)


def _canonical_root(path: Path) -> Path:
    return path.expanduser().resolve()


def _canonical_path(path: Path, configured_root: Path, canonical_root: Path) -> Path:
    """Map a configured-root alias without following children beneath it."""
    expanded = Path(os.path.abspath(path.expanduser()))
    for root in (
        Path(os.path.abspath(configured_root.expanduser())),
        canonical_root,
    ):
        try:
            return canonical_root / expanded.relative_to(root)
        except ValueError:
            continue
    return expanded


def _quarantine_name(name: str, generation: str) -> str:
    ownership = hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]
    return f".cleanup-{ownership}-{generation[:16]}"


def _materialized_candidates(
    materialize: Path, storage_key: str, filename: str
) -> tuple[list[Path], str | None]:
    """List literal filename copies inside one key-owned cache namespace."""
    key_directory = materialized_key_directory(materialize, storage_key)
    candidates: list[Path] = []
    try:
        with _parent_descriptor(key_directory / "entry", materialize) as parent:
            with os.scandir(parent) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        return candidates, "materialization namespace has a symlink"
                    if entry.is_dir(follow_symlinks=False):
                        candidates.append(key_directory / entry.name / filename)
    except FileNotFoundError:
        return candidates, None
    except (OSError, ValueError):
        return candidates, "materialization namespace is unverifiable"
    return candidates, None


def _locator(uri: str) -> str:
    """Persist routing identity without embedding authentication material."""
    parts = urlsplit(uri)
    return (
        parts._replace(netloc=parts.netloc.rsplit("@", 1)[-1], query="", fragment="")
        .geturl()
        .rstrip("/")
    )


def _provider_routing() -> dict[str, str]:
    options = get_file_storage_options()
    client = options.get("client_kwargs") or {}
    return {
        "endpoint": _locator(
            str(options.get("endpoint_url") or client.get("endpoint_url") or "")
        ),
        "region": str(client.get("region_name") or ""),
    }


def _identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns]


def _matches_resource(info: os.stat_result, expected: list[int] | None) -> bool:
    if expected is None:
        return False
    actual = _identity(info)
    return (
        actual[:3] == expected[:3] if stat.S_ISDIR(info.st_mode) else actual == expected
    )


@contextmanager
def _parent_descriptor(path: Path, root: Path) -> Iterator[int]:
    """Pin directories without following symlinks below the configured root."""
    relative = path.relative_to(root)
    if not relative.parts:
        raise CleanupResourceUncertain("Cleanup cannot remove a managed root")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def capture_resource(
    path: Path,
    *,
    root: Path,
    generation: str,
    checksum: str | None = None,
    allow_directory: bool = False,
) -> dict[str, Any]:
    """Capture a bounded owned locator before any destructive operation."""
    resource: dict[str, Any] = {
        "path": str(path),
        "root": str(root),
        "quarantine": _quarantine_name(path.name, generation),
        "checksum": checksum,
        "identity": None,
        "parent": None,
    }
    try:
        with _parent_descriptor(path, root) as parent:
            resource["parent"] = _identity(os.fstat(parent))[:3]
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            resource["identity"] = _identity(info)
            if not stat.S_ISREG(info.st_mode) and not (
                allow_directory and stat.S_ISDIR(info.st_mode)
            ):
                resource["uncertain"] = "unsupported resource or symlink"
            elif stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                resource["preserved"] = "shared hard-linked source"
    except FileNotFoundError:
        pass
    except OSError:
        resource["uncertain"] = "unverifiable path boundary"
    except ValueError:
        resource["uncertain"] = "unverifiable path boundary"
    return resource


def build_cleanup_manifest(snapshot: Any) -> dict[str, Any]:
    """Snapshot metadata, configured locators and local replacement evidence.

    The caller has returned its SQL connection. This performs only filesystem
    metadata reads; content verification belongs to execution after claim commit.
    """
    generation = uuid4().hex
    configured_uploads = get_uploads_dir()
    uploads = _canonical_root(configured_uploads)
    materialize = _canonical_root(get_file_materialize_dir())
    previews = _canonical_root(get_storage_root())
    checksum = _checksum_to_sha256_hex(snapshot.checksum or "")
    key = normalize_storage_key(snapshot.storage_key, strict=False)
    storage = get_user_file_storage(snapshot.user_id)
    expected_uri = f"{_locator(storage.base_uri)}/{quote(key, safe='/')}"
    manifest: dict[str, Any] = {
        "version": 1,
        "generation": generation,
        "file_id": snapshot.file_id,
        "user_id": snapshot.user_id,
        "storage_key": key,
        "checksum": snapshot.checksum,
        "etag": snapshot.etag,
        "storage_backend": snapshot.storage_backend or storage.backend,
        "storage_uri": _locator(snapshot.storage_uri)
        if snapshot.storage_uri
        else expected_uri,
        "base_uri": _locator(storage.base_uri),
        "routing": _provider_routing(),
        "roots": [str(uploads), str(materialize), str(previews)],
        "local": [],
        "previews": None,
        "done": [],
    }
    # User containment is necessary even for task-less rows; an external
    # allowlist permits reading a source, never destroying it.
    raw_path = Path(os.path.abspath(Path(snapshot.storage_path).expanduser()))
    path = _canonical_path(raw_path, configured_uploads, uploads)
    owner_root = scoped_user_root(uploads, snapshot.user_id)
    configured_external = tuple(get_external_upload_dirs())
    external = tuple(_canonical_root(root) for root in configured_external)
    external_source = any(
        raw_path.is_relative_to(Path(os.path.abspath(root.expanduser())))
        for root in configured_external
    ) or any(path.is_relative_to(root) for root in external)
    if path.is_relative_to(owner_root) and not external_source:
        manifest["local"].append(
            capture_resource(
                path, root=uploads, generation=generation, checksum=checksum
            )
        )
        if not checksum:
            manifest["local"][-1]["uncertain"] = (
                "managed source has no SHA-256 evidence"
            )
    else:
        manifest["preserved_source"] = str(path)
    filename = Path(snapshot.filename).name
    cached_paths, materialization_uncertainty = _materialized_candidates(
        materialize, key, filename
    )
    if checksum:
        cached_paths.append(
            materialized_file_path(materialize, key, checksum, filename)
        )
    else:
        materialization_uncertainty = "materialization checksum unavailable"
    if materialization_uncertainty:
        manifest["uncertain_materialization"] = materialization_uncertainty
    for cached in cached_paths:
        if not any(resource["path"] == str(cached) for resource in manifest["local"]):
            manifest["local"].append(
                capture_resource(
                    cached, root=materialize, generation=generation, checksum=checksum
                )
            )
    for resource in tuple(manifest["local"]):
        target = Path(resource["path"])
        prefixes: tuple[str, ...]
        if Path(resource["root"]) == materialize:
            prefixes = (
                f".{target.name}.",
                atomic_copy_temp_prefix(target.name),
            )
        else:
            legacy = (
                f".{hashlib.sha256(str(snapshot.file_id).encode()).hexdigest()[:24]}."
                f"{target.name}."
            )
            current = atomic_copy_temp_prefix(target.name, owner=str(snapshot.file_id))
            prefixes = (legacy, "." + legacy, current, "." + current)
        prefixes = tuple(dict.fromkeys(prefixes))
        try:
            with _parent_descriptor(target, Path(resource["root"])) as parent:
                with os.scandir(parent) as entries:
                    names = (
                        entry.name
                        for entry in entries
                        if entry.name.startswith(prefixes)
                        and entry.name.endswith(".tmp")
                    )
                    for name in names:
                        manifest["local"].append(
                            capture_resource(
                                target.parent / name,
                                root=Path(resource["root"]),
                                generation=generation,
                            )
                        )
        except FileNotFoundError:
            continue
        except OSError:
            resource["uncertain"] = "temporary namespace is unverifiable"
            continue
    return manifest


def validate_configuration(manifest: dict[str, Any]) -> None:
    current = [
        str(_canonical_root(get_uploads_dir())),
        str(_canonical_root(get_file_materialize_dir())),
        str(_canonical_root(get_storage_root())),
    ]
    storage = get_user_file_storage(manifest["user_id"])
    uri = f"{_locator(storage.base_uri)}/{quote(manifest['storage_key'], safe='/')}"
    if (
        current != manifest["roots"]
        or storage.backend != manifest["storage_backend"]
        or _locator(storage.base_uri) != manifest["base_uri"]
        or _provider_routing() != manifest["routing"]
        or uri != manifest["storage_uri"]
    ):
        raise CleanupResourceUncertain("Cleanup storage configuration changed")
    # Validate ownership with the same scope boundary used by storage consumers.
    storage._scoped(manifest["storage_key"], strict=False)


def dispose_resource(resource: dict[str, Any], *, budget: CleanupWorkBudget) -> None:
    """Quarantine before deleting; an unexpected replacement is never unlinked.

    The quarantine locator is already durable. Exit after rename therefore
    leaves a retryable resource, including when the original path is absent.
    """
    if resource.get("preserved"):
        return
    if resource.get("uncertain"):
        raise CleanupResourceUncertain(resource["uncertain"])
    path, root = Path(resource["path"]), Path(resource["root"])
    try:
        with _parent_descriptor(path, root) as parent:
            if resource["parent"] is not None and (
                _identity(os.fstat(parent))[:3] != resource["parent"]
            ):
                raise CleanupResourceUncertain("Cleanup parent was replaced")
            quarantine = resource["quarantine"]
            try:
                info = os.stat(quarantine, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    return
                if not _matches_resource(info, resource["identity"]):
                    raise CleanupResourceUncertain("Cleanup source was replaced")
                if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                    raise CleanupResourceUncertain("Cleanup source became shared")
                os.rename(path.name, quarantine, src_dir_fd=parent, dst_dir_fd=parent)
                os.fsync(parent)
                info = os.stat(quarantine, dir_fd=parent, follow_symlinks=False)
            if not _matches_resource(info, resource["identity"]):
                raise CleanupResourceUncertain(
                    "Retained quarantine contains a replacement"
                )
            if stat.S_ISDIR(info.st_mode):
                dispose_directory(parent, quarantine, budget)
            else:
                if resource.get("checksum"):
                    descriptor = os.open(
                        quarantine, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent
                    )
                    with os.fdopen(descriptor, "rb") as source:
                        digest = hashlib.file_digest(source, "sha256").hexdigest()
                        if digest != resource["checksum"]:
                            raise CleanupResourceUncertain(
                                "Managed copy checksum changed"
                            )
                    if (
                        _identity(
                            os.stat(quarantine, dir_fd=parent, follow_symlinks=False)
                        )
                        != resource["identity"]
                    ):
                        raise CleanupResourceUncertain(
                            "Quarantine changed during verification"
                        )
                os.unlink(quarantine, dir_fd=parent)
            os.fsync(parent)
            # A replacement published while quarantining is not our resource.
            try:
                os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return
            raise CleanupResourceUncertain("Cleanup path has a replacement")
    except FileNotFoundError:
        # Verify both top-level locators before treating an interruption as done.
        try:
            with _parent_descriptor(path, root) as parent:
                if resource["parent"] is not None and (
                    _identity(os.fstat(parent))[:3] != resource["parent"]
                ):
                    raise CleanupResourceUncertain("Cleanup parent was replaced")
                for name in (path.name, resource["quarantine"]):
                    try:
                        os.stat(name, dir_fd=parent, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    raise CleanupResourceUncertain(
                        "Resource remains after interrupted disposal"
                    )
        except FileNotFoundError:
            return


def capture_previews(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(manifest["roots"][2])
    file_id = manifest["file_id"]
    if Path(file_id).name != file_id or file_id in {".", ".."}:
        raise CleanupResourceUncertain("Invalid preview owner identity")
    resources = []
    for name in ("pptx_pdf_cache", "svg_png_cache"):
        directory = root / name
        try:
            with _parent_descriptor(directory / "entry", root) as parent:
                names = os.listdir(parent)
        except FileNotFoundError:
            continue
        for entry in names:
            owned = (
                (entry.startswith(f"{file_id}.") and entry.endswith(".preview.pdf"))
                or (
                    entry.startswith(f"{file_id}.")
                    and ".preview.png" in entry
                    and (entry.endswith(".preview.png") or entry.endswith(".tmp"))
                )
                or (
                    entry.startswith(f"{file_id}.preview.pdf.")
                    and entry.endswith(".tmp")
                )
                or entry.startswith(f".{file_id}.preview-")
            )
            if owned:
                resources.append(
                    capture_resource(
                        directory / entry,
                        root=root,
                        generation=manifest["generation"],
                        allow_directory=entry.startswith(f".{file_id}.preview-"),
                    )
                )
    return resources
