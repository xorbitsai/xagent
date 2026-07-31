"""Tests for ``check_sandbox_static_readiness`` (app-startup sandbox gate).

Domain discipline under test: ``SANDBOX_VOLUMES`` (host-domain triples,
built with the same ``host_side_sources`` flag the runtime mount-building
path uses) and code mounts (also already host-domain) are compared as-is;
external upload dirs are backend-domain paths, folded then mapped through
the same ``SandboxPathMapper`` before the conflict check runs -- entirely
in the post-mapper host domain, over the combined triple set.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from xagent.sandbox.base import SandboxRuntimeConflictError
from xagent.web.sandbox_manager import check_sandbox_static_readiness


class _ProbeStub:
    """Minimal stand-in exposing only the attribute readiness consults."""

    def __init__(self, supports_runtime_spec: bool) -> None:
        self._supports = supports_runtime_spec

    async def _resolve_backend_probe(self) -> bool:
        return self._supports


@pytest.mark.asyncio
async def test_readiness_skipped_when_backend_does_not_support_reconciliation():
    """Legacy (Boxlite) backends never reconcile a spec; nothing to protect
    here even when SANDBOX_VOLUMES and code mounts would otherwise conflict."""
    with (
        patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "/foo:/guest1:ro"},
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/foo", "/guest2", "ro")],
        ),
    ):
        await check_sandbox_static_readiness(_ProbeStub(False))


@pytest.mark.asyncio
async def test_readiness_skipped_when_no_relevant_env_configured():
    """Code mounts alone are never checked when neither SANDBOX_VOLUMES nor
    XAGENT_EXTERNAL_UPLOAD_DIRS is configured -- the self-conflicting code
    mounts below would raise if the empty-env skip did not short-circuit
    before the triple set is ever built."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/foo", "/guest1", "ro"), ("/foo", "/guest2", "ro")],
        ),
    ):
        await check_sandbox_static_readiness(_ProbeStub(True))


@pytest.mark.asyncio
async def test_readiness_raises_on_host_conflict():
    """Two mounts sharing a host path but disagreeing on guest path raise."""
    with (
        patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "/foo:/guest1:ro"},
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/foo", "/guest2", "ro")],
        ),
    ):
        with pytest.raises(SandboxRuntimeConflictError):
            await check_sandbox_static_readiness(_ProbeStub(True))


@pytest.mark.asyncio
async def test_readiness_raises_on_guest_crash():
    """Two mounts sharing a guest path but disagreeing on host source raise
    at startup, instead of only when a task's create() reaches the daemon."""
    with (
        patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "/foo:/guest:ro"},
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/bar", "/guest", "ro")],
        ),
    ):
        with pytest.raises(SandboxRuntimeConflictError):
            await check_sandbox_static_readiness(_ProbeStub(True))


@pytest.mark.asyncio
async def test_readiness_allows_identical_triple_duplicated_across_sources():
    """The exact same (host, guest, mode) triple showing up from two
    different sources is a legal duplicate, not a conflict."""
    with (
        patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "/foo:/guest:ro"},
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/foo", "/guest", "ro")],
        ),
    ):
        await check_sandbox_static_readiness(_ProbeStub(True))


@pytest.mark.asyncio
async def test_readiness_groups_paths_the_backend_would_collapse():
    """``//foo`` and ``/foo`` are one mount point to the backend.

    ``posixpath.normpath`` keeps a leading ``//``, so normalizing with it
    alone would file these under two different host keys and let a real
    duplicate-mount conflict past the startup gate; the shared
    ``canonical_sandbox_path`` owner collapses the leading slash run the
    same way the desired spec does.
    """
    with (
        patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "//foo:/guest1:ro"},
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/foo", "/guest2", "ro")],
        ),
    ):
        with pytest.raises(SandboxRuntimeConflictError):
            await check_sandbox_static_readiness(_ProbeStub(True))


@pytest.mark.asyncio
async def test_readiness_checks_folded_external_dirs_against_host_domain(
    tmp_path: Path,
):
    """External upload dirs are folded, then mapped through the same
    ``SandboxPathMapper``/``host_side_sources`` combination the runtime
    mount-building path uses, and checked in that same host domain.

    Here the external dir sits outside the storage root, so the mapper
    leaves its path unchanged (identity mapping); a SANDBOX_VOLUMES entry
    claiming the same host path under a different guest path must still be
    caught as a host conflict once both are in host domain.
    """
    backend_storage_root = tmp_path / "backend" / ".xagent"
    host_storage_root = tmp_path / "host" / ".xagent"
    external_dir = tmp_path / "shared" / "kb"
    external_dir.mkdir(parents=True)

    with (
        patch.dict(
            "os.environ",
            {
                "SANDBOX_VOLUMES": f"{external_dir}:/guest-other:ro",
                "XAGENT_EXTERNAL_UPLOAD_DIRS": str(external_dir),
                "XAGENT_STORAGE_ROOT": str(backend_storage_root),
                "XAGENT_SANDBOX_HOST_STORAGE_ROOT": str(host_storage_root),
            },
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[],
        ),
    ):
        with pytest.raises(SandboxRuntimeConflictError):
            await check_sandbox_static_readiness(_ProbeStub(True))


@pytest.mark.asyncio
async def test_readiness_allows_non_conflicting_external_dir(tmp_path: Path):
    """A well-formed deployment (disjoint external dir, no drift) passes."""
    backend_storage_root = tmp_path / "backend" / ".xagent"
    host_storage_root = tmp_path / "host" / ".xagent"
    external_dir = tmp_path / "shared" / "kb"
    external_dir.mkdir(parents=True)

    with (
        patch.dict(
            "os.environ",
            {
                "XAGENT_EXTERNAL_UPLOAD_DIRS": str(external_dir),
                "XAGENT_STORAGE_ROOT": str(backend_storage_root),
                "XAGENT_SANDBOX_HOST_STORAGE_ROOT": str(host_storage_root),
            },
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/repo/src", "/app/src", "ro")],
        ),
    ):
        await check_sandbox_static_readiness(_ProbeStub(True))


@pytest.mark.asyncio
async def test_readiness_absolutizes_relative_external_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A relative ``XAGENT_EXTERNAL_UPLOAD_DIRS`` entry is checked as the
    directory it actually names.

    ``get_external_upload_dirs`` accepts any spelling that resolves to an
    existing directory, including one relative to the process cwd, while
    the mount-path contract only compares absolute paths. Absolutizing
    through the same owner the runtime build path uses is what lets the
    startup gate see this entry and the equivalent absolute
    ``SANDBOX_VOLUMES`` spelling as the same host path -- and keeps a legal
    relative configuration from failing startup outright.
    """
    external_dir = tmp_path / "shared-kb"
    external_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    with (
        patch.dict(
            "os.environ",
            {
                "SANDBOX_VOLUMES": f"{Path.cwd() / 'shared-kb'}:/guest-other:ro",
                "XAGENT_EXTERNAL_UPLOAD_DIRS": "shared-kb",
            },
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[],
        ),
    ):
        with pytest.raises(SandboxRuntimeConflictError):
            await check_sandbox_static_readiness(_ProbeStub(True))
