"""Boundary regressions for complete uploaded-file cleanup resources."""

# The imported parameterized fixture is intentionally shadowed by test arguments.
# ruff: noqa: F401, F811

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.web.services.test_complete_upload_cleanup import (
    collect,
    completed,
)
from tests.web.services.test_complete_upload_cleanup import (
    lifecycle as cleanup_lifecycle,
)
from tests.web.services.test_complete_upload_cleanup import (
    recover,
)
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.storage import (
    FsspecFileStorage,
    atomic_copy_temp_prefix,
)
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services import uploaded_file_cleanup_resources as resources
from xagent.web.services.managed_file_ref import ManagedFileRef


def _rename_source(cleanup_lifecycle, name: str) -> Path:
    sessions, source, *_ = cleanup_lifecycle
    target = source.with_name(name)
    source.rename(target)
    with sessions.begin() as db:
        row = db.query(UploadedFile).one()
        row.storage_path = str(target)
        row.filename = name
    return target


def test_cleanup_follows_a_symlinked_configured_upload_root(
    cleanup_lifecycle, monkeypatch
):
    sessions, source, materialized, previews, key, _ = cleanup_lifecycle
    alias = source.parents[2] / "uploads-alias"
    alias.symlink_to(source.parents[1], target_is_directory=True)
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(alias))
    with sessions.begin() as db:
        db.query(UploadedFile).one().storage_path = str(alias / "user_1" / source.name)

    assert collect(cleanup_lifecycle).deleted == 1

    assert not source.exists()
    assert not materialized.exists()
    assert not any(path.exists() for path in previews)
    assert not get_unscoped_file_storage().exists(key)
    with sessions() as db:
        assert db.query(UploadedFile).count() == 0


def test_cleanup_preserves_a_configured_external_root_nested_under_uploads(
    cleanup_lifecycle, monkeypatch
):
    sessions, source, materialized, previews, key, _ = cleanup_lifecycle
    outside = source.parents[2] / "configured-external"
    outside.mkdir()
    external = source.parent / "external"
    external.symlink_to(outside, target_is_directory=True)
    target = outside / source.name
    source.rename(target)
    monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", str(external))
    with sessions.begin() as db:
        db.query(UploadedFile).one().storage_path = str(external / source.name)

    assert collect(cleanup_lifecycle).deleted == 1

    assert target.read_bytes()
    assert not materialized.exists()
    assert not any(path.exists() for path in previews)
    assert not get_unscoped_file_storage().exists(key)
    with sessions() as db:
        assert db.query(UploadedFile).count() == 0


def test_cleanup_retains_a_handle_for_a_symlink_beneath_the_managed_root(
    cleanup_lifecycle,
):
    sessions, source, _, _, key, _ = cleanup_lifecycle
    outside = source.parents[2] / "outside"
    outside.mkdir()
    target = outside / source.name
    source.rename(target)
    link = source.parent / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with sessions.begin() as db:
        db.query(UploadedFile).one().storage_path = str(link / source.name)

    assert collect(cleanup_lifecycle).deleted == 0

    assert target.exists()
    assert get_unscoped_file_storage().exists(key)
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.storage_status == "compensating"
        assert row.cleanup_manifest["done"] == []
        assert row.cleanup_manifest["local"][0]["uncertain"]


@pytest.mark.parametrize(
    "name",
    ["a" * 246 + ".txt", "界" * 80 + ".txt"],
    ids=["ascii-250-bytes", "cjk-244-bytes"],
)
def test_cleanup_quarantines_maximum_length_basenames(
    cleanup_lifecycle, monkeypatch, name
):
    target = _rename_source(cleanup_lifecycle, name)
    materialized = cleanup_lifecycle[2]
    materialized_target = materialized.with_name(name)
    materialized.rename(materialized_target)
    file_id = cleanup_lifecycle[-1]
    source_prefix = atomic_copy_temp_prefix(name, owner=file_id)
    interrupted = [
        target.with_name(f"{source_prefix}interrupted.tmp"),
        target.with_name(f".{source_prefix}nested-interrupted.tmp"),
        materialized_target.with_name(
            f"{atomic_copy_temp_prefix(name)}interrupted.tmp"
        ),
    ]
    for path in interrupted:
        path.write_bytes(b"interrupted copy")

    unlink = resources.os.unlink
    failed = False

    def interrupt_quarantine(path, *args, **kwargs):
        nonlocal failed
        if not failed and str(path).startswith(".cleanup-"):
            failed = True
            raise OSError("interrupted after quarantine")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as interruption:
        interruption.setattr(resources.os, "unlink", interrupt_quarantine)
        assert collect(cleanup_lifecycle).deleted == 0

    assert recover(cleanup_lifecycle).deleted == 1

    assert not target.exists()
    assert not materialized_target.exists()
    assert not any(path.exists() for path in interrupted)
    completed(cleanup_lifecycle)


@pytest.mark.parametrize(
    "name",
    ["r" * 246 + ".txt", "还" * 80 + ".txt"],
    ids=["ascii-250-bytes", "cjk-244-bytes"],
)
def test_ensure_local_restores_maximum_length_basenames(cleanup_lifecycle, name):
    sessions, *_ = cleanup_lifecycle
    target = _rename_source(cleanup_lifecycle, name)
    target.unlink()

    with sessions() as db:
        restored = ManagedFileRef(db.query(UploadedFile).one()).ensure_local()

    assert restored == target
    assert restored.read_bytes()


@pytest.mark.parametrize(
    "name",
    ["m" * 246 + ".txt", "料" * 80 + ".txt"],
    ids=["ascii-250-bytes", "cjk-244-bytes"],
)
def test_storage_materializes_maximum_length_basenames(cleanup_lifecycle, name):
    *_, key, _ = cleanup_lifecycle

    target = get_unscoped_file_storage().materialize(key, name)

    assert target.name == name
    assert target.read_bytes()


def test_cleanup_captures_the_storage_backends_materialized_locator(
    cleanup_lifecycle, monkeypatch
):
    sessions, _, materialized, _, _, _ = cleanup_lifecycle
    backend_hash = hashlib.sha256(b"backend content hash").hexdigest()
    actual = materialized.parent.parent / backend_hash / materialized.name
    actual.parent.mkdir(parents=True)
    materialized.rename(actual)
    monkeypatch.setattr(
        FsspecFileStorage,
        "content_hash",
        lambda _self, _key: backend_hash,
    )

    assert collect(cleanup_lifecycle).deleted == 1

    assert not actual.exists()
    with sessions() as db:
        assert db.query(UploadedFile).count() == 0


def test_cleanup_does_not_probe_provider_for_locally_identifiable_copies(
    cleanup_lifecycle, monkeypatch
):
    def unavailable(_self, _key):
        raise AssertionError("local manifest capture must not probe the provider")

    monkeypatch.setattr(FsspecFileStorage, "content_hash", unavailable)

    assert collect(cleanup_lifecycle).deleted == 1
    completed(cleanup_lifecycle)
