from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.managed_file_ref import (
    ManagedFileRef,
    create_uploaded_file_from_local_path,
    ensure_uploaded_file_local_path,
)


def test_create_uploaded_file_from_local_path_stores_durable_object(
    monkeypatch, tmp_path
):
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))
    get_unscoped_file_storage.cache_clear()

    source = tmp_path / "uploads" / "input.txt"
    source.parent.mkdir()
    source.write_text("hello object storage", encoding="utf-8")

    file_record = create_uploaded_file_from_local_path(
        local_path=source,
        user_id=7,
        filename="input.txt",
        file_id="file-123",
        task_id=42,
        mime_type="text/plain",
    )

    assert file_record.storage_path == str(source)
    assert file_record.storage_status == "available"
    assert file_record.storage_backend == "file"
    assert file_record.storage_key == "users/7/uploads/file-123/input.txt"
    assert file_record.storage_uri is not None
    assert file_record.checksum is not None
    assert file_record.file_size == len("hello object storage")

    source.unlink()

    materialized = ManagedFileRef(file_record).materialize()
    assert materialized.is_relative_to(tmp_path / "materialized")
    assert materialized.name == "input.txt"
    assert materialized.read_text(encoding="utf-8") == "hello object storage"


def test_managed_file_ref_materialize_prefers_existing_local_path(tmp_path):
    source = tmp_path / "still-local.txt"
    source.write_text("local copy", encoding="utf-8")

    file_record = UploadedFile(
        file_id="file-456",
        user_id=7,
        filename="still-local.txt",
        storage_path=str(source),
        storage_backend="file",
        storage_key="users/7/uploads/file-456/still-local.txt",
        storage_uri="file:///unused",
        storage_status="available",
        mime_type="text/plain",
        file_size=10,
    )

    assert ManagedFileRef(file_record).materialize() == source


def test_ensure_uploaded_file_local_path_restores_original_path(monkeypatch, tmp_path):
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    source = tmp_path / "uploads" / "restored.txt"
    source.parent.mkdir()
    source.write_text("restore me", encoding="utf-8")
    file_record = create_uploaded_file_from_local_path(
        local_path=source,
        user_id=7,
        filename="restored.txt",
        file_id="file-789",
        mime_type="text/plain",
    )
    source.unlink()

    restored = ensure_uploaded_file_local_path(file_record)

    assert restored == source
    assert source.read_text(encoding="utf-8") == "restore me"


def test_create_uploaded_file_from_local_path_accepts_custom_storage_key(
    monkeypatch, tmp_path
):
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()

    source = tmp_path / "workspace" / "output" / "report.txt"
    source.parent.mkdir(parents=True)
    source.write_text("generated output", encoding="utf-8")

    file_record = create_uploaded_file_from_local_path(
        local_path=source,
        user_id=7,
        filename="report.txt",
        file_id="file-output",
        task_id=42,
        storage_key="users/7/tasks/42/outputs/file-output/output/report.txt",
    )

    assert file_record.storage_key == (
        "users/7/tasks/42/outputs/file-output/output/report.txt"
    )
