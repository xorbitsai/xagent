"""Uploaded-file and physical KB compatibility facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Union

from ..storage.contracts import DocumentRecord

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.services.kb_collection_service import (
        CollectionPhysicalDeleteResult,
        CollectionPhysicalRenameResult,
    )
    from xagent.web.services.kb_file_service import (
        FileCompensationResult,
        UploadedFileRefreshSnapshot,
    )


class KBFileCompatibilityFacade:
    """Compatibility boundary for legacy uploaded-file and physical helpers.

    The current implementations intentionally remain in the web service modules
    so transaction ownership, cache invalidation, and filesystem semantics stay
    unchanged. This facade gives coordinator-owned callers a stable semantic
    entry point while preserving existing public helper behavior.
    """

    def upsert_uploaded_file_record(
        self,
        db: Session,
        *,
        user_id: Optional[int],
        filename: str,
        storage_path: Path,
        mime_type: Optional[str],
        file_size: int,
        file_id: Optional[str] = None,
    ) -> UploadedFile:
        from xagent.web.services.kb_file_service import (
            _upsert_uploaded_file_record_impl,
        )

        return _upsert_uploaded_file_record_impl(
            db,
            user_id=user_id,
            filename=filename,
            storage_path=storage_path,
            mime_type=mime_type,
            file_size=file_size,
            file_id=file_id,
        )

    def list_documents_for_user(
        self,
        *,
        user_id: Optional[int] = None,
        is_admin: bool,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from xagent.web.services.kb_file_service import _list_documents_for_user_impl

        return _list_documents_for_user_impl(
            user_id=user_id,
            is_admin=is_admin,
            collection_name=collection_name,
        )

    def find_referenced_file_ids(self, file_ids: Iterable[str]) -> set[str]:
        from xagent.web.services.kb_file_service import _find_referenced_file_ids_impl

        return _find_referenced_file_ids_impl(file_ids)

    def build_uploaded_filename_map(
        self, db: Session, *, user_id: Optional[int], file_ids: List[str]
    ) -> Dict[str, str]:
        from xagent.web.services.kb_file_service import (
            _build_uploaded_filename_map_impl,
        )

        return _build_uploaded_filename_map_impl(
            db,
            user_id=user_id,
            file_ids=file_ids,
        )

    def get_document_record_file_id(
        self,
        record: Union[Dict[str, Any], DocumentRecord],
    ) -> Optional[str]:
        from xagent.web.services.kb_file_service import (
            _get_document_record_file_id_impl,
        )

        return _get_document_record_file_id_impl(record)

    def resolve_document_filename(
        self,
        record: Union[Dict[str, Any], DocumentRecord],
        filename_map: Dict[str, str],
    ) -> Optional[str]:
        from xagent.web.services.kb_file_service import (
            _resolve_document_filename_impl,
        )

        return _resolve_document_filename_impl(record, filename_map)

    def delete_uploaded_file_if_orphaned(
        self,
        db: Session,
        *,
        file_id: str,
        user_id: Optional[int],
        remaining_file_ids: set[str],
    ) -> bool:
        from xagent.web.services.kb_file_service import (
            _delete_uploaded_file_if_orphaned_impl,
        )

        return _delete_uploaded_file_if_orphaned_impl(
            db,
            file_id=file_id,
            user_id=user_id,
            remaining_file_ids=remaining_file_ids,
        )

    def list_collection_uploaded_file_owner_ids(
        self,
        db: Session,
        *,
        collection_name: str,
    ) -> Set[int]:
        from xagent.web.services.kb_collection_service import (
            _list_collection_uploaded_file_owner_ids_impl,
        )

        return _list_collection_uploaded_file_owner_ids_impl(
            db,
            collection_name=collection_name,
        )

    def delete_collection_physical_dir(
        self,
        *,
        user_id: int,
        collection_name: str,
    ) -> CollectionPhysicalDeleteResult:
        from xagent.web.services.kb_collection_service import (
            _delete_collection_physical_dir_impl,
        )

        return _delete_collection_physical_dir_impl(
            user_id=user_id,
            collection_name=collection_name,
        )

    def delete_collection_uploaded_files(
        self,
        db: Session,
        *,
        user_id: int,
        collection_file_ids: Set[str],
        remaining_file_ids: Set[str],
        collection_dir: Optional[Path],
    ) -> int:
        from xagent.web.services.kb_collection_service import (
            _delete_collection_uploaded_files_impl,
        )

        return _delete_collection_uploaded_files_impl(
            db,
            user_id=user_id,
            collection_file_ids=collection_file_ids,
            remaining_file_ids=remaining_file_ids,
            collection_dir=collection_dir,
        )

    def rename_collection_storage(
        self,
        db: Session,
        *,
        user_id: int,
        old_collection_name: str,
        new_collection_name: str,
        collection_file_ids: Set[str],
    ) -> CollectionPhysicalRenameResult:
        from xagent.web.services.kb_collection_service import (
            _rename_collection_storage_impl,
        )

        return _rename_collection_storage_impl(
            db,
            user_id=user_id,
            old_collection_name=old_collection_name,
            new_collection_name=new_collection_name,
            collection_file_ids=collection_file_ids,
        )

    def aggregate_uploaded_file_statuses(
        self,
        *,
        file_ids: List[str],
        user_id: int,
        is_admin: bool,
        use_cache: bool = True,
    ) -> Dict[str, str]:
        from xagent.web.services.kb_file_service import (
            _aggregate_uploaded_file_statuses_impl,
        )

        return _aggregate_uploaded_file_statuses_impl(
            file_ids=file_ids,
            user_id=user_id,
            is_admin=is_admin,
            use_cache=use_cache,
        )

    def reconcile_uploaded_files(
        self,
        db: Session,
        *,
        user_id: int,
        is_admin: bool,
        stale_ttl_hours: int = 24 * 7,
        delete_stale: bool = True,
        deletable_statuses: Optional[set[str]] = None,
    ) -> Dict[str, int]:
        from xagent.web.services.kb_file_service import (
            _reconcile_uploaded_files_impl,
        )

        return _reconcile_uploaded_files_impl(
            db,
            user_id=user_id,
            is_admin=is_admin,
            stale_ttl_hours=stale_ttl_hours,
            delete_stale=delete_stale,
            deletable_statuses=deletable_statuses,
        )

    def cleanup_local_copied_file(
        self,
        *,
        file_path: Path,
        local_root: Optional[Path] = None,
    ) -> FileCompensationResult:
        from xagent.web.services.kb_file_service import (
            _cleanup_local_copied_file_impl,
        )

        return _cleanup_local_copied_file_impl(
            file_path=file_path,
            local_root=local_root,
        )

    def capture_uploaded_file_refresh_snapshot(
        self,
        file_record: UploadedFile,
        *,
        backup_path: Optional[Path] = None,
        reindex_marker_applied: bool = False,
    ) -> UploadedFileRefreshSnapshot:
        from xagent.web.services.kb_file_service import (
            _capture_uploaded_file_refresh_snapshot_impl,
        )

        return _capture_uploaded_file_refresh_snapshot_impl(
            file_record,
            backup_path=backup_path,
            reindex_marker_applied=reindex_marker_applied,
        )

    def restore_uploaded_file_refresh_snapshot(
        self,
        db: Session,
        snapshot: UploadedFileRefreshSnapshot,
    ) -> FileCompensationResult:
        from xagent.web.services.kb_file_service import (
            _restore_uploaded_file_refresh_snapshot_impl,
        )

        return _restore_uploaded_file_refresh_snapshot_impl(db, snapshot)
