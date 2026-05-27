"""
Web services module.
"""

from .chat_history_service import (
    load_task_transcript,
    persist_assistant_message,
    persist_user_message,
)
from .kb_collection_service import (
    CollectionPhysicalDeleteResult,
    CollectionPhysicalRenameResult,
    delete_collection_physical_dir,
    delete_collection_uploaded_files,
    rename_collection_storage,
)
from .kb_file_service import (
    build_uploaded_filename_map,
    delete_uploaded_file_if_orphaned,
    get_document_record_file_id,
    list_documents_for_user,
    resolve_document_filename,
    upsert_uploaded_file_record,
)
from .kb_ingest_targets import (
    admit_kb_ingest_target,
    is_latest_kb_ingest_generation,
    release_kb_ingest_target_generation,
    tombstone_kb_ingest_target,
    tombstone_kb_ingest_targets_for_collection,
)
from .model_service import (
    get_default_image_edit_model,
    get_default_image_generate_model,
    get_default_model,
    get_default_vision_model,
)
from .task_execution_context_service import (
    load_task_execution_context_messages,
    load_task_execution_recovery_state,
    summarize_tool_event,
)
from .uploaded_file_store import UploadedFileStore

__all__ = [
    "load_task_transcript",
    "load_task_execution_context_messages",
    "load_task_execution_recovery_state",
    "summarize_tool_event",
    "persist_assistant_message",
    "persist_user_message",
    "CollectionPhysicalDeleteResult",
    "CollectionPhysicalRenameResult",
    "delete_collection_physical_dir",
    "delete_collection_uploaded_files",
    "rename_collection_storage",
    "upsert_uploaded_file_record",
    "list_documents_for_user",
    "build_uploaded_filename_map",
    "get_document_record_file_id",
    "resolve_document_filename",
    "delete_uploaded_file_if_orphaned",
    "admit_kb_ingest_target",
    "is_latest_kb_ingest_generation",
    "release_kb_ingest_target_generation",
    "tombstone_kb_ingest_target",
    "tombstone_kb_ingest_targets_for_collection",
    "get_default_model",
    "get_default_vision_model",
    "get_default_image_generate_model",
    "get_default_image_edit_model",
    "UploadedFileStore",
]
