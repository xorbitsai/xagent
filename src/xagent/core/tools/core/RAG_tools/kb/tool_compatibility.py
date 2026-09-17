"""Tool compatibility facade for KB agent/tool entry points."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

from ..core.schemas import CollectionInfo, IngestionConfig
from .async_utils import maybe_await
from .models import KBStorageBackend
from .pipeline_compatibility import KB_STORAGE_METADATA_KEY

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ......web.tools.config import WebToolConfig
    from ....adapters.vibe.base import AbstractBaseTool
    from ....adapters.vibe.config import BaseToolConfig
    from ....adapters.vibe.document_search import (
        KnowledgeSearchTool,
        ListKnowledgeBasesTool,
    )
    from ...document_search import (
        KnowledgeSearchArgs,
        KnowledgeSearchResult,
        ListKnowledgeBasesArgs,
        ListKnowledgeBasesResult,
    )
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade


class KBToolCompatibilityFacade:
    """Compatibility boundary for KB-facing agent and tool surfaces.

    Tool modules keep their historical imports, factories, names, schemas, and
    sync/async behavior while this facade routes their KB semantics through the
    coordinator-owned management, pipeline, and storage boundaries.
    """

    def __init__(
        self,
        coordinator: KBCoordinator | None = None,
        storage_shim: KBStorageShimCompatibilityFacade | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._storage_shim = storage_shim

    def _active_storage_shim(self) -> KBStorageShimCompatibilityFacade | None:
        if self._storage_shim is not None:
            return self._storage_shim
        if self._coordinator is not None:
            return self._coordinator.storage_shim
        return None

    @contextmanager
    def _storage_context(self) -> Iterator[None]:
        storage_shim = self._active_storage_shim()
        if storage_shim is None:
            yield
            return

        from ..storage.factory import bind_storage_shim_for_current_context

        with bind_storage_shim_for_current_context(storage_shim):
            yield

    async def list_knowledge_bases(
        self,
        tool_args: ListKnowledgeBasesArgs,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        governing_team_id: Optional[int] = None,
    ) -> ListKnowledgeBasesResult:
        from ... import document_search as core_document_search

        with self._storage_context():
            return await core_document_search._list_knowledge_bases_impl(
                tool_args,
                user_id=user_id,
                is_admin=is_admin,
                governing_team_id=governing_team_id,
            )

    async def find_missing_knowledge_bases(
        self,
        knowledge_bases: list[str],
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> list[str]:
        from ... import document_search as core_document_search

        with self._storage_context():
            return await core_document_search._find_missing_knowledge_bases_impl(
                knowledge_bases,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def search_knowledge_base(
        self,
        tool_args: KnowledgeSearchArgs,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        governing_team_id: Optional[int] = None,
        agent_creator_user_id: Optional[int] = None,
        declared_knowledge_bases: Optional[list[str]] = None,
    ) -> KnowledgeSearchResult:
        from ... import document_search as core_document_search

        with self._storage_context():
            return await core_document_search._search_knowledge_base_impl(
                tool_args,
                user_id=user_id,
                is_admin=is_admin,
                governing_team_id=governing_team_id,
                agent_creator_user_id=agent_creator_user_id,
                declared_knowledge_bases=declared_knowledge_bases,
            )

    def get_list_knowledge_bases_tool(
        self,
        allowed_collections: Optional[list[str]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        governing_team_id: Optional[int] = None,
    ) -> ListKnowledgeBasesTool:
        from ....adapters.vibe import document_search as vibe_document_search

        return vibe_document_search._get_list_knowledge_bases_tool_impl(
            allowed_collections=allowed_collections,
            user_id=user_id,
            is_admin=is_admin,
            governing_team_id=governing_team_id,
        )

    def get_knowledge_search_tool(
        self,
        embedding_model_id: Optional[str] = None,
        rerank_model_id: Optional[str] = None,
        allowed_collections: Optional[list[str]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        governing_team_id: Optional[int] = None,
        agent_creator_user_id: Optional[int] = None,
        declared_knowledge_bases: Optional[list[str]] = None,
    ) -> KnowledgeSearchTool:
        from ....adapters.vibe import document_search as vibe_document_search

        return vibe_document_search._get_knowledge_search_tool_impl(
            embedding_model_id=embedding_model_id,
            rerank_model_id=rerank_model_id,
            allowed_collections=allowed_collections,
            user_id=user_id,
            is_admin=is_admin,
            governing_team_id=governing_team_id,
            agent_creator_user_id=agent_creator_user_id,
            declared_knowledge_bases=declared_knowledge_bases,
        )

    async def create_knowledge_tools(
        self,
        config: BaseToolConfig,
    ) -> list[Any]:
        from ....adapters.vibe import knowledge_tools

        return await knowledge_tools._create_knowledge_tools_impl(config)

    async def create_knowledge_base_from_file(
        self,
        args: Mapping[str, Any],
        *,
        user_id: int,
        is_admin: bool = False,
    ) -> Any:
        from ....adapters.vibe import file_ingestion_tool

        with self._storage_context():
            return await file_ingestion_tool._create_knowledge_base_from_file_impl(
                args,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def create_knowledge_base_from_url(
        self,
        args: Mapping[str, Any],
        *,
        user_id: int,
        is_admin: bool = False,
    ) -> Any:
        from ....adapters.vibe import web_ingestion_tool

        with self._storage_context():
            return await web_ingestion_tool._create_knowledge_base_from_url_impl(
                args,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def create_file_ingestion_tools(
        self,
        config: WebToolConfig,
    ) -> list[AbstractBaseTool]:
        from ....adapters.vibe import file_ingestion_tool

        return await file_ingestion_tool._create_file_ingestion_tools_impl(config)

    async def create_web_ingestion_tools(
        self,
        config: WebToolConfig,
    ) -> list[AbstractBaseTool]:
        from ....adapters.vibe import web_ingestion_tool

        return await web_ingestion_tool._create_web_ingestion_tools_impl(config)

    async def prepare_agent_collection(
        self,
        *,
        collection_name: str,
    ) -> str:
        from ....adapters.vibe import agent_kb_service

        with self._storage_context():
            return await agent_kb_service._prepare_collection_impl(
                collection_name=collection_name,
            )

    async def publish_agent_collection(
        self,
        *,
        collection_name: str,
        ingestion_config: IngestionConfig,
        user_id: int,
        collection_existed_before: bool = False,
    ) -> None:
        from ....adapters.vibe import agent_kb_service

        with self._storage_context():
            # The backend binding lands with the config, not before the ingest:
            # a metadata row written up front survives a failed import and makes
            # the name unusable while staying invisible to its owner.
            await self.ensure_agent_collection_backend_binding(collection_name)
            await agent_kb_service._publish_collection_impl(
                collection_name=collection_name,
                ingestion_config=ingestion_config,
                user_id=user_id,
                collection_existed_before=collection_existed_before,
            )

    async def agent_collection_exists(self, collection: str) -> bool:
        """Whether the collection already exists before an agent import starts."""
        from ..storage.factory import get_metadata_store

        with self._storage_context():
            try:
                return await get_metadata_store().get_collection(collection) is not None
            except ValueError:
                return False
            except Exception as exc:  # noqa: BLE001
                # Unknown state must not authorize the cleanup below.
                logger.warning(
                    "Could not read collection %s before agent ingest: %s",
                    collection,
                    exc,
                )
                return True

    async def cleanup_failed_agent_collection(
        self,
        collection: str,
        *,
        user_id: int,
    ) -> None:
        """Drop the metadata row the ingest pipeline wrote for a failed import.

        The row is invisible to its non-admin owner but still makes every route
        answer 409 for the name, so leaving it behind burns the name silently.
        """
        from ..storage.factory import get_metadata_store, get_vector_index_store

        with self._storage_context():
            try:
                # Owner-scoped on both sides, matching the delete below: an
                # admin-wide read would see another tenant's documents under the
                # same name and abort, leaving this import's orphaned row to
                # block the name forever. What this read cannot see is a sibling
                # whose documents landed but whose config is not published yet;
                # the delete's own "no tenant holds a config" check is the only
                # guard left there, and that window is tracked in issue #1242.
                records = get_vector_index_store().list_document_records(
                    collection_name=collection,
                    user_id=user_id,
                    is_admin=False,
                    max_results=1,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping failed agent import cleanup for %s: %s", collection, exc
                )
                return
            if records:
                return

            # Same call the API paths use: owner-scoped, and it only drops the
            # metadata row once no tenant holds a config for that name, so a
            # sibling that published under the same name is never disturbed.
            delete_metadata = getattr(
                get_metadata_store(), "delete_collection_metadata", None
            )
            if not callable(delete_metadata):
                return
            deleted = await maybe_await(
                delete_metadata(
                    collection_name=collection,
                    user_id=user_id,
                    is_admin=False,
                    delete_orphaned_metadata=True,
                )
            )
            logger.info(
                "Cleaned metadata left by a failed agent import for %s: %s",
                collection,
                deleted,
            )

    async def refresh_agent_collection_metadata(
        self,
        collection_name: str,
        *,
        user_id: int,
        is_admin: bool = False,
    ) -> None:
        from ....adapters.vibe import agent_kb_service

        with self._storage_context():
            await agent_kb_service._refresh_collection_metadata_impl(
                collection_name=collection_name,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def ensure_agent_collection_backend_binding(
        self,
        collection: str,
    ) -> CollectionInfo:
        """Create a collection-level backend binding for agent/tool-created KBs."""
        from ..storage.factory import get_metadata_store

        with self._storage_context():
            metadata_store = get_metadata_store()
            try:
                collection_info = await metadata_store.get_collection(collection)
            except ValueError:
                collection_info = CollectionInfo(name=collection)

            extra_metadata = dict(collection_info.extra_metadata or {})
            if extra_metadata.get(KB_STORAGE_METADATA_KEY) is not None:
                return collection_info

            extra_metadata[KB_STORAGE_METADATA_KEY] = {
                "backend": KBStorageBackend.LANCEDB.value
            }
            updated_collection = collection_info.model_copy(
                update={"extra_metadata": extra_metadata}
            )
            await metadata_store.save_collection(updated_collection)
            return updated_collection
