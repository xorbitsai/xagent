import logging

from ...core.RAG_tools.core.schemas import IngestionConfig

logger = logging.getLogger(__name__)


class AgentKnowledgeBaseService:
    """Shared collection setup/refresh flow for agent-triggered KB creation."""

    def __init__(self, user_id: int, is_admin: bool = False) -> None:
        self.user_id = user_id
        self.is_admin = is_admin

    async def prepare_collection(
        self,
        collection_name: str,
        ingestion_config: IngestionConfig,
    ) -> str:
        from .....web.config import sanitize_path_component
        from ...core.RAG_tools.storage.factory import get_metadata_store

        safe_collection = sanitize_path_component(collection_name, "collection")
        metadata_store = get_metadata_store()

        try:
            await metadata_store.save_collection_config(
                collection=safe_collection,
                config_json=ingestion_config.model_dump_json(exclude_unset=True),
                user_id=self.user_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to save collection config for agent knowledge base %s: %s",
                safe_collection,
                exc,
            )

        return safe_collection

    async def refresh_collection_metadata(self, collection_name: str) -> None:
        from ...core.RAG_tools.management.collections import list_collections

        try:
            # Refresh metadata cache so agent-created KBs are visible like API-created ones.
            await list_collections(
                user_id=self.user_id,
                is_admin=self.is_admin,
                force_realtime=True,
            )
        except Exception as exc:
            logger.warning(
                "Failed to refresh collection metadata after agent ingestion for %s: %s",
                collection_name,
                exc,
            )
