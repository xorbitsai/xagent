import logging
from typing import TYPE_CHECKING

from ...core.RAG_tools.core.schemas import IngestionConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...core.RAG_tools.kb import KBToolCompatibilityFacade


class AgentKnowledgeBaseError(RuntimeError):
    """Raised when agent-triggered knowledge base setup cannot be completed."""


def _get_tool_compatibility_facade() -> "KBToolCompatibilityFacade":
    """Return the coordinator-owned KB tool compatibility facade."""
    from ...core.RAG_tools.kb import get_kb_coordinator

    return get_kb_coordinator().tool_compatibility


class AgentKnowledgeBaseService:
    """Shared collection setup/refresh flow for agent-triggered KB creation."""

    def __init__(self, user_id: int, is_admin: bool = False) -> None:
        self.user_id = user_id
        self.is_admin = is_admin

    async def prepare_collection(self, collection_name: str) -> str:
        """Resolve the target name. Writes nothing, so a failure leaves nothing."""
        return await _get_tool_compatibility_facade().prepare_agent_collection(
            collection_name=collection_name,
        )

    async def collection_exists(self, collection_name: str) -> bool:
        return await _get_tool_compatibility_facade().agent_collection_exists(
            collection_name
        )

    async def cleanup_failed_collection(self, collection_name: str) -> None:
        """Drop the metadata row the pipeline wrote, so the name stays reusable."""
        await _get_tool_compatibility_facade().cleanup_failed_agent_collection(
            collection_name,
            user_id=self.user_id,
        )

    async def publish_collection(
        self,
        collection_name: str,
        ingestion_config: IngestionConfig,
        collection_existed_before: bool = False,
    ) -> None:
        """Make the knowledge base visible; call only once it holds documents."""
        await _get_tool_compatibility_facade().publish_agent_collection(
            collection_name=collection_name,
            ingestion_config=ingestion_config,
            user_id=self.user_id,
            collection_existed_before=collection_existed_before,
        )

    async def refresh_collection_metadata(self, collection_name: str) -> None:
        await _get_tool_compatibility_facade().refresh_agent_collection_metadata(
            collection_name,
            user_id=self.user_id,
            is_admin=self.is_admin,
        )


async def _prepare_collection_impl(*, collection_name: str) -> str:
    """Resolve the target collection name without publishing it.

    The config row is what makes a knowledge base appear in the list, so it is
    written by :func:`_publish_collection_impl` once the ingest produced
    documents. Writing it here left a failed ingest permanently visible and empty.
    """
    from .....web.config import sanitize_path_component

    return sanitize_path_component(collection_name, "collection")


async def _publish_collection_impl(
    *,
    collection_name: str,
    ingestion_config: IngestionConfig,
    user_id: int,
    collection_existed_before: bool = False,
) -> None:
    from ...core.RAG_tools.kb.config_merge import merge_collection_config_json
    from ...core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    config_json = ingestion_config.model_dump_json(exclude_unset=True)
    try:
        # Agent crawls are the longest-running ingests and set only the embedding
        # model, so a plain overwrite here would drop whatever the user changed
        # in the UI while the crawl ran.
        existing = await metadata_store.get_collection_config(
            collection_name, user_id, is_admin=False
        )
    except Exception as exc:  # noqa: BLE001
        if collection_existed_before:
            # The write replaces the row wholesale and the agent config sets only
            # the embedding model, so writing blind here would wipe the chunking
            # and rerank settings the user saved. The collection is already
            # listed, so keeping its settings costs this run nothing.
            logger.error(
                "Could not read the existing config of agent knowledge base %s; "
                "keeping its settings rather than overwriting: %s",
                collection_name,
                exc,
            )
            return
        logger.warning(
            "Could not read the config of new agent knowledge base %s, saving "
            "this ingest's settings alone: %s",
            collection_name,
            exc,
        )
        existing = None

    try:
        await metadata_store.save_collection_config(
            collection=collection_name,
            config_json=merge_collection_config_json(
                existing if isinstance(existing, str) else None,
                config_json,
            ),
            user_id=user_id,
        )
    except Exception as exc:
        logger.error(
            "Failed to save collection config for agent knowledge base %s: %s",
            collection_name,
            exc,
        )
        raise AgentKnowledgeBaseError(
            f"Failed to save collection config for knowledge base '{collection_name}'"
        ) from exc


async def _refresh_collection_metadata_impl(
    *,
    collection_name: str,
    user_id: int,
    is_admin: bool = False,
) -> None:
    from ...core.RAG_tools.management.collections import list_collections

    if not is_admin:
        # Non-admin realtime refreshes do not persist metadata and only add scan cost.
        return

    try:
        # Refresh metadata cache so agent-created KBs are visible like API-created ones.
        await list_collections(
            user_id=user_id,
            is_admin=is_admin,
            force_realtime=True,
        )
    except Exception as exc:
        logger.error(
            "Failed to refresh collection metadata after agent ingestion for %s: %s",
            collection_name,
            exc,
        )
        raise AgentKnowledgeBaseError(
            f"Failed to refresh knowledge base metadata for '{collection_name}'"
        ) from exc
