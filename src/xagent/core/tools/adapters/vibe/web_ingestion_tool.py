import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from .....web.tools.config import WebToolConfig
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .factory import register_tool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...core.RAG_tools.kb import KBToolCompatibilityFacade


class CreateKnowledgeBaseFromUrlArgs(BaseModel):
    url: str = Field(
        description="The starting URL of the website to import (e.g. https://www.example.com)"
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Optional name for the knowledge base. If not provided, a name will be generated.",
    )
    max_pages: int = Field(default=10, description="Maximum number of pages to crawl")


class CreateKnowledgeBaseFromUrlResult(BaseModel):
    success: bool
    collection_name: str
    message: str
    pages_crawled: int


class CreateKnowledgeBaseFromUrlTool(AbstractBaseTool):
    """Tool to create a knowledge base by crawling a website."""

    category = ToolCategory.KNOWLEDGE

    def __init__(
        self,
        user_id: int,
        is_admin: bool = False,
    ) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self.user_id = user_id
        self.is_admin = is_admin

    @property
    def name(self) -> str:
        return "create_knowledge_base_from_url"

    @property
    def description(self) -> str:
        return (
            "Create a new knowledge base by crawling and importing a website. "
            "Use this tool when the user provides a specific URL and wants the agent to answer questions based on it. "
            "This tool will automatically crawl the website and create a knowledge base, returning the collection name. "
            "You MUST NOT use this tool if the user hasn't provided a URL."
        )

    def args_type(self) -> Type[BaseModel]:
        return CreateKnowledgeBaseFromUrlArgs

    def return_type(self) -> Type[BaseModel]:
        return CreateKnowledgeBaseFromUrlResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("Only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return await _get_tool_compatibility_facade().create_knowledge_base_from_url(
            args,
            user_id=self.user_id,
            is_admin=self.is_admin,
        )


def _get_tool_compatibility_facade() -> "KBToolCompatibilityFacade":
    """Return the coordinator-owned KB tool compatibility facade."""
    from ...core.RAG_tools.kb import get_kb_coordinator

    return get_kb_coordinator().tool_compatibility


async def _create_knowledge_base_from_url_impl(
    args: Mapping[str, Any],
    *,
    user_id: int,
    is_admin: bool = False,
) -> Any:
    try:
        import hashlib
        import re
        import time

        from ...core.RAG_tools.core.schemas import (
            DEFAULT_EMBEDDING_MODEL_ID,
            IngestionConfig,
            WebCrawlConfig,
        )
        from ...core.RAG_tools.pipelines.web_ingestion import run_web_ingestion
        from .agent_kb_service import (
            AgentKnowledgeBaseError,
            AgentKnowledgeBaseService,
        )

        tool_args = CreateKnowledgeBaseFromUrlArgs.model_validate(args)

        # Generate a safe collection name if not provided
        if not tool_args.collection_name:
            base_name = re.sub(
                r"[^a-zA-Z0-9_-]", "_", tool_args.url.split("//")[-1].split("/")[0]
            )[:30]
            url_hash = hashlib.md5(tool_args.url.encode()).hexdigest()[:6]
            collection_name = f"{base_name}_{url_hash}_{int(time.time())}"
        else:
            collection_name = tool_args.collection_name

        crawl_config = WebCrawlConfig(
            start_url=tool_args.url,
            max_pages=min(tool_args.max_pages, 50),
            max_depth=2,
        )

        ingest_config = IngestionConfig(embedding_model_id=DEFAULT_EMBEDDING_MODEL_ID)
        kb_service = AgentKnowledgeBaseService(
            user_id=user_id,
            is_admin=is_admin,
        )
        collection_name = await kb_service.prepare_collection(collection_name)
        # The ingest pipeline writes a metadata row of its own; remember whether
        # the collection is ours to clean up if the import produces nothing.
        collection_existed_before = await kb_service.collection_exists(collection_name)

        logger.info(
            "Starting background web ingestion for %s into %s",
            tool_args.url,
            collection_name,
        )

        result = await run_web_ingestion(
            collection=collection_name,
            crawl_config=crawl_config,
            ingestion_config=ingest_config,
            user_id=user_id,
            is_admin=is_admin,
        )

        if result.status == "error":
            if not collection_existed_before:
                await kb_service.cleanup_failed_collection(collection_name)
            return CreateKnowledgeBaseFromUrlResult(
                success=False,
                collection_name=collection_name,
                message=f"Failed to crawl website: {result.message}",
                pages_crawled=0,
            ).model_dump()

        # A crawl can finish without a single recorded failure and still ingest
        # nothing (robots.txt, an empty site). Publishing then would leave a
        # visible, empty knowledge base.
        if int(result.documents_created or 0) <= 0:
            if not collection_existed_before:
                await kb_service.cleanup_failed_collection(collection_name)
            return CreateKnowledgeBaseFromUrlResult(
                success=False,
                collection_name=collection_name,
                message=(
                    f"No pages were ingested from {tool_args.url}, so the "
                    f"knowledge base was not created"
                ),
                pages_crawled=result.pages_crawled,
            ).model_dump()

        try:
            await kb_service.publish_collection(
                collection_name,
                ingest_config,
                collection_existed_before=collection_existed_before,
            )
        except AgentKnowledgeBaseError as exc:
            # The pages landed; retrying would re-crawl and duplicate them.
            logger.error("Could not publish agent knowledge base: %s", exc)
            return CreateKnowledgeBaseFromUrlResult(
                success=False,
                collection_name=collection_name,
                message=(
                    f"Imported {result.pages_crawled} page(s) into "
                    f"'{collection_name}' but could not publish it, so it is not "
                    f"listed yet. Do not re-import; retry publishing: {exc}"
                ),
                pages_crawled=result.pages_crawled,
            ).model_dump()
        await kb_service.refresh_collection_metadata(collection_name)

        if result.crawl_blocked_by_site:
            # Narrower than status == "partial" on purpose: a partial caused by
            # individual pages failing to parse keeps reporting success, which
            # is the existing policy. This is the other kind - nothing failed,
            # the site simply refused everything past the start page. The pages
            # that did land stay published (re-importing would re-crawl them),
            # but calling it success would let an agent build on a knowledge
            # base holding one page.
            return CreateKnowledgeBaseFromUrlResult(
                success=False,
                collection_name=collection_name,
                message=(
                    f"{result.message} '{collection_name}' is published and "
                    f"usable with what did land. Re-importing under the same "
                    f"crawl settings will hit the same refusals."
                ),
                pages_crawled=result.pages_crawled,
            ).model_dump()

        return CreateKnowledgeBaseFromUrlResult(
            success=True,
            collection_name=collection_name,
            message=f"Successfully imported website {tool_args.url} into knowledge base '{collection_name}'",
            pages_crawled=result.pages_crawled,
        ).model_dump()
    except ValidationError as e:
        errors = e.errors()
        message = errors[0]["msg"] if errors else str(e)
        if isinstance(message, str) and message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        logger.warning("Invalid URL for knowledge base creation: %s", message)
        return CreateKnowledgeBaseFromUrlResult(
            success=False,
            collection_name="",
            message=message,
            pages_crawled=0,
        ).model_dump()

    except Exception as e:
        logger.exception("Error creating knowledge base from URL: %s", e)
        return CreateKnowledgeBaseFromUrlResult(
            success=False,
            collection_name="",
            message=str(e),
            pages_crawled=0,
        ).model_dump()


@register_tool(categories={"knowledge"})
async def create_web_ingestion_tools(config: WebToolConfig) -> list[AbstractBaseTool]:
    """Create web ingestion tools."""
    return await _get_tool_compatibility_facade().create_web_ingestion_tools(config)


async def _create_web_ingestion_tools_impl(
    config: WebToolConfig,
) -> list[AbstractBaseTool]:
    """Create web ingestion tools."""
    try:
        user_id = config.get_user_id()
        is_admin = config.is_admin()
        if not user_id:
            return []

        tool = CreateKnowledgeBaseFromUrlTool(
            user_id=user_id,
            is_admin=is_admin,
        )
        logger.debug(f"Created CreateKnowledgeBaseFromUrlTool for user {user_id}")
        return [tool]
    except Exception as e:
        logger.warning(f"Failed to create CreateKnowledgeBaseFromUrlTool: {e}")
        return []
