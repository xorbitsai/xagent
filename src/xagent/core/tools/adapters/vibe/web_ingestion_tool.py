import logging
from typing import Any, Mapping, Optional, Type

from pydantic import BaseModel, Field

from .....web.tools.config import WebToolConfig
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .factory import register_tool

logger = logging.getLogger(__name__)


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
        try:
            import time

            from ...core.RAG_tools.core.schemas import IngestionConfig, WebCrawlConfig
            from ...core.RAG_tools.pipelines.web_ingestion import run_web_ingestion

            tool_args = CreateKnowledgeBaseFromUrlArgs.model_validate(args)

            # Generate a safe collection name if not provided
            if not tool_args.collection_name:
                import hashlib
                import re

                base_name = re.sub(
                    r"[^a-zA-Z0-9_-]", "_", tool_args.url.split("//")[-1].split("/")[0]
                )[:30]
                url_hash = hashlib.md5(tool_args.url.encode()).hexdigest()[:6]
                collection_name = f"{base_name}_{url_hash}_{int(time.time())}"
            else:
                collection_name = tool_args.collection_name

            crawl_config = WebCrawlConfig(
                start_url=tool_args.url,
                max_pages=min(
                    tool_args.max_pages, 50
                ),  # Cap at 50 to avoid long blocking
                max_depth=2,
            )

            ingest_config = IngestionConfig(embedding_model_id="text-embedding-v4")

            logger.info(
                f"Starting background web ingestion for {tool_args.url} into {collection_name}"
            )

            # Run ingestion
            result = await run_web_ingestion(
                collection=collection_name,
                crawl_config=crawl_config,
                ingestion_config=ingest_config,
                user_id=self.user_id,
                is_admin=self.is_admin,
            )

            if result.status == "error":
                return CreateKnowledgeBaseFromUrlResult(
                    success=False,
                    collection_name=collection_name,
                    message=f"Failed to crawl website: {result.message}",
                    pages_crawled=0,
                ).model_dump()

            return CreateKnowledgeBaseFromUrlResult(
                success=True,
                collection_name=collection_name,
                message=f"Successfully imported website {tool_args.url} into knowledge base '{collection_name}'",
                pages_crawled=result.pages_crawled,
            ).model_dump()

        except Exception as e:
            logger.exception(f"Error creating knowledge base from URL: {e}")
            return CreateKnowledgeBaseFromUrlResult(
                success=False, collection_name="", message=str(e), pages_crawled=0
            ).model_dump()


@register_tool
async def create_web_ingestion_tools(config: WebToolConfig) -> list[AbstractBaseTool]:
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
