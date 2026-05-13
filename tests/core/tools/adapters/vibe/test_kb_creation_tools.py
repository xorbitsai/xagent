import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from xagent.core.tools.adapters.vibe.agent_kb_service import (
    AgentKnowledgeBaseError,
    AgentKnowledgeBaseService,
)
from xagent.core.tools.adapters.vibe.file_ingestion_tool import (
    CreateKnowledgeBaseFromFileTool,
)
from xagent.core.tools.adapters.vibe.web_ingestion_tool import (
    CreateKnowledgeBaseFromUrlTool,
)
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DEFAULT_EMBEDDING_MODEL_ID,
    IngestionConfig,
    IngestionResult,
    WebIngestionResult,
)


@pytest.mark.asyncio
async def test_agent_kb_service_prepare_collection_persists_config_and_sanitizes():
    metadata_store = MagicMock()
    metadata_store.save_collection_config = AsyncMock()
    service = AgentKnowledgeBaseService(user_id=71, is_admin=False)
    ingest_config = IngestionConfig(embedding_model_id=DEFAULT_EMBEDDING_MODEL_ID)

    with patch(
        "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
        return_value=metadata_store,
    ):
        collection_name = await service.prepare_collection(
            "  agent url kb  ", ingest_config
        )

    assert collection_name == "agent url kb"
    metadata_store.save_collection_config.assert_awaited_once()
    _, save_kwargs = metadata_store.save_collection_config.await_args
    assert save_kwargs["collection"] == "agent url kb"
    assert save_kwargs["user_id"] == 71
    assert json.loads(save_kwargs["config_json"]) == {
        "embedding_model_id": DEFAULT_EMBEDDING_MODEL_ID
    }


@pytest.mark.asyncio
async def test_agent_kb_service_refresh_collection_metadata_forces_realtime_for_admin():
    refresh_metadata = AsyncMock()
    service = AgentKnowledgeBaseService(user_id=71, is_admin=True)

    with patch(
        "xagent.core.tools.core.RAG_tools.management.collections.list_collections",
        new=refresh_metadata,
    ):
        await service.refresh_collection_metadata("agent_url_kb")

    refresh_metadata.assert_awaited_once_with(
        user_id=71,
        is_admin=True,
        force_realtime=True,
    )


@pytest.mark.asyncio
async def test_agent_kb_service_refresh_collection_metadata_skips_non_admin_refresh():
    refresh_metadata = AsyncMock()
    service = AgentKnowledgeBaseService(user_id=71, is_admin=False)

    with patch(
        "xagent.core.tools.core.RAG_tools.management.collections.list_collections",
        new=refresh_metadata,
    ):
        await service.refresh_collection_metadata("agent_url_kb")

    refresh_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_kb_service_prepare_collection_raises_on_config_save_failure():
    metadata_store = MagicMock()
    metadata_store.save_collection_config = AsyncMock(
        side_effect=RuntimeError("config save failed")
    )
    service = AgentKnowledgeBaseService(user_id=71, is_admin=False)
    ingest_config = IngestionConfig(embedding_model_id=DEFAULT_EMBEDDING_MODEL_ID)

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        pytest.raises(
            AgentKnowledgeBaseError, match="Failed to save collection config"
        ),
    ):
        await service.prepare_collection("agent kb", ingest_config)


@pytest.mark.asyncio
async def test_agent_kb_service_refresh_collection_metadata_raises_on_failure():
    refresh_metadata = AsyncMock(side_effect=RuntimeError("refresh failed"))
    service = AgentKnowledgeBaseService(user_id=71, is_admin=True)

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.list_collections",
            new=refresh_metadata,
        ),
        pytest.raises(
            AgentKnowledgeBaseError,
            match="Failed to refresh knowledge base metadata",
        ),
    ):
        await service.refresh_collection_metadata("agent_url_kb")


@pytest.mark.asyncio
async def test_create_kb_from_url_uses_shared_service():
    ingest_result = WebIngestionResult(
        status="success",
        collection="agent_url_kb",
        total_urls_found=1,
        pages_crawled=1,
        pages_failed=0,
        documents_created=1,
        chunks_created=3,
        embeddings_created=3,
        crawled_urls=["https://example.com"],
        failed_urls={},
        message="ok",
        warnings=[],
        elapsed_time_ms=123,
    )
    service = MagicMock()
    service.prepare_collection = AsyncMock(return_value="agent_url_kb")
    service.refresh_collection_metadata = AsyncMock()

    with (
        patch(
            "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
            return_value=service,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.pipelines.web_ingestion.run_web_ingestion",
            new=AsyncMock(return_value=ingest_result),
        ),
    ):
        tool = CreateKnowledgeBaseFromUrlTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"url": "https://example.com", "collection_name": "agent_url_kb"}
        )

    assert result["success"] is True
    service.prepare_collection.assert_awaited_once()
    _, prepare_kwargs = service.prepare_collection.await_args
    assert prepare_kwargs["collection_name"] == "agent_url_kb"
    assert (
        prepare_kwargs["ingestion_config"].embedding_model_id
        == DEFAULT_EMBEDDING_MODEL_ID
    )
    service.refresh_collection_metadata.assert_awaited_once_with("agent_url_kb")


@pytest.mark.asyncio
async def test_create_kb_from_url_returns_error_when_shared_service_fails():
    service = MagicMock()
    service.prepare_collection = AsyncMock(
        side_effect=AgentKnowledgeBaseError("config save failed")
    )
    service.refresh_collection_metadata = AsyncMock()

    with patch(
        "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
        return_value=service,
    ):
        tool = CreateKnowledgeBaseFromUrlTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"url": "https://example.com", "collection_name": "agent_url_kb"}
        )

    assert result["success"] is False
    assert result["message"] == "config save failed"


@pytest.mark.asyncio
async def test_create_kb_from_file_uses_shared_service(tmp_path):
    source_file = tmp_path / "notes.txt"
    source_file.write_text("hello", encoding="utf-8")
    file_record = SimpleNamespace(
        filename="notes.txt",
        storage_path=str(source_file),
        file_id="file-1",
    )

    query = MagicMock()
    query.filter.return_value = query
    query.all.return_value = [file_record]

    db = MagicMock()
    db.query.return_value = query

    def fake_get_db():
        yield db

    ingest_result = IngestionResult(
        status="success",
        doc_id="doc-1",
        parse_hash="parse-1",
        chunk_count=2,
        embedding_count=2,
        vector_count=2,
        completed_steps=[],
        failed_step=None,
        message="ok",
        warnings=[],
        file_id="file-1",
    )
    service = MagicMock()
    service.prepare_collection = AsyncMock(return_value="agent_file_kb")
    service.refresh_collection_metadata = AsyncMock()

    with (
        patch("xagent.web.models.database.get_db", side_effect=fake_get_db),
        patch(
            "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
            return_value=service,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.pipelines.document_ingestion.run_document_ingestion",
            new=Mock(return_value=ingest_result),
        ),
    ):
        tool = CreateKnowledgeBaseFromFileTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"file_ids": ["file-1"], "collection_name": "agent_file_kb"}
        )

    assert result["success"] is True
    service.prepare_collection.assert_awaited_once()
    _, prepare_kwargs = service.prepare_collection.await_args
    assert prepare_kwargs["collection_name"] == "agent_file_kb"
    assert (
        prepare_kwargs["ingestion_config"].embedding_model_id
        == DEFAULT_EMBEDDING_MODEL_ID
    )
    service.refresh_collection_metadata.assert_awaited_once_with("agent_file_kb")
    db.close.assert_called_once()


@pytest.mark.asyncio
async def test_create_kb_from_file_returns_error_when_metadata_refresh_fails(tmp_path):
    source_file = tmp_path / "notes.txt"
    source_file.write_text("hello", encoding="utf-8")
    file_record = SimpleNamespace(
        filename="notes.txt",
        storage_path=str(source_file),
        file_id="file-1",
    )

    query = MagicMock()
    query.filter.return_value = query
    query.all.return_value = [file_record]

    db = MagicMock()
    db.query.return_value = query

    def fake_get_db():
        yield db

    ingest_result = IngestionResult(
        status="success",
        doc_id="doc-1",
        parse_hash="parse-1",
        chunk_count=2,
        embedding_count=2,
        vector_count=2,
        completed_steps=[],
        failed_step=None,
        message="ok",
        warnings=[],
        file_id="file-1",
    )
    service = MagicMock()
    service.prepare_collection = AsyncMock(return_value="agent_file_kb")
    service.refresh_collection_metadata = AsyncMock(
        side_effect=AgentKnowledgeBaseError("metadata refresh failed")
    )

    with (
        patch("xagent.web.models.database.get_db", side_effect=fake_get_db),
        patch(
            "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
            return_value=service,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.pipelines.document_ingestion.run_document_ingestion",
            new=Mock(return_value=ingest_result),
        ),
    ):
        tool = CreateKnowledgeBaseFromFileTool(user_id=71, is_admin=False)
        result = await tool.run_json_async(
            {"file_ids": ["file-1"], "collection_name": "agent_file_kb"}
        )

    assert result["success"] is False
    assert result["message"] == "metadata refresh failed"
    db.close.assert_called_once()
