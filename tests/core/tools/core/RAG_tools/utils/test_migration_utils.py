from unittest.mock import MagicMock, patch

from xagent.core.tools.core.RAG_tools.utils.migration_utils import (
    _infer_embedding_config_from_collection,
    _model_tag_to_model_id,
    migrate_collection_metadata,
)
from xagent.core.tools.core.RAG_tools.utils.tag_mapping import register_tag_mapping


class TestMigrateCollectionMetadata:
    """Test collection metadata migration."""

    def test_migrate_from_version_0_0_0_to_1_0_0_basic(self):
        """Test basic migration from pre-versioned to 1.0.0."""
        legacy_data = {
            "name": "test_collection",
            "documents": 5,
            "document_names": ["doc1.txt", "doc2.txt"],
            "created_at": "2024-01-01T00:00:00",
            "extra_metadata": {"custom": "value"},
        }

        # Mock inference to avoid real DB connection
        with patch(
            "xagent.core.tools.core.RAG_tools.utils.migration_utils._infer_embedding_config_from_collection"
        ) as mock_infer:
            mock_infer.return_value = (None, None)
            result = migrate_collection_metadata(legacy_data)

        assert result["schema_version"] == "1.0.0"
        assert result["name"] == "test_collection"
        assert result["documents"] == 5
        assert result["document_names"] == ["doc1.txt", "doc2.txt"]
        assert result["extra_metadata"] == {"custom": "value"}
        assert result["embedding_model_id"] is None
        assert result["embedding_dimension"] is None

    def test_migrate_already_at_target_version(self):
        """Test migration when already at target version."""
        data = {
            "schema_version": "1.0.0",
            "name": "test_collection",
            "embedding_model_id": "text-embedding-ada-002",
            "embedding_dimension": 1536,
        }

        result = migrate_collection_metadata(data)

        assert result["schema_version"] == "1.0.0"
        assert result["embedding_model_id"] == "text-embedding-ada-002"
        assert result["embedding_dimension"] == 1536

    @patch(
        "xagent.core.tools.core.RAG_tools.utils.migration_utils._infer_embedding_config_from_collection"
    )
    def test_migrate_with_embedding_inference(self, mock_infer):
        """Test migration with embedding config inference."""
        mock_infer.return_value = ("text-embedding-ada-002", 1536)

        legacy_data = {
            "name": "test_collection",
            "documents": 10,
        }

        result = migrate_collection_metadata(legacy_data)

        assert result["embedding_model_id"] == "text-embedding-ada-002"
        assert result["embedding_dimension"] == 1536
        mock_infer.assert_called_once_with("test_collection")

    @patch(
        "xagent.core.tools.core.RAG_tools.utils.migration_utils._infer_embedding_config_from_collection"
    )
    def test_migrate_without_embedding_inference_skips_db(self, mock_infer):
        """Read-safe migration must not scan LanceDB for embedding config."""
        legacy_data = {
            "name": "test_collection",
            "documents": 10,
        }

        result = migrate_collection_metadata(legacy_data, infer_embedding=False)

        mock_infer.assert_not_called()
        assert result["schema_version"] == "1.0.0"
        assert result["embedding_model_id"] is None
        assert result["embedding_dimension"] is None


class TestInferEmbeddingConfigFromCollection:
    """Test embedding config inference."""

    @patch(
        "xagent.core.tools.core.RAG_tools.utils.migration_utils.get_vector_store_raw_connection"
    )
    def test_infer_no_tables_found(self, mock_conn):
        """Test inference when no embedding tables exist."""
        mock_connection = MagicMock(spec=["table_names", "open_table"])
        mock_conn.return_value = mock_connection
        mock_connection.table_names.return_value = []

        result = _infer_embedding_config_from_collection("test_collection")

        assert result == (None, None)

    @patch(
        "xagent.core.tools.core.RAG_tools.utils.migration_utils.get_vector_store_raw_connection"
    )
    def test_infer_single_model(self, mock_conn):
        """Test inference with single embedding model."""
        mock_connection = MagicMock(spec=["table_names", "open_table"])
        mock_conn.return_value = mock_connection
        mock_connection.table_names.return_value = [
            "embeddings_OPENAI_text_embedding_ada_002"
        ]

        mock_table = MagicMock()
        mock_connection.open_table.return_value = mock_table

        # Mock schema with dimension
        mock_field = MagicMock()
        mock_field.name = "vector"
        mock_field.type.list_size = 1536
        mock_table.schema.__iter__.return_value = [mock_field]

        # Mock search result
        mock_result = MagicMock()
        mock_result.empty = False
        mock_result.__len__.return_value = 10
        mock_table.search.return_value.where.return_value.limit.return_value.to_pandas.return_value = mock_result

        result = _infer_embedding_config_from_collection("test_collection")

        assert result == ("text-embedding-ada-002", 1536)

    @patch(
        "xagent.core.tools.core.RAG_tools.utils.migration_utils.get_vector_store_raw_connection"
    )
    def test_infer_multiple_models_choose_most_used(self, mock_conn):
        """Test inference with multiple models chooses most used."""
        mock_connection = MagicMock(spec=["table_names", "open_table"])
        mock_conn.return_value = mock_connection
        mock_connection.table_names.return_value = [
            "embeddings_OPENAI_text_embedding_ada_002",
            "embeddings_BAAI_bge_large_zh_v1_5",
        ]

        def mock_open_table(table_name):
            mt = MagicMock()
            mt.name = table_name
            mf = MagicMock()
            mf.name = "vector"
            mf.type.list_size = 1536 if "OPENAI" in table_name else 1024
            mt.schema.__iter__.return_value = [mf]

            mr = MagicMock()
            mr.empty = False
            mr.__len__.return_value = 20 if "OPENAI" in table_name else 5
            mt.search.return_value.where.return_value.limit.return_value.to_pandas.return_value = mr
            return mt

        mock_connection.open_table.side_effect = mock_open_table

        with patch(
            "xagent.core.tools.core.RAG_tools.utils.migration_utils.logger"
        ) as mock_logger:
            result = _infer_embedding_config_from_collection("test_collection")

        assert result == ("text-embedding-ada-002", 1536)
        mock_logger.warning.assert_called_once()


class TestHubTagMapping:
    """Test tag collision handling when building hub lookup maps."""

    def test_register_hub_tag_mapping_warns_on_collision(self) -> None:
        mapping = {"OPENAI_text_embedding_3_large": "hub-id-a"}
        mock_logger = MagicMock()

        register_tag_mapping(
            mapping,
            "OPENAI_text_embedding_3_large",
            "hub-id-b",
            get_identity=lambda item: item,
            logger=mock_logger,
        )

        assert mapping["OPENAI_text_embedding_3_large"] == "hub-id-a"
        mock_logger.warning.assert_called_once_with(
            "Tag collision: %s -> %s vs %s",
            "OPENAI_text_embedding_3_large",
            "hub-id-a",
            "hub-id-b",
        )


class TestModelTagToModelId:
    """Test model tag to model ID conversion."""

    def test_openai_model_conversion(self):
        """Test OpenAI model tag conversion."""
        result = _model_tag_to_model_id("OPENAI_text_embedding_ada_002")
        assert result == "text-embedding-ada-002"

    def test_baai_model_conversion(self):
        """Test BAAI model tag conversion."""
        result = _model_tag_to_model_id("BAAI_bge_large_zh_v1_5")
        assert result == "bge-large-zh-v1-5"
