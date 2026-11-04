from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd

from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_embeddings_table,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_vector_index_store,
    reset_rag_storage_for_tests,
)
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import (
    list_table_names,
)


def test_forward_migrate_legacy_embeddings_table_to_hub_id(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """Legacy embeddings tables can be migrated to Hub-ID table names using storage API.

    Scenario:
    - Only legacy table exists: embeddings_{to_model_tag(model_name)}
    - Primary Hub-ID table missing: embeddings_{to_model_tag(hub_id)}
    - Using migrate_embeddings_table() creates the primary table and copies rows
      from legacy, rewriting row["model"] to hub_id.
    """
    hub_id = "text-embedding-v4-openai-1"
    legacy_model_name = "text-embedding-v4"
    vector_dim = 3

    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / ".lancedb"))
    reset_rag_storage_for_tests()
    vector_store = get_vector_index_store()
    conn = vector_store.get_raw_connection()

    legacy_tag = to_model_tag(legacy_model_name)
    legacy_table_name = f"embeddings_{legacy_tag}"
    ensure_embeddings_table(conn, legacy_tag, vector_dim=vector_dim)
    legacy_table = conn.open_table(legacy_table_name)

    # Insert one legacy row (model stored as provider model_name in older versions)
    legacy_table.add(
        [
            {
                "collection": "c1",
                "doc_id": "d1",
                "chunk_id": "ch1",
                "parse_hash": "p1",
                "model": legacy_model_name,
                "vector": [0.1, 0.2, 0.3],
                "text": "t",
                "chunk_hash": "h",
                "created_at": pd.Timestamp.now(tz="UTC"),
                "vector_dimension": vector_dim,
                "metadata": None,
                "user_id": None,
            }
        ]
    )

    primary_table_name = f"embeddings_{to_model_tag(hub_id)}"
    # Sanity: primary should not exist yet
    assert primary_table_name not in set(list_table_names(conn))

    # Patch resolver so hub_id -> model_name mapping is available for migration.
    cfg = EmbeddingModelConfig(
        id=hub_id,
        model_name=legacy_model_name,
        model_provider="openai",
        dimension=vector_dim,
        api_key="k",
        base_url="http://example",
        timeout=1.0,
        abilities=["embedding"],
    )

    with patch(
        "xagent.core.tools.core.RAG_tools.utils.model_resolver.resolve_embedding_adapter",
        return_value=(cfg, object()),
    ):
        # Use the storage layer migration method
        result = vector_store.migrate_embeddings_table(hub_id)

        assert result["success"] is True
        assert result["source_table"] == legacy_table_name
        assert result["target_table"] == primary_table_name
        assert result["rows_migrated"] == 1

    # Verify primary table was created
    assert primary_table_name in set(list_table_names(conn))
    primary_table = conn.open_table(primary_table_name)
    rows = primary_table.search().to_pandas()
    assert len(rows) == 1
    assert rows.iloc[0]["model"] == hub_id
