from __future__ import annotations

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_name,
    to_model_tag,
)


def test_to_model_tag_basic() -> None:
    assert to_model_tag("BAAI/bge-large-zh-v1.5") == "BAAI_bge_large_zh_v1_5"
    assert to_model_tag("text-embedding-3-large") == "text_embedding_3_large"
    assert (
        to_model_tag("  OpenAI/Text-Embeddings-ada-002  ")
        == "OPENAI_text_embeddings_ada_002"
    )


def test_embeddings_table_name() -> None:
    assert (
        embeddings_table_name("BAAI/bge-large-zh-v1.5")
        == "embeddings_BAAI_bge_large_zh_v1_5"
    )
