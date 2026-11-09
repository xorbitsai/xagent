from __future__ import annotations

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_candidates,
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


def test_candidates_of_a_vendor_prefixed_id_cover_both_spellings() -> None:
    assert embeddings_table_candidates("platform/text-embedding-v4") == [
        "embeddings_PLATFORM_text_embedding_v4",
        "embeddings_platform_text_embedding_v4",
    ]


def test_candidates_of_an_id_without_a_slash_is_one_name() -> None:
    assert embeddings_table_candidates("text-embedding-v4") == [
        "embeddings_text_embedding_v4"
    ]
    assert embeddings_table_candidates("Text-Embedding-V4") == [
        "embeddings_text_embedding_v4"
    ]


def test_candidates_are_deduplicated() -> None:
    candidates = embeddings_table_candidates("platform_text_embedding_v4")
    assert candidates == ["embeddings_platform_text_embedding_v4"]
    assert len(candidates) == len(set(candidates))


def test_last_candidate_is_the_table_the_write_path_creates() -> None:
    # The ingestion path passes to_model_tag(model) to upsert_embeddings, which tags again.
    model_id = "platform/text-embedding-v4"
    assert embeddings_table_candidates(model_id)[-1] == embeddings_table_name(
        to_model_tag(model_id)
    )
