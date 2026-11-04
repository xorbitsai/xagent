from xagent.core.tools.core.RAG_tools.core import config as rag_config
from xagent.core.tools.core.RAG_tools.core.factory_utils import (
    canonicalize_model_name,
    get_chunk_param_whitelist,
    get_default_index_policy,
    get_parse_param_whitelist,
)


def test_canonicalize_model_name_qwen_and_bge() -> None:
    """Canonicalization should map known synonyms and preserve unknown names."""
    # Qwen synonyms
    assert canonicalize_model_name("text-embedding-v4") == "QWEN/text-embedding-v4"
    assert canonicalize_model_name(" text-embedding-v3 ") == "QWEN/text-embedding-v3"

    # BGE synonyms
    assert canonicalize_model_name("bge-large-zh-v1.5") == "BAAI/bge-large-zh-v1.5"

    # Unknown model should be returned as-is (after strip)
    assert canonicalize_model_name("  my-embed  ") == "my-embed"


def test_get_parse_param_whitelist_matches_config() -> None:
    """Parse whitelist returned by factory should mirror config and be immutable tuple."""
    wl = get_parse_param_whitelist()
    assert isinstance(wl, tuple)
    assert wl == tuple(rag_config.PARSE_PARAM_WHITELIST)
    # Spot-check a couple of expected keys
    assert "extract_tables" in wl
    assert "ocr_enabled" in wl


def test_get_chunk_param_whitelist_matches_config() -> None:
    """Chunk whitelist returned by factory should mirror config and be immutable tuple."""
    wl = get_chunk_param_whitelist()
    assert isinstance(wl, tuple)
    assert wl == tuple(rag_config.CHUNK_PARAM_WHITELIST)
    # Spot-check expected keys
    assert "chunk_strategy" in wl
    assert "chunk_overlap" in wl


def test_get_default_index_policy() -> None:
    """Test get_default_index_policy returns static default values for backward compatibility.

    Note: This function returns static defaults only. The actual dynamic index type
    selection based on data scale (HNSW for 50k-10M rows, IVFPQ for >=10M rows) is
    implemented in storage.lancedb_stores.LanceDBVectorIndexStore.create_index().
    """
    threshold, index_type = get_default_index_policy()

    assert threshold == rag_config.DEFAULT_INDEX_POLICY.enable_threshold_rows
    assert index_type == rag_config.DEFAULT_INDEX_TYPE  # Static default: "HNSW"

    # Sanity checks for backward compatibility defaults
    assert threshold == 50_000
    assert index_type == "HNSW"
