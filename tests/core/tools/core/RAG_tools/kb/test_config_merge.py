"""Merge rule for the collection config row, shared by every ingest entry point."""

from xagent.core.tools.core.RAG_tools.kb.config_merge import (
    merge_collection_config_json,
)


def test_new_settings_win_and_foreign_keys_survive():
    merged = merge_collection_config_json(
        '{"chunk_size": 111, "rerank_model_id": "rr-1"}',
        '{"chunk_size":2048}',
    )

    assert merged == '{"chunk_size":2048,"rerank_model_id":"rr-1"}'


def test_no_existing_config_is_reserialized_compactly():
    assert merge_collection_config_json(None, '{"chunk_size": 2048}') == (
        '{"chunk_size":2048}'
    )
    assert merge_collection_config_json("", '{"chunk_size": 2048}') == (
        '{"chunk_size":2048}'
    )


def test_unparsable_new_config_is_handed_back_untouched():
    """Nothing to merge into, and reshaping it would corrupt what the caller sent."""
    assert merge_collection_config_json('{"chunk_size":111}', "not json") == "not json"


def test_non_object_new_config_is_handed_back_untouched():
    assert merge_collection_config_json('{"chunk_size":111}', "[1, 2]") == "[1, 2]"


def test_unreadable_existing_config_is_replaced():
    assert merge_collection_config_json("not json", '{"chunk_size": 2048}') == (
        '{"chunk_size":2048}'
    )


def test_non_object_existing_config_is_replaced():
    assert merge_collection_config_json("[1, 2]", '{"chunk_size": 2048}') == (
        '{"chunk_size":2048}'
    )
