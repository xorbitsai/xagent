"""Tests for Zhipu web search normalization."""

import pytest
from pydantic import ValidationError

from xagent.core.tools.adapters.vibe.zhipu_web_search import ZhipuWebSearchArgs
from xagent.core.tools.core.zhipu_web_search import ZhipuWebSearchCore


def _response():
    return {
        "search_result": [
            {
                "title": "Example",
                "link": "https://example.com",
                "content": "Provider summary",
                "media": "Example Media",
                "publish_date": "2026-01-01",
            }
        ]
    }


def test_zhipu_args_default_to_lightweight_results():
    args = ZhipuWebSearchArgs(query="test")

    assert args.content_size == "low"
    assert args.include_content is False


def test_zhipu_args_reject_invalid_content_size():
    with pytest.raises(ValidationError):
        ZhipuWebSearchArgs(query="test", content_size="full")


def test_zhipu_normalize_omits_content_by_default():
    results = ZhipuWebSearchCore.normalize_results(_response())

    assert results[0]["snippet"] == "Provider summary"
    assert "content" not in results[0]


def test_zhipu_normalize_includes_content_when_requested():
    results = ZhipuWebSearchCore.normalize_results(_response(), include_content=True)

    assert results[0]["snippet"] == "Provider summary"
    assert results[0]["content"] == "Provider summary"
