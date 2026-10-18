"""Tests for ``xinference`` rerank adapter."""

from unittest.mock import Mock, patch

import pytest
import requests

from xagent.core.model.rerank import XinferenceRerank
from xagent.core.model.rerank.xinference import (
    XinferenceRerank as XinferenceRerankDirect,
)


class TestXinferenceRerankInit:
    """Initialization tests."""

    def test_initialization_with_explicit_values(self):
        client = XinferenceRerank(
            model="bge-reranker-base",
            api_key="test-key",
            base_url="http://localhost:9997/",
            top_n=3,
            timeout=30.0,
        )
        assert client.model == "bge-reranker-base"
        assert client.api_key == "test-key"
        # base_url trailing slash is stripped
        assert client.base_url == "http://localhost:9997"
        assert client.top_n == 3
        assert client.timeout == 30.0
        assert client._model_uid == "bge-reranker-base"

    def test_initialization_uses_model_uid(self):
        client = XinferenceRerank(
            model="bge-reranker-base",
            model_uid="custom-uid-123",
            base_url="http://localhost:9997",
        )
        assert client._model_uid == "custom-uid-123"

    def test_initialization_default_base_url(self):
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url=None,
        )
        assert client.base_url == "http://localhost:9997"

    def test_initialization_picks_up_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("XINFERENCE_API_KEY", "env-key")
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        assert client.api_key == "env-key"

    def test_initialization_no_api_key_allowed(self, monkeypatch):
        # Xinference can run without auth, so no error is expected.
        monkeypatch.delenv("XINFERENCE_API_KEY", raising=False)
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        assert client.api_key is None

    def test_headers_without_api_key(self):
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        assert client._headers() == {"Content-Type": "application/json"}

    def test_headers_with_api_key(self):
        client = XinferenceRerank(
            model="bge-reranker-base",
            api_key="k",
            base_url="http://localhost:9997",
        )
        headers = client._headers()
        assert headers["Authorization"] == "Bearer k"
        assert headers["Content-Type"] == "application/json"


class TestXinferenceRerankCompress:
    """End-to-end behaviour of ``compress``."""

    @patch("xagent.core.model.public_requests.post")
    def test_compress_reorders_documents(self, mock_post):
        """Documents are returned in the order the reranker requested."""
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        documents = ["Paris is the capital.", "Eiffel Tower is tall.", "London is big."]
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.1},
            ]
        }
        mock_post.return_value = mock_response

        result = client.compress(documents, "capital of France")

        assert result == [
            "Paris is the capital.",
            "London is big.",
            "Eiffel Tower is tall.",
        ]
        # Verify request payload and URL.
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:9997/v1/rerank"
        payload = kwargs["json"]
        assert payload["model"] == "bge-reranker-base"
        assert payload["query"] == "capital of France"
        assert payload["documents"] == documents
        # No top_n => not in payload
        assert "top_n" not in payload

    @patch("xagent.core.model.public_requests.post")
    def test_compress_with_top_n(self, mock_post):
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
            top_n=2,
        )
        documents = ["a", "b", "c"]
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
            ]
        }
        mock_post.return_value = mock_response

        result = client.compress(documents, "query")

        # top_n=2 means "c", "a"; "b" gets appended at the end.
        assert result == ["c", "a", "b"]
        payload = mock_post.call_args.kwargs["json"]
        assert payload["top_n"] == 2

    @patch("xagent.core.model.public_requests.post")
    def test_compress_empty_documents_returns_empty(self, mock_post):
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        result = client.compress([], "query")
        assert result == []
        mock_post.assert_not_called()

    @patch("xagent.core.model.public_requests.post")
    def test_compress_preserves_dropped_documents(self, mock_post):
        """Documents missing from the response are appended in original order."""
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        documents = ["a", "b", "c", "d"]
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        # Server only returns indices 0 and 2; "b" and "d" must be preserved.
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
            ]
        }
        mock_post.return_value = mock_response

        result = client.compress(documents, "query")

        assert result == ["c", "a", "b", "d"]

    @patch("xagent.core.model.public_requests.post")
    def test_compress_http_error_raises(self, mock_post):
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            response=Mock(status_code=500, text="boom")
        )
        mock_post.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            client.compress(["a", "b"], "q")

    @patch("xagent.core.model.public_requests.post")
    def test_compress_invalid_json_raises_value_error(self, mock_post):
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("bad json")
        mock_response.text = "<html>error</html>"
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="non-JSON response"):
            client.compress(["a", "b"], "q")

    @patch("xagent.core.model.public_requests.post")
    def test_compress_missing_results_raises_value_error(self, mock_post):
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"usage": {"tokens": 1}}
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="missing 'results'"):
            client.compress(["a", "b"], "q")

    @patch("xagent.core.model.public_requests.post")
    def test_compress_invalid_index_raises_value_error(self, mock_post):
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [{"relevance_score": 0.9}]  # missing "index"
        }
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="invalid 'index'"):
            client.compress(["a", "b"], "q")

    @patch("xagent.core.model.public_requests.post")
    def test_compress_skips_out_of_range_and_duplicate_indices(self, mock_post):
        client = XinferenceRerank(
            model="bge-reranker-base",
            base_url="http://localhost:9997",
        )
        documents = ["a", "b"]
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"index": 5, "relevance_score": 0.9},  # out of range
                {"index": 0, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.4},  # duplicate
            ]
        }
        mock_post.return_value = mock_response

        result = client.compress(documents, "q")

        # "a" appears once, "b" is appended at the end (not in response).
        assert result == ["a", "b"]


class TestXinferenceRerankListAvailable:
    """Tests for ``list_available_models``."""

    def test_list_filters_rerank_models(self):
        class _StubClient:
            def __init__(self, *args, **kwargs):
                pass

            def list_models(self):
                return {
                    "uid-1": {
                        "model_type": "rerank",
                        "model_name": "bge-reranker-base",
                        "model_ability": ["rerank"],
                        "model_description": "BGE reranker",
                    },
                    "uid-2": {
                        "model_type": "embedding",
                        "model_name": "bge-base-en",
                    },
                    "uid-3": {
                        "model_type": "rerank",
                        "model_name": "qwen3-rerank",
                        "model_ability": ["rerank", "chat"],
                    },
                }

            def close(self):
                pass

        with patch.object(
            XinferenceRerankDirect, "_StubClient", _StubClient, create=True
        ):
            # The function uses an inline import; patch the import.
            with patch(
                "xagent.core.model.public_xinference.PublicXinferenceClient",
                _StubClient,
            ):
                result = XinferenceRerankDirect.list_available_models(
                    base_url="http://localhost:9997",
                )

        assert len(result) == 2
        ids = {m["id"] for m in result}
        assert ids == {"bge-reranker-base", "qwen3-rerank"}
        for m in result:
            assert m["model_type"] == "rerank"
            assert "model_uid" in m
            assert "model_ability" in m

    def test_list_handles_failure(self):
        class _StubClient:
            def __init__(self, *args, **kwargs):
                pass

            def list_models(self):
                raise RuntimeError("connection refused")

            def close(self):
                pass

        with patch(
            "xagent.core.model.public_xinference.PublicXinferenceClient", _StubClient
        ):
            result = XinferenceRerankDirect.list_available_models(
                base_url="http://localhost:9997",
            )

        assert result == []

    def test_list_skips_non_dict_entries(self):
        class _StubClient:
            def __init__(self, *args, **kwargs):
                pass

            def list_models(self):
                return {
                    "uid-1": "not-a-dict",
                    "uid-2": {"model_type": "rerank", "model_name": "x"},
                }

            def close(self):
                pass

        with patch(
            "xagent.core.model.public_xinference.PublicXinferenceClient", _StubClient
        ):
            result = XinferenceRerankDirect.list_available_models(
                base_url="http://localhost:9997",
            )

        assert len(result) == 1
        assert result[0]["id"] == "x"
