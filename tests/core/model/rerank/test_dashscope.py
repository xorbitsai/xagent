from unittest.mock import Mock, patch

import pytest
import requests

from xagent.core.model.rerank import DashscopeRerank


class TestDashscopeRerank:
    """Test DashscopeRerank client."""

    def test_initialization(self):
        """Test client initialization."""
        client = DashscopeRerank(
            model="test-rerank-model",
            api_key="test_key",
            top_n=5,
            instruct="Rerank instruction",
        )

        assert client.model == "test-rerank-model"
        assert client.api_key == "test_key"
        assert client.top_n == 5
        assert client.instruct == "Rerank instruction"
        assert (
            client.url
            == "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        )

    def test_default_initialization(self):
        """Test client initialization with defaults."""
        client = DashscopeRerank(api_key="test_key")

        assert client.model == "qwen3-rerank"
        assert client.top_n is None
        assert client.instruct is None
        assert client.timeout == 60.0

    def test_requests_carry_the_configured_timeout(self):
        """Every rerank call must be bounded; an unbounded post hangs retrieval."""
        client = DashscopeRerank(api_key="test_key", timeout=12.0)
        response = Mock()
        response.json.return_value = {"results": [{"index": 0}]}

        with patch("requests.post", return_value=response) as post:
            client.compress(["doc"], "query")
            client.compress_with_scores(["doc"], "query")

        assert [call.kwargs["timeout"] for call in post.call_args_list] == [12.0, 12.0]

    def test_missing_api_key(self, monkeypatch):
        """Test error when API key is missing and not in env."""
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            DashscopeRerank()

    @pytest.mark.parametrize(
        "model,is_new_format",
        [
            ("qwen3-rerank", True),  # New format
            ("gte-rerank-v2", False),  # Old format
        ],
    )
    @patch("requests.post")
    def test_compress_success(self, mock_post, model, is_new_format):
        """Test successful reranking (compress) for both formats."""
        query = "What is the capital of France?"
        documents = ["Paris is the capital.", "Eiffel Tower is tall.", "London is big."]
        expected_reranked = [
            "Paris is the capital.",
            "London is big.",
            "Eiffel Tower is tall.",
        ]

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        if is_new_format:
            # New format response
            mock_response.json.return_value = {
                "object": "list",
                "results": [
                    {"index": 0, "relevance_score": 0.9},  # Paris is the capital.
                    {"index": 2, "relevance_score": 0.5},  # London is big.
                    {"index": 1, "relevance_score": 0.1},  # Eiffel Tower is tall.
                ],
                "model": "qwen3-rerank",
                "id": "test-id-123",
                "usage": {"total_tokens": 105},
            }
        else:
            # Old format response
            mock_response.json.return_value = {
                "output": {
                    "results": [
                        {"document": {"text": expected_reranked[0]}, "score": 0.9},
                        {"document": {"text": expected_reranked[1]}, "score": 0.5},
                        {"document": {"text": expected_reranked[2]}, "score": 0.1},
                    ]
                }
            }

        mock_post.return_value = mock_response

        client = DashscopeRerank(api_key="test_key", model=model)
        reranked = client.compress(documents, query)

        assert reranked == expected_reranked

        # Verify request payload
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["model"] == model

        if is_new_format:
            # New format: query and documents at top level
            assert json_payload["query"] == query
            assert json_payload["documents"] == documents
            assert "input" not in json_payload
            assert "parameters" not in json_payload
        else:
            # Old format: query and documents in input field
            assert json_payload["input"]["query"] == query
            assert json_payload["input"]["documents"] == documents
            assert json_payload["parameters"]["return_documents"] is True
            assert "query" not in json_payload  # Not at top level
            assert "documents" not in json_payload  # Not at top level

    @pytest.mark.parametrize(
        "model,is_new_format",
        [
            ("qwen3-rerank", True),  # New format
            ("gte-rerank-v2", False),  # Old format
        ],
    )
    @patch("requests.post")
    def test_compress_with_top_n_and_instruct(self, mock_post, model, is_new_format):
        """Test successful reranking with top_n and instruct for both formats."""
        query = "Query"
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        expected_reranked = ["Doc 1", "Doc 2"]

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        if is_new_format:
            # New format response (returns only top 2)
            mock_response.json.return_value = {
                "object": "list",
                "results": [
                    {"index": 0, "relevance_score": 0.9},  # Doc 1
                    {"index": 1, "relevance_score": 0.5},  # Doc 2
                ],
                "model": "qwen3-rerank",
                "id": "test-id-456",
                "usage": {"total_tokens": 80},
            }
        else:
            # Old format response (returns only top 2)
            mock_response.json.return_value = {
                "output": {
                    "results": [
                        {"document": {"text": expected_reranked[0]}, "score": 0.9},
                        {"document": {"text": expected_reranked[1]}, "score": 0.5},
                    ]
                }
            }

        mock_post.return_value = mock_response

        client = DashscopeRerank(
            api_key="test_key", model=model, top_n=2, instruct="Be precise"
        )
        reranked = client.compress(documents, query)

        assert reranked == expected_reranked

        # Verify request payload
        mock_post.assert_called_once()
        json_payload = mock_post.call_args[1]["json"]
        assert json_payload["model"] == model

        if is_new_format:
            # New format: parameters at top level
            assert json_payload["query"] == query
            assert json_payload["documents"] == documents
            assert json_payload["top_n"] == 2
            assert json_payload["instruct"] == "Be precise"
            assert "input" not in json_payload
            assert "parameters" not in json_payload
        else:
            # Old format: parameters in parameters field
            assert json_payload["input"]["query"] == query
            assert json_payload["input"]["documents"] == documents
            assert json_payload["parameters"]["return_documents"] is True
            assert json_payload["parameters"]["top_n"] == 2
            assert json_payload["parameters"]["instruct"] == "Be precise"

    @patch("requests.post")
    def test_compress_api_error(self, mock_post):
        """Test handling of API errors (non-2xx status)."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "400 Client Error"
        )
        mock_post.return_value = mock_response

        client = DashscopeRerank(api_key="test_key")

        with pytest.raises(requests.HTTPError, match="400 Client Error"):
            client.compress(["Doc"], "Query")

    @patch("requests.post")
    def test_compress_invalid_response_format(self, mock_post):
        """Test handling of invalid response format."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"invalid": "response"}
        mock_post.return_value = mock_response

        client = DashscopeRerank(api_key="test_key")

        with pytest.raises(KeyError):
            client.compress(["Doc"], "Query")


class TestDashscopeRerankEndpointRouting:
    """Default endpoint selection by model name.

    qwen3-rerank uses the OpenAI-compatible endpoint
    (``/compatible-api/v1/reranks``); gte-rerank-v2 and qwen3-vl-rerank
    use the legacy WebAPI endpoint.
    """

    def test_qwen3_rerank_uses_compatible_url(self):
        client = DashscopeRerank(model="qwen3-rerank", api_key="k")
        assert client.url == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        assert client.is_new_format is True

    def test_gte_rerank_v2_uses_legacy_url(self):
        client = DashscopeRerank(model="gte-rerank-v2", api_key="k")
        assert (
            client.url
            == "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        )
        assert client.is_new_format is False

    def test_qwen3_vl_rerank_uses_legacy_url(self):
        client = DashscopeRerank(model="qwen3-vl-rerank", api_key="k")
        assert (
            client.url
            == "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        )
        assert client.is_new_format is False

    def test_unknown_model_falls_back_to_legacy_url(self):
        client = DashscopeRerank(model="some-future-model", api_key="k")
        assert (
            client.url
            == "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        )
        assert client.is_new_format is False

    def test_explicit_base_url_overrides_default(self):
        client = DashscopeRerank(
            model="qwen3-rerank", api_key="k", base_url="https://custom.example/rerank"
        )
        assert client.url == "https://custom.example/rerank"

    def test_model_name_is_case_insensitive_for_routing(self):
        client = DashscopeRerank(model="QWEN3-RERANK", api_key="k")
        assert client.url == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        assert client.is_new_format is True

    @patch("requests.post")
    def test_qwen3_rerank_uses_compatible_url_in_request(self, mock_post):
        """End-to-end: the new model is called on /compatible-api/v1/reranks."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.1},
            ]
        }
        mock_post.return_value = mock_response

        client = DashscopeRerank(api_key="k", model="qwen3-rerank")
        result = client.compress(["a", "b"], "q")

        # URL
        assert mock_post.call_args.args[0] == (
            "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        )
        # New payload shape
        payload = mock_post.call_args.kwargs["json"]
        assert payload["query"] == "q"
        assert payload["documents"] == ["a", "b"]
        assert "input" not in payload
        assert "parameters" not in payload
        # Result mapping
        assert result == ["a", "b"]

    @patch("requests.post")
    def test_empty_documents_skips_request(self, mock_post):
        client = DashscopeRerank(api_key="k", model="qwen3-rerank")
        result = client.compress([], "q")
        assert result == []
        mock_post.assert_not_called()
