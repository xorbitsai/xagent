"""Embedding and rerank providers meter at the provider boundary.

One row per billed HTTP call: recorded once the provider accepted the call,
before its body is validated, and never for a call it rejected. Covers every
provider, the identity stamped from the configured row, the retry wrapper, and
the search pipeline that calls the rerank provider past its adapter.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.embedding.adapter import create_embedding_adapter
from xagent.core.model.embedding.dashscope import DashScopeEmbedding
from xagent.core.model.embedding.openai import OpenAIEmbedding
from xagent.core.model.embedding.xinference import XinferenceEmbedding
from xagent.core.model.model import EmbeddingModelConfig, RerankModelConfig
from xagent.core.model.rerank import dashscope as dashscope_rerank_module
from xagent.core.model.rerank import xinference as xinference_rerank_module
from xagent.core.model.rerank.adapter import create_rerank_adapter
from xagent.core.model.rerank.dashscope import DashscopeRerank
from xagent.core.model.rerank.xinference import XinferenceRerank


def _response(status: int, body: Any) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = "http://provider.test"
    response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return response


def _rows(manager: TokenContextManager) -> list[dict]:
    return [d for d in manager.get_usage().details if d.get("type") == "media"]


def _tokens(row: dict) -> tuple:
    return (
        row["provider_input_tokens"],
        row.get("provider_text_input_tokens"),
        row.get("provider_image_input_tokens"),
    )


class _Session:
    """Stands in for the ``requests.Session`` the embedding clients hold."""

    def __init__(self, *responses: requests.Response) -> None:
        self.responses = list(responses)
        self.payloads: list[dict] = []
        self.headers: dict = {}

    def post(self, url: str, json: dict) -> requests.Response:
        self.payloads.append(json)
        return self.responses.pop(0)


def _openai(session: _Session) -> OpenAIEmbedding:
    model = OpenAIEmbedding(
        model="text-embedding-3-small", api_key="k", model_id="emb-1"
    )
    model._session = session  # type: ignore[assignment]
    return model


def _dashscope(session: _Session) -> DashScopeEmbedding:
    model = DashScopeEmbedding(api_key="k", model_id="platform/text-embedding-v4")
    model._session = session  # type: ignore[assignment]
    return model


def _vectors(n: int) -> list[dict]:
    return [{"embedding": [0.1, 0.2]} for _ in range(n)]


# --- OpenAI embedding ------------------------------------------------------------


def test_openai_embedding_records_texts_and_reported_tokens():
    session = _Session(
        _response(200, {"data": _vectors(3), "usage": {"prompt_tokens": 42}})
    )
    with TokenContextManager() as manager:
        _openai(session).encode(["a", "b", "c"])

    [row] = _rows(manager)
    assert (row["call_type"], row["unit"], row["quantity"]) == (
        "embedding",
        "texts",
        3.0,
    )
    assert (row["model"], row["model_id"]) == ("text-embedding-3-small", "emb-1")
    assert _tokens(row) == (42, 42, 0)
    assert row["tokens_estimated"] is False


def test_openai_embedding_falls_back_to_total_tokens():
    session = _Session(
        _response(200, {"data": _vectors(1), "usage": {"total_tokens": 9}})
    )
    with TokenContextManager() as manager:
        _openai(session).encode("a")

    [row] = _rows(manager)
    assert (row["quantity"], _tokens(row)) == (1.0, (9, 9, 0))


@pytest.mark.parametrize(
    "usage", [None, {}, {"prompt_tokens": True}, {"prompt_tokens": -1}, "n/a"]
)
def test_embedding_without_a_usable_token_report_is_not_estimated(usage):
    body: dict = {"data": _vectors(2)}
    if usage is not None:
        body["usage"] = usage
    with TokenContextManager() as manager:
        _openai(_Session(_response(200, body))).encode(["a", "b"])

    [row] = _rows(manager)
    assert row["quantity"] == 2.0
    assert row["provider_input_tokens"] == 0
    # No split: the row must read as "tokens unknown", not "zero tokens".
    assert "provider_text_input_tokens" not in row
    assert "provider_image_input_tokens" not in row


def test_embedding_rejected_by_the_provider_records_nothing():
    session = _Session(_response(429, {"error": {"message": "slow down"}}))
    with TokenContextManager() as manager, pytest.raises(RuntimeError):
        _openai(session).encode(["a"])

    assert _rows(manager) == []


@pytest.mark.parametrize(
    "body",
    [b"<html>gateway</html>", {"unexpected": True}],
    ids=["unparsable", "wrong-shape"],
)
def test_embedding_accepted_but_unusable_is_still_billed(body):
    with TokenContextManager() as manager, pytest.raises(RuntimeError):
        _openai(_Session(_response(200, body))).encode(["a", "b"])

    [row] = _rows(manager)
    assert (row["quantity"], row["provider_input_tokens"]) == (2.0, 0)


# --- DashScope embedding ---------------------------------------------------------


def test_dashscope_embedding_records_one_row_per_billed_sub_batch():
    session = _Session(
        *[
            _response(
                200,
                {
                    "output": {"embeddings": _vectors(n)},
                    "usage": {"total_tokens": tokens},
                },
            )
            for n, tokens in ((10, 100), (10, 90), (3, 21))
        ]
    )
    with TokenContextManager() as manager:
        vectors = _dashscope(session).encode([f"t{i}" for i in range(23)])

    assert len(vectors) == 23
    rows = _rows(manager)
    assert [row["quantity"] for row in rows] == [10.0, 10.0, 3.0]
    assert [_tokens(row) for row in rows] == [
        (100, 100, 0),
        (90, 90, 0),
        (21, 21, 0),
    ]
    assert {row["model_id"] for row in rows} == {"platform/text-embedding-v4"}
    assert [len(p["input"]["texts"]) for p in session.payloads] == [10, 10, 3]


def test_dashscope_embedding_failed_sub_batch_keeps_the_billed_ones():
    session = _Session(
        _response(200, {"output": {"embeddings": _vectors(10)}, "usage": {}}),
        _response(503, {"message": "busy"}),
    )
    with TokenContextManager() as manager, pytest.raises(RuntimeError):
        _dashscope(session).encode([f"t{i}" for i in range(15)])

    assert [row["quantity"] for row in _rows(manager)] == [10.0]


# --- Xinference embedding --------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        {"data": [{"embedding": [0.1]}] * 2, "usage": {"prompt_tokens": 11}},
        SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1])] * 2,
            usage=SimpleNamespace(prompt_tokens=11, total_tokens=11),
        ),
    ],
    ids=["dict", "object"],
)
def test_xinference_embedding_records_reported_tokens(result):
    model = XinferenceEmbedding(model="bge-m3", model_id="xinf-emb")
    model._model_handle = SimpleNamespace(create_embedding=lambda input: result)
    with TokenContextManager() as manager:
        model.encode(["a", "b"])

    [row] = _rows(manager)
    assert (row["quantity"], row["model"], row["model_id"]) == (
        2.0,
        "bge-m3",
        "xinf-emb",
    )
    assert _tokens(row) == (11, 11, 0)


def test_xinference_embedding_unusable_result_is_still_billed():
    model = XinferenceEmbedding(model="bge-m3")
    model._model_handle = SimpleNamespace(create_embedding=lambda input: {"data": []})
    with TokenContextManager() as manager, pytest.raises(RuntimeError):
        model.encode(["a"])

    [row] = _rows(manager)
    assert (row["quantity"], row["model_id"]) == (1.0, "")


def test_xinference_embedding_failed_call_records_nothing():
    def fail(input):
        raise RuntimeError("model not launched")

    model = XinferenceEmbedding(model="bge-m3")
    model._model_handle = SimpleNamespace(create_embedding=fail)
    with TokenContextManager() as manager, pytest.raises(RuntimeError):
        model.encode(["a"])

    assert _rows(manager) == []


# --- Identity from the configured row --------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "dashscope", "xinference"])
def test_embedding_adapter_stamps_the_configured_id_on_the_provider(provider):
    def inner_model_id(**overrides: Any) -> str:
        config = EmbeddingModelConfig(
            id="row-7",
            model_name="m",
            model_provider=provider,
            api_key="k",
            **overrides,
        )
        return create_embedding_adapter(config)._inner._embedding_model.model_id

    assert inner_model_id() == "row-7"
    assert inner_model_id(billing_model_id="platform/x") == "platform/x"
    # "" means "bill under no id", not "fall back to id".
    assert inner_model_id(billing_model_id="") == ""


@pytest.mark.parametrize("provider", ["dashscope", "xinference"])
def test_rerank_adapter_stamps_the_configured_id_on_the_provider(provider):
    config = RerankModelConfig(
        id="rr-3", model_name="qwen3-rerank", model_provider=provider, api_key="k"
    )
    assert create_rerank_adapter(config)._inner._rerank_model.model_id == "rr-3"


def test_placeholder_identities_are_never_billed():
    model = OpenAIEmbedding(model="default", api_key="k", model_id="None")
    model._session = _Session(_response(200, {"data": _vectors(1)}))  # type: ignore[assignment]
    with TokenContextManager() as manager:
        model.encode("a")

    [row] = _rows(manager)
    assert (row["model"], row["model_id"]) == ("", "")


# --- Rerank providers ------------------------------------------------------------


@pytest.fixture
def rerank_posts(monkeypatch):
    responses: list[requests.Response] = []
    payloads: list[dict] = []

    def post(url, headers=None, json=None, timeout=None):
        payloads.append(json)
        return responses.pop(0)

    for module in (dashscope_rerank_module, xinference_rerank_module):
        monkeypatch.setattr(module.requests, "post", post)
    return SimpleNamespace(responses=responses, payloads=payloads)


_NEW_FORMAT = {
    "results": [{"index": 1, "relevance_score": 0.9}],
    "usage": {"total_tokens": 31},
}
_OLD_FORMAT = {
    "output": {
        "results": [{"index": 1, "relevance_score": 0.9, "document": {"text": "b"}}]
    },
    "usage": {"total_tokens": 31},
}


@pytest.mark.parametrize("method", ["compress", "compress_with_scores"])
@pytest.mark.parametrize(
    ("model", "body"),
    [("qwen3-rerank", _NEW_FORMAT), ("gte-rerank-v2", _OLD_FORMAT)],
    ids=["new-format", "old-format"],
)
def test_dashscope_rerank_records_one_request(rerank_posts, method, model, body):
    rerank_posts.responses.append(_response(200, body))
    rerank = DashscopeRerank(model=model, api_key="k", model_id="rr-1")
    with TokenContextManager() as manager:
        getattr(rerank, method)(["a", "b"], "q")

    [row] = _rows(manager)
    assert (row["call_type"], row["unit"], row["quantity"]) == (
        "rerank",
        "requests",
        1.0,
    )
    assert (row["model"], row["model_id"], row["provider_input_tokens"]) == (
        model,
        "rr-1",
        31,
    )
    assert "provider_text_input_tokens" not in row


def test_dashscope_rerank_rejected_records_nothing(rerank_posts):
    rerank_posts.responses.append(_response(500, {"message": "boom"}))
    with TokenContextManager() as manager, pytest.raises(requests.HTTPError):
        DashscopeRerank(api_key="k").compress(["a"], "q")

    assert _rows(manager) == []


@pytest.mark.parametrize(
    ("body", "error"),
    [({"no": "results"}, KeyError), (b"oops", ValueError)],
    ids=["wrong-shape", "unparsable"],
)
def test_dashscope_rerank_accepted_but_unusable_is_still_billed(
    rerank_posts, body, error
):
    rerank_posts.responses.append(_response(200, body))
    with TokenContextManager() as manager, pytest.raises(error):
        DashscopeRerank(api_key="k").compress_with_scores(["a"], "q")

    [row] = _rows(manager)
    assert row["provider_input_tokens"] == 0


@pytest.mark.parametrize("method", ["compress", "compress_with_scores"])
def test_xinference_rerank_asks_for_and_records_input_tokens(rerank_posts, method):
    rerank_posts.responses.append(
        _response(
            200,
            {
                "results": [{"index": 0, "relevance_score": 0.5}],
                # Xinference mirrors the input count into output_tokens; it is
                # not a second charge. Distinct here so the field read is pinned.
                "meta": {"tokens": {"input_tokens": 17, "output_tokens": 23}},
            },
        )
    )
    rerank = XinferenceRerank(model="bge-reranker", model_id="xr")
    with TokenContextManager() as manager:
        getattr(rerank, method)(["a", "b"], "q")

    assert rerank_posts.payloads[0]["return_len"] is True
    [row] = _rows(manager)
    assert (row["model_id"], row["provider_input_tokens"]) == ("xr", 17)
    assert row["provider_output_tokens"] == 0


def test_xinference_rerank_without_meta_or_with_unparsable_body(rerank_posts):
    rerank = XinferenceRerank(model="bge-reranker")
    rerank_posts.responses.append(_response(200, {"results": []}))
    rerank_posts.responses.append(_response(200, b"not json"))
    with TokenContextManager() as manager:
        rerank.compress(["a"], "q")
        with pytest.raises(ValueError, match="non-JSON"):
            rerank.compress(["a"], "q")

    assert [row["provider_input_tokens"] for row in _rows(manager)] == [0, 0]


def test_xinference_rerank_rejected_records_nothing(rerank_posts):
    rerank_posts.responses.append(_response(404, {"detail": "no model"}))
    with TokenContextManager() as manager, pytest.raises(requests.HTTPError):
        XinferenceRerank(model="bge-reranker").compress_with_scores(["a"], "q")

    assert _rows(manager) == []


# --- Retry wrapper ---------------------------------------------------------------


def test_rerank_retry_bills_only_the_attempt_the_provider_accepted(
    rerank_posts, monkeypatch
):
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    rerank_posts.responses.extend(
        [
            _response(503, {"message": "busy"}),
            _response(200, {"results": [{"index": 0}], "usage": {"total_tokens": 5}}),
        ]
    )
    adapter = create_rerank_adapter(
        RerankModelConfig(
            id="rr-9", model_name="qwen3-rerank", api_key="k", max_retries=2
        )
    )
    with TokenContextManager() as manager:
        assert list(adapter.compress(["a"], "q")) == ["a"]

    assert len(rerank_posts.payloads) == 2
    [row] = _rows(manager)
    assert (row["model_id"], row["provider_input_tokens"]) == ("rr-9", 5)


# --- Search pipeline: rerank called past the adapter -----------------------------


def _search_result(text: str):
    from xagent.core.tools.core.RAG_tools.core.schemas import SearchResult

    return SearchResult(
        doc_id="d",
        chunk_id=text,
        text=text,
        score=0.5,
        parse_hash="p",
        model_tag="m",
        created_at=datetime.now(timezone.utc),
    )


def test_kb_search_rerank_past_the_adapter_is_billed(rerank_posts, monkeypatch):
    from xagent.core.tools.core.RAG_tools.core.schemas import SearchConfig
    from xagent.core.tools.core.RAG_tools.pipelines import document_search

    config = RerankModelConfig(
        id="kb-rerank", model_name="qwen3-rerank", api_key="k", max_retries=0
    )
    monkeypatch.setattr(
        document_search,
        "resolve_rerank_adapter",
        lambda **_kwargs: (config, create_rerank_adapter(config)),
    )
    rerank_posts.responses.append(_response(200, _NEW_FORMAT))
    with TokenContextManager() as manager:
        reranked, used_rerank, _warnings = document_search._apply_rerank_if_needed(
            [_search_result("a"), _search_result("b")],
            "q",
            SearchConfig(embedding_model_id="e", rerank_model_id="kb-rerank"),
        )

    assert used_rerank is True and reranked[0].text == "b"
    [row] = _rows(manager)
    assert (row["call_type"], row["model_id"]) == ("rerank", "kb-rerank")


def test_env_configured_rerank_records_no_model_id(monkeypatch):
    from xagent.core.tools.core.RAG_tools.pipelines import document_search

    monkeypatch.setenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank")
    monkeypatch.setenv("DASHSCOPE_RERANK_API_KEY", "k")
    monkeypatch.delenv("DASHSCOPE_RERANK_ENABLED", raising=False)
    rerank = document_search._resolve_dashscope_rerank_from_env()
    assert rerank is not None and rerank.model_id == ""


def test_env_configured_models_record_no_model_id(monkeypatch):
    # The env names a provider model, not a configured id a price table knows,
    # matching the env-built DashscopeRerank.
    from xagent.core.tools.core.RAG_tools.utils.model_resolver import (
        resolve_embedding_from_env,
        resolve_rerank_from_env,
    )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("DASHSCOPE_RERANK_MODEL", "gte-rerank-v2")

    assert resolve_embedding_from_env().usage_model_id == ""
    assert resolve_rerank_from_env().usage_model_id == ""
