"""Search input errors must be raised before coordinator/storage access (#671)."""

import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    DocumentValidationError,
    VectorValidationError,
)
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)

PUBLIC_MODES = ["dense", "dense_async", "sparse", "sparse_async", "hybrid"]


def public_search(mode, monkeypatch):
    name = mode.removesuffix("_async")
    module = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.retrieval.search_" + name
    )
    coordinator = Mock()
    method = "search_" + name + ("" if mode.endswith("async") else "_sync")
    delegate = (
        AsyncMock(return_value="result")
        if mode.endswith("async")
        else Mock(return_value="result")
    )
    setattr(coordinator, method, delegate)
    getter = Mock(return_value=coordinator)
    monkeypatch.setattr(module, "_get_coordinator", getter)
    kwargs = dict(collection="col1", model_tag="test", top_k=10)
    if name != "dense":
        kwargs["query_text"] = "query"
    if name != "sparse":
        kwargs["query_vector"] = [0.1, 0.2]
    return getattr(module, "search_" + mode), kwargs, getter


async def invoke(function, kwargs):
    result = function(**kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest.mark.parametrize("mode", PUBLIC_MODES)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("collection", ""),
        ("collection", None),
        ("model_tag", ""),
        ("model_tag", None),
        ("top_k", 0),
        ("top_k", 1001),
    ],
)
async def test_common_input_validation(mode, field, value, monkeypatch):
    function, kwargs, getter = public_search(mode, monkeypatch)
    kwargs[field] = value
    with pytest.raises(DocumentValidationError):
        await invoke(function, kwargs)
    getter.assert_not_called()


@pytest.mark.parametrize("mode", PUBLIC_MODES)
@pytest.mark.parametrize("top_k", [1, 1000])
async def test_top_k_boundaries_are_accepted(mode, top_k, monkeypatch):
    function, kwargs, getter = public_search(mode, monkeypatch)
    kwargs["top_k"] = top_k
    assert await invoke(function, kwargs) == "result"
    getter.assert_called_once()


@pytest.mark.parametrize("mode", ["sparse", "sparse_async", "hybrid"])
@pytest.mark.parametrize("query_text", ["", None, 42])
async def test_text_input_validation(mode, query_text, monkeypatch):
    function, kwargs, getter = public_search(mode, monkeypatch)
    kwargs["query_text"] = query_text
    with pytest.raises(DocumentValidationError):
        await invoke(function, kwargs)
    getter.assert_not_called()


@pytest.mark.parametrize("mode", ["dense", "dense_async", "hybrid"])
@pytest.mark.parametrize(
    "query_vector", [[], None, ["bad"], [float("nan")], [float("inf")]]
)
async def test_vector_input_uses_dense_exception(mode, query_vector, monkeypatch):
    function, kwargs, getter = public_search(mode, monkeypatch)
    kwargs["query_vector"] = query_vector
    with pytest.raises(VectorValidationError):
        await invoke(function, kwargs)
    getter.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("collection", "", DocumentValidationError),
        ("model_tag", "", DocumentValidationError),
        ("top_k", 0, DocumentValidationError),
        ("top_k", 1001, DocumentValidationError),
        ("query_text", "", DocumentValidationError),
        ("query_vector", [], VectorValidationError),
        ("query_vector", [float("nan")], VectorValidationError),
    ],
)
async def test_internal_hybrid_validates_before_children(
    async_mode, field, value, error, monkeypatch
):
    handle = LanceDBCollectionHandle.__new__(LanceDBCollectionHandle)
    object.__setattr__(
        handle,
        "context",
        SimpleNamespace(
            collection=value if field == "collection" else "col1",
            capabilities=SimpleNamespace(supports_search=True),
        ),
    )
    suffix = "_async" if async_mode else ""
    dense = Mock(side_effect=AssertionError("dense must not run"))
    sparse = Mock(side_effect=AssertionError("sparse must not run"))
    monkeypatch.setattr(LanceDBCollectionHandle, "search_dense" + suffix, dense)
    monkeypatch.setattr(LanceDBCollectionHandle, "search_sparse" + suffix, sparse)
    kwargs = dict(model_tag="test", query_text="query", query_vector=[0.1], top_k=10)
    if field != "collection":
        kwargs[field] = value
    with pytest.raises(error):
        await invoke(getattr(handle, "search_hybrid" + suffix), kwargs)
    dense.assert_not_called()
    sparse.assert_not_called()
