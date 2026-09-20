"""#796 - search resolves through the coordinator and never touches a backend.

The coordinator owns collection-context resolution for search exactly as it
does for the other data-plane families; the legacy facade and the public
``retrieval.search_*`` functions are thin adapters over it.

Every case asserts the *full* keyword set, so a parameter silently dropped at
any of the three hops fails here rather than degrading a search at runtime.
"""

import asyncio
import importlib
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
from xagent.core.tools.core.RAG_tools.kb.legacy_step_compatibility import (
    KBLegacyStepCompatibilityFacade,
)
from xagent.core.tools.core.RAG_tools.kb.models import KBAccessMode

COMMON = {
    "top_k": 4,
    "filters": {"field": "value"},
    "readonly": True,
    "nprobes": 7,
    "refine_factor": 3,
    "user_id": 5,
    "is_admin": True,
}
FUSION = object()

# (coordinator method, handle method, facade method, retrieval module,
#  public function, args after `collection`, kwargs, is_async)
CASES = [
    (
        "search_dense_sync",
        "search_dense",
        "search_dense",
        "search_dense",
        "search_dense",
        ("model-x", [0.1]),
        COMMON,
        False,
    ),
    (
        "search_dense",
        "search_dense_async",
        "search_dense_async",
        "search_dense",
        "search_dense_async",
        ("model-x", [0.1]),
        COMMON,
        True,
    ),
    (
        "search_sparse_sync",
        "search_sparse",
        "search_sparse",
        "search_sparse",
        "search_sparse",
        ("model-y", "query text"),
        COMMON,
        False,
    ),
    (
        "search_sparse",
        "search_sparse_async",
        "search_sparse_async",
        "search_sparse",
        "search_sparse_async",
        ("model-y", "query text"),
        COMMON,
        True,
    ),
    (
        "search_hybrid_sync",
        "search_hybrid",
        "search_hybrid",
        "search_hybrid",
        "search_hybrid",
        ("model-z", "query", [0.3, 0.4]),
        {**COMMON, "fusion_config": FUSION},
        False,
    ),
    (
        "search_hybrid",
        "search_hybrid_async",
        None,
        None,
        None,
        ("model-z", "query", [0.3, 0.4]),
        {**COMMON, "fusion_config": FUSION},
        True,
    ),
]
IDS = [f"{c[0]}" for c in CASES]


async def _async_return(value):
    return value


def _handle_for(method: str, is_async: bool) -> MagicMock:
    handle = MagicMock()
    if is_async:
        getattr(handle, method).side_effect = lambda *a, **k: _async_return(MagicMock())
    return handle


def _coordinator_with_handle(handle: MagicMock) -> KBCoordinator:
    # Bare instance: the search entry points only open a handle and delegate,
    # so a coordinator with no store wiring proves nothing else is reached.
    coordinator = KBCoordinator.__new__(KBCoordinator)
    coordinator.open_collection_sync = MagicMock(return_value=handle)
    coordinator.open_collection = MagicMock(
        side_effect=lambda _request: _async_return(handle)
    )
    return coordinator


def _run(bound, is_async, *args, **kwargs):
    return asyncio.run(bound(*args, **kwargs)) if is_async else bound(*args, **kwargs)


@pytest.mark.parametrize(
    "coord_method, handle_method, _f, _m, _p, args, kwargs, is_async", CASES, ids=IDS
)
def test_coordinator_opens_a_read_handle_and_forwards_every_argument(
    coord_method, handle_method, _f, _m, _p, args, kwargs, is_async
):
    handle = _handle_for(handle_method, is_async)
    coordinator = _coordinator_with_handle(handle)

    _run(getattr(coordinator, coord_method), is_async, "col", *args, **kwargs)

    opener = (
        coordinator.open_collection if is_async else coordinator.open_collection_sync
    )
    request = opener.call_args.args[0]
    assert request.collection == "col"
    assert request.access_mode == KBAccessMode.READ
    assert request.user_id == kwargs["user_id"]
    assert request.is_admin == kwargs["is_admin"]
    assert request.hide_missing is True

    getattr(handle, handle_method).assert_called_once_with(*args, **kwargs)


@pytest.mark.parametrize(
    "coord_method, handle_method, _f, _m, _p, args, kwargs, is_async", CASES, ids=IDS
)
def test_sync_entry_points_never_take_the_async_handle_path(
    coord_method, handle_method, _f, _m, _p, args, kwargs, is_async
):
    """A sync caller must not be pushed onto the async handle-opening path."""
    handle = _handle_for(handle_method, is_async)
    coordinator = _coordinator_with_handle(handle)

    _run(getattr(coordinator, coord_method), is_async, "col", *args, **kwargs)

    unused = (
        coordinator.open_collection
        if not is_async
        else coordinator.open_collection_sync
    )
    unused.assert_not_called()


@pytest.mark.parametrize(
    "coord_method, _h, facade_method, _m, _p, args, kwargs, is_async",
    [c for c in CASES if c[2] is not None],
    ids=[c[2] for c in CASES if c[2] is not None],
)
def test_legacy_facade_forwards_every_argument_to_the_coordinator(
    coord_method, _h, facade_method, _m, _p, args, kwargs, is_async
):
    facade = KBLegacyStepCompatibilityFacade()
    coordinator = MagicMock()
    if is_async:
        getattr(coordinator, coord_method).side_effect = lambda *a, **k: _async_return(
            MagicMock()
        )

    with patch.object(facade, "_active_coordinator", return_value=coordinator):
        _run(getattr(facade, facade_method), is_async, "col", *args, **kwargs)

    getattr(coordinator, coord_method).assert_called_once_with("col", *args, **kwargs)


@pytest.mark.parametrize(
    "coord_method, handle_method, _f, module_name, public_name, args, kwargs, is_async",
    [c for c in CASES if c[3] is not None],
    ids=[c[4] for c in CASES if c[3] is not None],
)
def test_public_retrieval_functions_route_through_the_coordinator(
    coord_method, handle_method, _f, module_name, public_name, args, kwargs, is_async
):
    """A real coordinator here, not a mock: it also pins the keyword signatures."""
    module = importlib.import_module(
        f"xagent.core.tools.core.RAG_tools.retrieval.{module_name}"
    )
    handle = _handle_for(handle_method, is_async)
    coordinator = _coordinator_with_handle(handle)

    with patch.object(module, "_get_coordinator", return_value=coordinator):
        _run(getattr(module, public_name), is_async, "col", *args, **kwargs)

    getattr(handle, handle_method).assert_called_once_with(*args, **kwargs)
