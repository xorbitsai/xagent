"""Make a request's model resolutions once, for its gate and its pipeline.

The standalone usage gate is told which models a knowledge-base request will
call before the request runs; the pipeline then resolves those models itself,
deep inside the ingestion or search code. Resolution reads state that can
change in between (a collection's binding, the model hub, the environment),
so resolving twice could gate one model and call another.

Under a :class:`ModelResolutionPin`, the first resolution of a given request
is kept -- its failure too -- and every later identical one returns it: the
effective embedding model of a collection, and the config a model id resolves
to. The entry point resolves under the pin to learn the model ids, gates on
them, and runs the pipeline under the same pin. Adapters are not kept; each
resolution still builds a fresh one from the kept config.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Hashable, Iterator, Optional, TypeVar, cast

_T = TypeVar("_T")

_active_pin: ContextVar[Optional["ModelResolutionPin"]] = ContextVar(
    "rag_model_resolution_pin", default=None
)


class ModelResolutionPin:
    """The resolutions made while it is bound (:meth:`bound`, :meth:`run`).

    Carried wherever the context is (``asyncio.to_thread``,
    ``contextvars.copy_context().run``); work handed to a bare executor must
    bind it again with :meth:`run`. Safe to share between threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[Hashable, Any] = {}
        self._failures: dict[Hashable, Exception] = {}

    @contextmanager
    def bound(self) -> Iterator["ModelResolutionPin"]:
        """Bind this pin for the body of a ``with`` block."""
        token = _active_pin.set(self)
        try:
            yield self
        finally:
            _active_pin.reset(token)

    def run(self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
        """Call ``fn(*args, **kwargs)`` with this pin bound."""
        with self.bound():
            return fn(*args, **kwargs)

    def _resolve(self, key: Hashable, compute: Callable[[], _T]) -> _T:
        with self._lock:
            known = key in self._results or key in self._failures
        if not known:
            # Outside the lock: resolution may hop threads. Two threads racing
            # on one key both compute, and both use whichever lands first.
            try:
                outcome: tuple[Any, Exception | None] = (compute(), None)
            except Exception as exc:  # noqa: BLE001 - replayed below
                outcome = (None, exc)
            with self._lock:
                if key not in self._results and key not in self._failures:
                    if outcome[1] is None:
                        self._results[key] = outcome[0]
                    else:
                        self._failures[key] = outcome[1]
        with self._lock:
            failure = self._failures.get(key)
            if failure is not None:
                raise failure
            return cast(_T, self._results[key])


def pinned_resolution(key: Hashable, compute: Callable[[], _T]) -> _T:
    """``compute()``, or what it returned (or raised) earlier under the pin.

    ``key`` identifies the resolution and everything its result depends on
    that the caller passes in. Without a bound pin this is ``compute()``.
    """
    pin = _active_pin.get()
    if pin is None:
        return compute()
    return pin._resolve(key, compute)
