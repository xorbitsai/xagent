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

Failures are kept as deliberately as results. If a resolution failed while
the gate was asked, the gate was told the work calls nothing; resolving
again could succeed and call a priced model the gate never saw. The cost is
that a pin outlives one call: for an ``/ingest-web`` crawl, an
``/ingest-cloud`` batch or a background job, a hub error -- transient or
not, including one that made resolution fall back to the environment --
holds for every page or file, where each used to resolve on its own.
Likewise, a batch into an unbound collection keeps the model it resolved
first: if a concurrent ingestion binds the collection to another model,
the remaining files fail as already initialized with that model instead of
adopting it, which would call a model the gate was not told about.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    Any,
    Callable,
    Hashable,
    Iterator,
    NoReturn,
    Optional,
    TypeVar,
    cast,
)

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

    def __init__(self, *, enabled: bool = True) -> None:
        # A disabled pin binds nothing: resolution is as if there were none.
        # For work no gate is asked about, which has nothing to keep.
        self._enabled = enabled
        self._lock = threading.Lock()
        # key -> (result, None) or (None, the exception it raised)
        self._outcomes: dict[Hashable, tuple[Any, Optional[Exception]]] = {}

    @contextmanager
    def bound(self) -> Iterator["ModelResolutionPin"]:
        """Bind this pin for the body of a ``with`` block."""
        if not self._enabled:
            yield self
            return
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
            outcome = self._outcomes.get(key)
        if outcome is None:
            # Outside the lock: resolution may hop threads. Two threads racing
            # on one key both compute, and both use whichever lands first.
            try:
                computed: tuple[Any, Optional[Exception]] = (compute(), None)
            except Exception as exc:  # noqa: BLE001 - replayed below
                computed = (None, exc)
            with self._lock:
                outcome = self._outcomes.setdefault(key, computed)
        result, failure = outcome
        if failure is not None:
            _raise_copy(failure)
        return cast(_T, result)


def _raise_copy(failure: Exception) -> NoReturn:
    """Raise ``failure`` again as a copy, chained to the kept original.

    One kept exception may be replayed many times, from many threads;
    raising that instance each time would keep growing its traceback. The
    copy is built without ``__init__``, which may reformat the message.
    """
    cls = type(failure)
    try:
        replay = cls.__new__(cls, *failure.args)
        replay.__dict__.update(vars(failure))
        replay.args = failure.args
    except Exception:  # noqa: BLE001 - not copyable: raise the original
        raise failure.with_traceback(None)
    raise replay from failure


def pinned_resolution(key: Hashable, compute: Callable[[], _T]) -> _T:
    """``compute()``, or what it returned (or raised) earlier under the pin.

    ``key`` identifies the resolution and everything its result depends on
    that the caller passes in. Without a bound pin this is ``compute()``.
    """
    pin = _active_pin.get()
    if pin is None:
        return compute()
    return pin._resolve(key, compute)
