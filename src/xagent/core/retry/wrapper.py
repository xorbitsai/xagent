import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    Type,
    TypeVar,
    cast,
    runtime_checkable,
)

from .strategy import ExponentialBackoff, RetryStrategy

logger = logging.getLogger(__name__)


@runtime_checkable
class Retryable(Protocol):
    def invoke(self, *args: Any, **kwargs: Any) -> Any: ...

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any: ...


class RetryWrapper(Retryable):
    def __init__(
        self,
        target: Retryable,
        strategy: RetryStrategy = ExponentialBackoff(),
        max_retries: int = 10,
        retry_on: Callable[[Exception], bool] = lambda _: True,
    ):
        self.target = target
        self.strategy = strategy
        self.max_retries = max_retries
        self.retry_on = retry_on

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                return self.target.invoke(*args, **kwargs)
            except Exception as e:
                if not self.retry_on(e):
                    raise

                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.strategy.get_delay(attempt) / 1000.0
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Retry failed with no exception")

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                return await self.target.ainvoke(*args, **kwargs)
            except Exception as e:
                if not self.retry_on(e):
                    raise

                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.strategy.get_delay(attempt) / 1000.0
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Retry failed with no exception")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.invoke(*args, **kwargs)


T = TypeVar("T")


def create_retry_wrapper(
    inner: T,
    base_class: Type[T],
    *,
    retry_methods: set[str] | None = None,
    strategy: RetryStrategy | None = None,
    max_retries: int = 10,
    retry_on: Callable[[Exception], bool] = lambda _: True,
) -> T:
    # 1. Ensure we implement the mandatory abstract methods of Runnable
    methods_to_implement = retry_methods or set()

    class GenericRetryWrapper(base_class):  # type: ignore
        def __init__(self) -> None:
            self._inner = inner
            self._retry_wrapper = RetryWrapper(
                target=_GenericRetryableTarget(inner),
                strategy=strategy or ExponentialBackoff(),
                max_retries=max_retries,
                retry_on=retry_on,
            )

        def __getattr__(self, name: str) -> Any:
            # This only handles attributes NOT present in base_class
            return getattr(self._inner, name)

    # 2. Implement the RETRY methods (invoke/ainvoke)
    for name in methods_to_implement:
        original_method = getattr(inner, name, None)

        # IMPORTANT: Check for async generators FIRST, before iscoroutinefunction
        # Because iscoroutinefunction returns False for async generators
        if original_method and inspect.isasyncgenfunction(original_method):
            # For async generators, create a wrapper that retries on failure
            def make_async_gen_retry_wrapper(method_name: str) -> Any:
                def _async_gen_wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    # Define an inner async generator that handles retries
                    async def _retry_generator() -> AsyncIterator[Any]:
                        last_exception: Optional[Exception] = None

                        for attempt in range(self._retry_wrapper.max_retries):
                            yielded_item = False
                            try:
                                # Get a new async generator for each attempt
                                async_gen = getattr(self._inner, method_name)(
                                    *args, **kwargs
                                )

                                # Yield items from the generator
                                async for item in async_gen:
                                    yielded_item = True
                                    yield item

                                # If we get here, the generator completed successfully
                                return

                            except Exception as e:
                                if yielded_item:
                                    raise

                                if not self._retry_wrapper.retry_on(e):
                                    raise

                                last_exception = e
                                if attempt < self._retry_wrapper.max_retries - 1:
                                    delay = (
                                        self._retry_wrapper.strategy.get_delay(attempt)
                                        / 1000.0
                                    )
                                    logger.warning(
                                        f"Async generator {method_name} attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s..."
                                    )
                                    await asyncio.sleep(delay)
                                else:
                                    logger.error(
                                        f"Async generator {method_name} failed after {self._retry_wrapper.max_retries} attempts"
                                    )

                        # If we exhausted all retries, raise the last exception
                        if last_exception:
                            raise last_exception
                        raise RuntimeError(
                            "Async generator retry failed with no exception"
                        )

                    # Return the async generator
                    return _retry_generator()

                return _async_gen_wrapped

            setattr(GenericRetryWrapper, name, make_async_gen_retry_wrapper(name))
        elif original_method and inspect.iscoroutinefunction(original_method):
            # Regular async functions
            def make_async_retry_wrapper(method_name: str) -> Any:
                async def _async_wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    return await self._retry_wrapper.ainvoke(
                        method=method_name, args=args, kwargs=kwargs
                    )

                return _async_wrapped

            setattr(GenericRetryWrapper, name, make_async_retry_wrapper(name))
        elif original_method:
            # Sync functions
            def make_sync_retry_wrapper(method_name: str) -> Any:
                def _sync_wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    return self._retry_wrapper.invoke(
                        method=method_name, args=args, kwargs=kwargs
                    )

                return _sync_wrapped

            setattr(GenericRetryWrapper, name, make_sync_retry_wrapper(name))

    # 3. Implement DELEGATE methods (Fix for Shadowing)
    # We iterate over inner's attributes. If base_class also has them,
    # we must overwrite them to ensure they delegate to _inner.
    for name in dir(inner):
        if name.startswith("_"):
            continue  # Skip private
        if name in methods_to_implement:
            continue  # Already handled above

        # If the base class has this method, inheritance will hide it from __getattr__.
        # We must explicitly forward it.
        if hasattr(base_class, name):
            attr = getattr(inner, name)

            if not callable(attr):
                # Handle properties
                def make_prop(prop_name: str) -> Any:
                    return property(fget=lambda self: getattr(self._inner, prop_name))

                setattr(GenericRetryWrapper, name, make_prop(name))
                continue

            if inspect.iscoroutinefunction(attr):

                def make_async_delegate(method_name: str) -> Any:
                    async def _async_delegate(
                        self: Any, *args: Any, **kwargs: Any
                    ) -> Any:
                        return await getattr(self._inner, method_name)(*args, **kwargs)

                    return _async_delegate

                setattr(GenericRetryWrapper, name, make_async_delegate(name))
            else:

                def make_sync_delegate(method_name: str) -> Any:
                    def _sync_delegate(self: Any, *args: Any, **kwargs: Any) -> Any:
                        return getattr(self._inner, method_name)(*args, **kwargs)

                    return _sync_delegate

                setattr(GenericRetryWrapper, name, make_sync_delegate(name))

    # 4. Forcefully clear abstract methods
    # Since we are a dynamic proxy that delegates everything to 'inner',
    # we satisfy all interfaces that 'inner' satisfies.
    if hasattr(GenericRetryWrapper, "__abstractmethods__"):
        GenericRetryWrapper.__abstractmethods__ = frozenset()

    return cast(T, GenericRetryWrapper())


class _GenericRetryableTarget(Retryable):
    """Generic retryable target."""

    def __init__(self, inner: Any):
        self.inner = inner

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        method = kwargs.pop("method")
        method_args = kwargs.pop("args", ())
        method_kwargs = kwargs.pop("kwargs", {})
        return getattr(self.inner, method)(*method_args, **method_kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        method = kwargs.pop("method")
        method_args = kwargs.pop("args", ())
        method_kwargs = kwargs.pop("kwargs", {})
        return await getattr(self.inner, method)(*method_args, **method_kwargs)
