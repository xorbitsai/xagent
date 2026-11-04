from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Protocol

import pytest

from xagent.core.retry.strategy import FixedDelay
from xagent.core.retry.wrapper import create_retry_wrapper


class SimpleProtocol(Protocol):
    def process(self, value: str) -> str: ...
    async def aprocess(self, value: str) -> str: ...


class StreamProtocol(Protocol):
    def stream(self) -> AsyncIterator[str]: ...


class ConcreteImpl:
    def __init__(self):
        self.call_count = 0
        self.async_call_count = 0

    def process(self, value: str) -> str:
        self.call_count += 1
        if self.call_count <= 2:
            raise ValueError("Mock failure")
        return f"processed: {value}"

    async def aprocess(self, value: str) -> str:
        self.async_call_count += 1
        if self.async_call_count <= 2:
            raise ValueError("Mock async failure")
        return f"async processed: {value}"


class BaseRunnable(ABC):
    @abstractmethod
    def invoke(self, input: str) -> str: ...

    @abstractmethod
    async def ainvoke(self, input: str) -> str: ...

    def helper_method(self) -> str:
        return "helper"


class RunnableImpl(BaseRunnable):
    def __init__(self):
        self.invoke_count = 0
        self.ainvoke_count = 0

    def invoke(self, input: str) -> str:
        self.invoke_count += 1
        if self.invoke_count <= 1:
            raise ValueError("Invoke failure")
        return f"invoked: {input}"

    async def ainvoke(self, input: str) -> str:
        self.ainvoke_count += 1
        if self.ainvoke_count <= 1:
            raise ValueError("Ainvoke failure")
        return f"ainvoked: {input}"


def test_create_retry_wrapper_sync_method():
    impl = ConcreteImpl()
    wrapper = create_retry_wrapper(
        impl,
        SimpleProtocol,
        retry_methods={"process"},
        strategy=FixedDelay(delay_ms=10),
        max_retries=5,
    )

    result = wrapper.process("test")

    assert result == "processed: test"
    assert impl.call_count == 3


@pytest.mark.asyncio
async def test_create_retry_wrapper_async_method():
    impl = ConcreteImpl()
    wrapper = create_retry_wrapper(
        impl,
        SimpleProtocol,
        retry_methods={"aprocess"},
        strategy=FixedDelay(delay_ms=10),
        max_retries=5,
    )

    result = await wrapper.aprocess("test")

    assert result == "async processed: test"
    assert impl.async_call_count == 3


def test_create_retry_wrapper_with_abstract_base():
    impl = RunnableImpl()
    wrapper = create_retry_wrapper(
        impl,
        BaseRunnable,
        retry_methods={"invoke", "ainvoke"},
        strategy=FixedDelay(delay_ms=10),
        max_retries=3,
    )

    result = wrapper.invoke("test")

    assert result == "invoked: test"
    assert impl.invoke_count == 2


@pytest.mark.asyncio
async def test_create_retry_wrapper_abstract_async():
    impl = RunnableImpl()
    wrapper = create_retry_wrapper(
        impl,
        BaseRunnable,
        retry_methods={"invoke", "ainvoke"},
        strategy=FixedDelay(delay_ms=10),
        max_retries=3,
    )

    result = await wrapper.ainvoke("test")

    assert result == "ainvoked: test"
    assert impl.ainvoke_count == 2


def test_create_retry_wrapper_delegates_non_retry_methods():
    impl = RunnableImpl()
    wrapper = create_retry_wrapper(
        impl, BaseRunnable, retry_methods={"invoke"}, max_retries=3
    )

    # helper_method should be delegated without retry
    result = wrapper.helper_method()

    assert result == "helper"


def test_create_retry_wrapper_delegates_non_retry_methods_no_retry():
    class BaseWithHelper(BaseRunnable):
        def helper_method(self) -> str:
            return "base-helper"

    class Impl(RunnableImpl):
        def __init__(self):
            super().__init__()
            self.helper_calls = 0

        def helper_method(self) -> str:
            self.helper_calls += 1
            # If wrapper incorrectly retries non-retry methods, this would eventually succeed.
            if self.helper_calls == 1:
                raise ValueError("helper failed once")
            return "inner-helper"

    impl = Impl()
    wrapper = create_retry_wrapper(
        impl,
        BaseWithHelper,
        retry_methods={"invoke"},
        max_retries=5,
        strategy=FixedDelay(delay_ms=1),
    )

    with pytest.raises(ValueError, match="helper failed once"):
        wrapper.helper_method()

    assert impl.helper_calls == 1  # no retry happened


def test_create_retry_wrapper_retry_on_predicate():
    class SelectiveError(Exception):
        pass

    class FailingImpl:
        def __init__(self):
            self.count = 0

        def process(self, value: str) -> str:
            self.count += 1
            if self.count == 1:
                raise ValueError("Retryable")
            if self.count == 2:
                raise SelectiveError("Non-retryable")
            return "success"

    impl = FailingImpl()
    wrapper = create_retry_wrapper(
        impl,
        SimpleProtocol,
        retry_methods={"process"},
        max_retries=5,
        retry_on=lambda e: isinstance(e, ValueError),
    )

    with pytest.raises(SelectiveError, match="Non-retryable"):
        wrapper.process("test")

    assert impl.count == 2


def test_create_retry_wrapper_preserves_attributes():
    class ImplWithAttr:
        custom_attr = "custom_value"

        def process(self, value: str) -> str:
            return value

    impl = ImplWithAttr()
    wrapper = create_retry_wrapper(impl, SimpleProtocol, retry_methods={"process"})

    assert wrapper.custom_attr == "custom_value"


@pytest.mark.asyncio
async def test_create_retry_wrapper_retries_async_generator_before_first_item():
    class FlakyStreamImpl:
        def __init__(self):
            self.call_count = 0

        async def stream(self) -> AsyncIterator[str]:
            self.call_count += 1
            if self.call_count == 1:
                raise ValueError("stream failed before payload")
            yield "ok"

    impl = FlakyStreamImpl()
    wrapper = create_retry_wrapper(
        impl,
        StreamProtocol,
        retry_methods={"stream"},
        strategy=FixedDelay(delay_ms=1),
        max_retries=2,
        retry_on=lambda e: isinstance(e, ValueError),
    )

    chunks = [chunk async for chunk in wrapper.stream()]

    assert chunks == ["ok"]
    assert impl.call_count == 2


@pytest.mark.asyncio
async def test_create_retry_wrapper_does_not_retry_async_generator_after_item():
    class FailingAfterYieldStreamImpl:
        def __init__(self):
            self.call_count = 0

        async def stream(self) -> AsyncIterator[str]:
            self.call_count += 1
            yield f"chunk-{self.call_count}"
            raise ValueError("stream failed after payload")

    impl = FailingAfterYieldStreamImpl()
    wrapper = create_retry_wrapper(
        impl,
        StreamProtocol,
        retry_methods={"stream"},
        strategy=FixedDelay(delay_ms=1),
        max_retries=2,
        retry_on=lambda e: isinstance(e, ValueError),
    )

    chunks = []
    with pytest.raises(ValueError, match="stream failed after payload"):
        async for chunk in wrapper.stream():
            chunks.append(chunk)

    assert chunks == ["chunk-1"]
    assert impl.call_count == 1


def test_create_retry_wrapper_no_retry_methods():
    impl = ConcreteImpl()
    wrapper = create_retry_wrapper(
        impl, SimpleProtocol, retry_methods=set(), max_retries=1
    )

    # Should fail immediately without retry
    with pytest.raises(ValueError):
        wrapper.process("test")

    assert impl.call_count == 1


def test_create_retry_wrapper_delegates_abstract_properties():
    class BaseWithProperty(ABC):
        @property
        @abstractmethod
        def model_name(self) -> str:
            pass

        @abstractmethod
        def invoke(self) -> str:
            pass

    class ImplWithProperty(BaseWithProperty):
        @property
        def model_name(self) -> str:
            return "deepseek"

        def invoke(self) -> str:
            return "done"

    impl = ImplWithProperty()
    wrapper = create_retry_wrapper(impl, BaseWithProperty, retry_methods={"invoke"})

    assert wrapper.model_name == "deepseek"
    assert wrapper.invoke() == "done"
