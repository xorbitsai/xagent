"""Repository-level coverage for one shared invariant across non-OpenAI
chat clients: no ``_xagent_``-prefixed internal message key ever reaches a
provider's wire request.

``_strip_internal_message_keys`` (now on ``BaseLLM``, see basic/base.py) is
the mechanism for clients that forward a message dict's keys mostly
unchanged (Zhipu, Xinference). Claude and Gemini take a different, equally
valid path to the same result: their message-conversion functions rebuild
each provider message field-by-field and never copy a message dict
wholesale, so an internal key is simply never read in the first place. Each
check below asserts the *result* (no leaked key), not the mechanism, so a
provider that later switches mechanisms without breaking the outcome does
not need this file to change.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

import pytest

import xagent.core.model.chat.basic as basic_pkg
from xagent.core.model.chat.basic.azure_openai import AzureOpenAILLM
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.claude import ClaudeLLM
from xagent.core.model.chat.basic.dashscope import DashScopeLLM
from xagent.core.model.chat.basic.deepseek import DeepSeekLLM
from xagent.core.model.chat.basic.gemini import GeminiLLM
from xagent.core.model.chat.basic.openai import OpenAICompatibleLLM, OpenAILLM
from xagent.core.model.chat.basic.openrouter import OpenRouterLLM
from xagent.core.model.chat.basic.router import RouterLLM, _ResolvedRouterLLM
from xagent.core.model.chat.basic.xinference import XinferenceLLM
from xagent.core.model.chat.basic.zhipu import ZhipuLLM
from xagent.core.model.chat.types import PROVIDER_STATE_METADATA_KEY

_MARKED_HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "Search xagent"},
    {
        "role": "assistant",
        "content": "",
        PROVIDER_STATE_METADATA_KEY: {"deepseek": {"reasoning_content": "prior"}},
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "result"},
]


def _assert_no_internal_keys(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        for key in message:
            assert not key.startswith("_xagent_"), (
                f"leaked internal key {key!r} in outbound message {message!r}"
            )


def test_claude_message_conversion_drops_internal_keys(claude_llm_config):
    """Claude: safety comes from rebuilding each message field-by-field, not
    from an explicit strip call -- see the comment in
    ``_convert_messages_to_anthropic_format``.
    """
    llm = ClaudeLLM(**claude_llm_config)

    _system, anthropic_messages = llm._convert_messages_to_anthropic_format(
        _MARKED_HISTORY
    )

    _assert_no_internal_keys(anthropic_messages)


def test_gemini_message_conversion_drops_internal_keys(gemini_llm_config):
    """Gemini: same field-by-field rebuild guarantee as Claude."""
    llm = GeminiLLM(**gemini_llm_config)

    _system, gemini_messages = llm._convert_messages_to_gemini_format(_MARKED_HISTORY)

    _assert_no_internal_keys(gemini_messages)


@pytest.mark.asyncio
async def test_zhipu_chat_strips_internal_keys_before_the_sdk_call(mocker):
    """Zhipu: forwards message dicts close to unchanged, so it needs (and
    now has) an explicit ``_strip_internal_message_keys`` call.
    """
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "done"
    mock_response.usage = None
    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch(
        "xagent.core.model.chat.basic.zhipu.ZhipuAiClient",
        return_value=mock_client,
    )
    llm = ZhipuLLM(api_key="test-key")

    await llm.chat(_MARKED_HISTORY)

    sent_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    _assert_no_internal_keys(sent_messages)


@pytest.mark.asyncio
async def test_zhipu_vision_chat_strips_internal_keys_before_the_sdk_call(mocker):
    """Zhipu: ``vision_chat`` assembles its own request instead of delegating
    to ``chat``, so it carries its own strip call and needs its own check.
    Registering the class once is not the same as exercising each entrypoint.
    """
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "done"
    mock_response.usage = None
    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch(
        "xagent.core.model.chat.basic.zhipu.ZhipuAiClient",
        return_value=mock_client,
    )
    llm = ZhipuLLM(
        model_name="glm-4.5v",
        api_key="test-key",
        abilities=["chat", "tool_calling", "vision"],
    )

    await llm.vision_chat(_MARKED_HISTORY)

    sent_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    _assert_no_internal_keys(sent_messages)


@pytest.mark.asyncio
async def test_zhipu_stream_chat_strips_internal_keys_before_the_sdk_call(mocker):
    """Zhipu: ``stream_chat`` is the third entrypoint that builds its own
    request, on a separate producer-thread code path from ``chat``.
    """
    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = []
    mocker.patch(
        "xagent.core.model.chat.basic.zhipu.ZhipuAiClient",
        return_value=mock_client,
    )
    llm = ZhipuLLM(api_key="test-key")

    async for _chunk in llm.stream_chat(_MARKED_HISTORY):
        pass

    sent_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    _assert_no_internal_keys(sent_messages)


@pytest.mark.asyncio
async def test_xinference_chat_strips_internal_keys_before_the_sdk_call(mocker):
    """Xinference: same wholesale-forwarding shape as Zhipu, same fix."""

    class _ModelHandle:
        async def chat(self, **kwargs: Any):
            self.received_messages = kwargs["messages"]
            return {
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]
            }

    llm = XinferenceLLM(model_name="qwen3.8")
    handle = _ModelHandle()
    llm._client = mocker.MagicMock()
    llm._model_handle = handle

    await llm.chat(_MARKED_HISTORY)

    _assert_no_internal_keys(handle.received_messages)


@pytest.mark.asyncio
async def test_xinference_stream_chat_strips_internal_keys_before_the_sdk_call(mocker):
    """Xinference: ``stream_chat`` is a second wholesale-forwarding path with
    its own strip call. ``vision_chat`` needs no separate check here -- it
    delegates straight to ``chat`` (see xinference.py) rather than building
    a request of its own.
    """

    class _EmptyStream:
        def __aiter__(self) -> "_EmptyStream":
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    class _StreamHandle:
        async def chat(self, **kwargs: Any) -> _EmptyStream:
            self.received_messages = kwargs["messages"]
            return _EmptyStream()

    llm = XinferenceLLM(model_name="qwen3.8")
    handle = _StreamHandle()
    llm._client = mocker.MagicMock()
    llm._model_handle = handle

    async for _chunk in llm.stream_chat(_MARKED_HISTORY):
        pass

    _assert_no_internal_keys(handle.received_messages)


# --- Discovery guard: every BaseLLM subclass must be accounted for ---------
#
# The tests above cover four clients by name. That list goes stale silently
# the moment a new BaseLLM subclass is added and nobody remembers to give it
# the same coverage. The two registries and the test below turn that into a
# loud failure: every concrete BaseLLM subclass found anywhere under
# ``xagent.core.model.chat.basic`` must appear in one of them, or the
# discovery test fails and names the class that is missing.

# Classes with a dedicated leak-guard test, here or elsewhere:
#   - ClaudeLLM / GeminiLLM / ZhipuLLM / XinferenceLLM: the four tests above.
#   - OpenAILLM: test_openai.py::test_internal_xagent_message_keys_are_stripped
#     exercises _build_request_messages -> _strip_internal_message_keys
#     directly.
#   - DeepSeekLLM / OpenRouterLLM / AzureOpenAILLM / DashScopeLLM: none of
#     these override _prepare_messages_for_request's sanitization step or
#     _strip_internal_message_keys, so they run OpenAILLM's own
#     _build_request_messages unchanged and are covered by the same test.
_STRIP_GUARD_COVERED: frozenset[type] = frozenset(
    {
        ClaudeLLM,
        GeminiLLM,
        ZhipuLLM,
        XinferenceLLM,
        OpenAILLM,
        DeepSeekLLM,
        OpenRouterLLM,
        AzureOpenAILLM,
        DashScopeLLM,
    }
)

# Classes deliberately excluded, with why leaking is structurally impossible
# for them rather than merely untested:
_STRIP_GUARD_EXEMPT: dict[type, str] = {
    OpenAICompatibleLLM: (
        "intermediate base that implements the stripping call itself; never "
        "constructed directly as a provider, and its behavior is covered "
        "transitively by the five registered concrete subclasses above -- "
        "removing the stripping call turns their tests red."
    ),
    RouterLLM: (
        "the virtual auto-router: chat/vision_chat/stream_chat resolve a "
        "downstream BaseLLM and forward the same message list object to it "
        "unchanged (see router.py); it reads message content for routing "
        "but never constructs any provider-bound message dict itself, so "
        "the downstream client's own guard is what runs."
    ),
    _ResolvedRouterLLM: (
        "a thin per-call wrapper around one resolved downstream client; "
        "same reasoning as RouterLLM above."
    ),
}


def _all_basellm_subclasses() -> set[type]:
    """Recursively discover every production BaseLLM subclass registered so far.

    Imports every module under ``xagent.core.model.chat.basic`` first, so a
    subclass defined in a module nothing else in this test file happens to
    import still gets registered on ``BaseLLM.__subclasses__()`` before the
    walk below runs.

    ``BaseLLM.__subclasses__()`` is a process-wide registry: it also picks up
    test-double subclasses defined inside other test modules once pytest has
    collected them (e.g. a scripted fake LLM used to exercise the router).
    Those are excluded by module, not by name, since this file has no way to
    tell "a fake used only in one test" from "a real provider" other than
    where the class is defined: only classes whose ``__module__`` lives
    under ``xagent.core.model.chat.basic`` itself count as production
    clients this guard is responsible for.
    """
    for _, module_name, _ in pkgutil.iter_modules(basic_pkg.__path__):
        importlib.import_module(f"{basic_pkg.__name__}.{module_name}")

    discovered: set[type] = set()

    def _walk(cls: type) -> None:
        for subclass in cls.__subclasses__():
            if subclass in discovered:
                continue
            discovered.add(subclass)
            _walk(subclass)

    _walk(BaseLLM)
    _production_prefix = f"{basic_pkg.__name__}."
    return {
        subclass
        for subclass in discovered
        if subclass.__module__.startswith(_production_prefix)
    }


def test_every_basellm_subclass_is_accounted_for_by_the_strip_guard():
    """A newly added BaseLLM subclass with no leak-guard coverage must fail
    here, not go unnoticed. This test does not re-verify the four dedicated
    tests' outcome (they already do); it only verifies that every subclass
    that exists is *known* to one of the two registries above.
    """
    discovered = _all_basellm_subclasses()
    unaccounted = discovered - _STRIP_GUARD_COVERED - set(_STRIP_GUARD_EXEMPT)

    assert not unaccounted, (
        "New BaseLLM subclass(es) with no _xagent_ leak-guard coverage: "
        f"{sorted(cls.__qualname__ for cls in unaccounted)}. Add a "
        "dedicated test asserting no '_xagent_'-prefixed message key "
        "reaches this client's wire request (see the ClaudeLLM/ZhipuLLM "
        "tests above for the two known-good shapes) and add the class to "
        "_STRIP_GUARD_COVERED, or if it never reads/copies a message dict "
        "itself, add it to _STRIP_GUARD_EXEMPT with a comment explaining "
        "why."
    )
