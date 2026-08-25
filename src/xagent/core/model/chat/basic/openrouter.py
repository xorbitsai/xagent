import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import openai

from .....config import get_openrouter_official_providers_only
from ..error import retry_on
from ..exceptions import LLMRetryableError, LLMToolProtocolError
from ..timeout_config import TimeoutConfig
from ..tool_protocol import TOOL_PROTOCOL_ERROR_KEY, get_tool_protocol_error
from ..types import StreamChunk
from .deepseek_tool_protocol import (
    adapt_deepseek_stream,
    normalize_deepseek_response,
)
from .openai import OpenAILLM

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEEPSEEK_FUNCTION_PREFIX_ERROR = "function call should not be used with prefix"

# OpenAILLM.chat/vision_chat normally convert every openai.BadRequestError
# into a RuntimeError before returning. Their response_format pop-and-retry
# path is an exception, though: it re-issues the request from inside its own
# ``except openai.BadRequestError`` block, and if that retried call also fails
# with a BadRequestError, nothing wraps the second failure -- it escapes as a
# bare SDK exception. (stream_chat is not affected: its resend sits in a
# nested try, so the outer handler still wraps a second failure; the streaming
# loop catches the tuple for symmetry and defense.) openai.BadRequestError's
# MRO does not include RuntimeError, so the compat retry loops below must
# catch both explicitly to keep covering that case. The historical
# implementation caught bare ``Exception`` here; this tuple is the precise,
# intentionally narrowed replacement.
_COMPAT_RETRYABLE_ERRORS = (RuntimeError, openai.BadRequestError)

# Pinning to these provider slugs via `only` + `allow_fallbacks: False` routes
# every request for the author to one of the listed official endpoints. Before
# adding an author here, confirm each of its official endpoints actually supports
# every parameter this client can send (tools, tool_choice, response_format,
# structured outputs, temperature) — an unsupported parameter is now silently
# ignored by the endpoint rather than rejected, so a wrong entry degrades
# request semantics instead of failing loudly.
_OPENROUTER_OFFICIAL_PROVIDERS_BY_AUTHOR: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "deepseek": ("deepseek",),
    "google": ("google-ai-studio", "google-vertex"),
    "minimax": ("minimax",),
    "openai": ("openai",),
    "z-ai": ("z-ai",),
}


def _openrouter_model_author(model_name: str) -> str:
    model_slug = model_name.strip().split(":", 1)[0]
    parts = [part for part in model_slug.split("/") if part]
    if len(parts) >= 3 and parts[0].lower() == "openrouter":
        return parts[1].lower()
    if len(parts) >= 2:
        return parts[0].lower()
    return ""


def _strip_assistant_tool_call_prefixes(
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], bool]:
    """Remove assistant prefixes that DeepSeek cannot combine with tools."""
    sanitized: List[Dict[str, Any]] = []
    changed = False
    for message in messages:
        sanitized_message = dict(message)
        if (
            sanitized_message.get("role") == "assistant"
            and sanitized_message.get("tool_calls")
            and str(sanitized_message.get("content") or "").strip()
        ):
            sanitized_message["content"] = ""
            changed = True
        sanitized.append(sanitized_message)

    # A non-blocking ``send_message`` control call records its result and may
    # leave a standalone assistant progress message at the end of older
    # checkpoints. OpenRouter treats that trailing assistant turn as prefix
    # completion, which DeepSeek rejects when tools are also present. The tool
    # result already contains the same progress text, so dropping only trailing
    # assistant-only turns preserves the tool chain and avoids replaying stale
    # progress as a generation prefix.
    has_completed_tool_chain = any(
        message.get("role") == "assistant" and message.get("tool_calls")
        for message in sanitized
    ) and any(message.get("role") == "tool" for message in sanitized)
    if has_completed_tool_chain:
        while (
            sanitized
            and sanitized[-1].get("role") == "assistant"
            and not sanitized[-1].get("tool_calls")
        ):
            sanitized.pop()
            changed = True
    return sanitized, changed


def _force_single_required_deepseek_tool(
    tools: Optional[List[Any]],
    tool_choice: Optional[str | Dict[str, Any]],
) -> Optional[str | Dict[str, Any]]:
    """Turn DeepSeek's ambiguous single-tool requirement into a named choice."""
    if tool_choice != "required" or not tools or len(tools) != 1:
        return tool_choice
    tool = tools[0]
    if not isinstance(tool, dict):
        return tool_choice
    function = tool.get("function")
    if not isinstance(function, dict):
        return tool_choice
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return tool_choice
    return {
        "type": "function",
        "function": {"name": name},
    }


def _deepseek_tool_protocol_retry_error(response: Any) -> LLMRetryableError | None:
    error = get_tool_protocol_error(response)
    if error is None:
        return None
    code = str(error.get("code") or "invalid_tool_protocol")
    # Replaying the same narrowed schema cannot make the requested tool
    # available. Surface this response to the agent pattern so it can restore
    # the appropriate tool set and re-decide with explicit feedback.
    if code == "unavailable_tool_call":
        return None
    message = str(error.get("message") or "DeepSeek returned an invalid tool call.")
    return LLMToolProtocolError(
        provider="deepseek",
        code=code,
        message=message,
        details=error.get("details")
        if isinstance(error.get("details"), dict)
        else None,
    )


# Normalized intents translated into OpenRouter's reasoning/thinking request body.
_DISABLE_DOWNSTREAM_THINKING = {"type": "disabled", "enable": False}
_ENABLE_DOWNSTREAM_THINKING = {"type": "enabled", "enable": True}

_ACTION_ENABLE_THINKING = "enable_thinking"
_ACTION_DISABLE_THINKING = "disable_thinking"
_ACTION_RELAX_TOOL_CHOICE = "relax_tool_choice"


def _should_retry_with_thinking(
    exc: Exception,
    *,
    thinking: Optional[Dict[str, Any]],
) -> bool:
    # This is the primary stop condition after a retry swaps in enabled thinking;
    # retry-action tracking remains defense in depth for the shared retry loop.
    if isinstance(thinking, dict) and (
        thinking.get("type") == "enabled" or thinking.get("enable") is True
    ):
        return False

    # OpenRouter currently exposes this provider constraint only through an
    # untyped 400 response. Retry the same selected model once with reasoning
    # enabled instead of repeating the rejected payload or rerouting.
    # Replace string matching with typed provider errors when available.
    exc_msg = str(exc).lower()
    return "reasoning is mandatory" in exc_msg and "cannot be disabled" in exc_msg


def _should_retry_without_thinking(
    exc: Exception,
    *,
    thinking: Optional[Dict[str, Any]],
    tool_choice: Optional[str | Dict[str, Any]],
) -> bool:
    # Deliberate OpenRouter/DeepSeek compatibility bridge: the provider returns
    # an OpenAI-compatible 400 without a typed error for this thinking/tool_choice
    # conflict. Replace this with provider-owned typed exceptions once the
    # follow-up tracking issue lands.
    exc_msg = str(exc).lower()
    return (
        tool_choice is not None and "thinking" in exc_msg and "tool_choice" in exc_msg
    )


def _should_retry_with_relaxed_tool_choice(
    exc: Exception,
    *,
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[str | Dict[str, Any]],
) -> bool:
    if not tools or tool_choice in (None, "auto", "none"):
        return False

    # Deliberate OpenRouter compatibility bridge: official provider routing can
    # reject strict tool_choice values before selecting an endpoint. This
    # degrades forced tool use to "auto" instead of failing the whole agent run.
    # Replace string matching with typed provider errors when available.
    exc_msg = str(exc).lower()
    return "no endpoints found" in exc_msg and "tool_choice" in exc_msg


def _next_compat_adjustment(
    exc: Exception,
    *,
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[str | Dict[str, Any]],
    thinking: Optional[Dict[str, Any]],
    attempted: set[str],
    render: Callable[[Optional[Dict[str, Any]]], Dict[str, Any]],
) -> tuple[Optional[str | Dict[str, Any]], Optional[Dict[str, Any]], str, str] | None:
    """Pick the next OpenRouter provider-compatibility adjustment for a failure.

    Candidates are checked strictest-first (relax tool_choice, then disable
    thinking, then enable thinking) and tried in that written order: the
    first candidate whose predicate matches and would actually change the
    rendered request wins. Order is the mechanism here, not a side detail:
    reordering the ``candidates`` tuple changes which adjustment fires first
    and is not a safe refactor. A rule that matches but would leave the
    actually rendered request unchanged compared to the current
    ``(tool_choice, render(thinking))`` state is a no-op and is skipped
    without spending its retry budget, so a rule that cannot fix anything
    does not block a later rule that can. A rule whose action was already
    attempted stops the search outright, even if a later, not-yet-attempted
    rule would also match: one exhausted rule ends the whole search rather
    than yielding to the next candidate.
    """
    current_state = (tool_choice, render(thinking))
    candidates: tuple[
        tuple[Optional[str | Dict[str, Any]], Optional[Dict[str, Any]], str, str]
        | None,
        ...,
    ] = (
        (
            (
                "auto",
                thinking,
                "selected OpenRouter endpoint rejected tool_choice; retrying with tool_choice=auto",
                _ACTION_RELAX_TOOL_CHOICE,
            )
            if _should_retry_with_relaxed_tool_choice(
                exc, tools=tools, tool_choice=tool_choice
            )
            else None
        ),
        (
            (
                tool_choice,
                _DISABLE_DOWNSTREAM_THINKING,
                "selected model rejected thinking with tool_choice; retrying without thinking",
                _ACTION_DISABLE_THINKING,
            )
            if _should_retry_without_thinking(
                exc, thinking=thinking, tool_choice=tool_choice
            )
            else None
        ),
        (
            (
                tool_choice,
                _ENABLE_DOWNSTREAM_THINKING,
                "selected model requires reasoning; retrying with thinking enabled",
                _ACTION_ENABLE_THINKING,
            )
            if _should_retry_with_thinking(exc, thinking=thinking)
            else None
        ),
    )

    for candidate in candidates:
        if candidate is None:
            continue
        next_tool_choice, next_thinking, _log_message, action_key = candidate
        if (next_tool_choice, render(next_thinking)) == current_state:
            continue
        if action_key in attempted:
            return None
        return candidate

    return None


class OpenRouterLLM(OpenAILLM):
    """OpenRouter client using the OpenAI SDK with OpenRouter-specific options."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url or OPENROUTER_BASE_URL,
            api_key=api_key,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            timeout=timeout,
            abilities=abilities,
            timeout_config=timeout_config,
        )

    @property
    def _uses_deepseek_tool_protocol(self) -> bool:
        return _openrouter_model_author(self._model_name) == "deepseek"

    def _is_official_openrouter_client(self) -> bool:
        return self.base_url.rstrip("/") == OPENROUTER_BASE_URL

    def _deepseek_function_prefix_retry_messages(
        self,
        exc: Exception,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]] | None:
        if _openrouter_model_author(self._model_name) != "deepseek":
            return None
        if _DEEPSEEK_FUNCTION_PREFIX_ERROR not in str(exc).lower():
            return None
        sanitized, changed = _strip_assistant_tool_call_prefixes(messages)
        return sanitized if changed else None

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Send a chat completion through OpenRouter.

        Applies DeepSeek-specific request shaping and DeepSeek tool-protocol
        handling, plus OpenRouter provider-compatibility retries: a retry may
        relax a strict ``tool_choice`` down to ``"auto"``, or flip
        ``thinking`` between enabled and disabled, each adjustment at most
        once per call (see ``_next_compat_adjustment``).
        """
        if self._uses_deepseek_tool_protocol:
            tool_choice = _force_single_required_deepseek_tool(tools, tool_choice)
        response = await self._run_chat_with_compat_retry(
            self._chat_with_prefix_retry,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            kwargs=kwargs,
            sanitize_messages=True,
        )

        if not self._uses_deepseek_tool_protocol:
            return response
        response = normalize_deepseek_response(response, tools=tools)
        retry_error = _deepseek_tool_protocol_retry_error(response)
        if retry_error is not None:
            raise retry_error
        return response

    async def vision_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Send a vision-capable chat completion through OpenRouter.

        Applies the same OpenRouter provider-compatibility retries as
        ``chat``: a retry may relax a strict ``tool_choice`` down to
        ``"auto"``, or flip ``thinking`` between enabled and disabled, each
        adjustment at most once per call. Unlike ``chat``, this skips
        DeepSeek prefix retries and DeepSeek tool-protocol handling entirely
        (see the comment below).
        """
        # Vision requests carry no DeepSeek prefix-retry need, so the inner
        # call goes straight to the OpenAI implementation. This also skips
        # chat()'s DeepSeek tool-protocol handling (_force_single_required_
        # deepseek_tool, normalize_deepseek_response, the protocol-error
        # retry), which is a deliberate omission rather than an oversight:
        # those branches only take effect when tools are passed, and no
        # current caller passes tools into vision_chat.
        return await self._run_chat_with_compat_retry(
            super().vision_chat,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            kwargs=kwargs,
        )

    async def _chat_with_prefix_retry(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        sanitized_out: Optional[List[List[Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Send ``messages``, retrying once with DeepSeek's prefix stripped.

        ``sanitized_out``, when given, receives the sanitized message list at
        the moment a prefix retry fires, so a caller looping over repeated
        provider-compat errors (see ``_run_chat_with_compat_retry``) can reuse
        the already-sanitized messages instead of re-triggering this same
        prefix rejection on every iteration.
        """
        try:
            response = await super().chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=thinking,
                output_config=output_config,
                **kwargs,
            )
        except RuntimeError as exc:
            sanitized_messages = self._deepseek_function_prefix_retry_messages(
                exc, messages
            )
            if sanitized_messages is None:
                raise

            if sanitized_out is not None:
                sanitized_out.append(sanitized_messages)

            logger.info(
                "OpenRouter DeepSeek rejected function-call history with an "
                "assistant prefix; retrying once without tool-call prefixes"
            )
            response = await super().chat(
                messages=sanitized_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=thinking,
                output_config=output_config,
                **kwargs,
            )

        return response

    async def _run_chat_with_compat_retry(
        self,
        call: Callable[..., Any],
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str | Dict[str, Any]],
        response_format: Optional[Dict[str, Any]],
        thinking: Optional[Dict[str, Any]],
        output_config: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
        sanitize_messages: bool = False,
    ) -> Any:
        """Retry ``call`` once per matching OpenRouter provider-compat rule.

        Shared by ``chat`` (wrapping ``_chat_with_prefix_retry``, with
        ``sanitize_messages=True``) and ``vision_chat`` (wrapping the
        inherited ``OpenAILLM.vision_chat`` directly, which has no
        ``sanitized_out`` parameter and must never receive one). A retryable
        error raised by the inner call (e.g. ``LLMEmptyContentError`` after an
        empty-content response, or a ``RuntimeError`` wrapping a rate-limit or
        5xx error that ``retry_on`` recognizes via ``__cause__``) is left for
        the shared LLM retry wrapper and is never treated as a compat
        adjustment opportunity. A single ``call`` invocation may itself issue
        one additional upstream request through the response_format
        pop-and-retry path shared by ``OpenAILLM.chat`` and ``vision_chat``,
        but the two behave differently on a successful resend: ``chat``'s
        branch returns the processed result of the resent call, or lets that
        second call's failure propagate uncaught; ``vision_chat``'s
        same-named branch does neither on a successful resend -- it falls
        out of the ``except`` block with no ``return`` statement, so the
        call implicitly yields ``None`` instead of the resent response
        (tracked by #1650). No caller in this repository currently passes
        ``response_format`` into ``chat``; the one caller that does supply
        it in this repository (``vision_tool.py``) calls ``vision_chat``.

        Known limitation: ``OpenAILLM.chat``'s structured-output degrade path
        can rewrite ``thinking`` internally (disabling it after a non-JSON
        response) without reporting the change back here, so
        ``current_thinking`` does not necessarily reflect what was actually
        sent on that path.

        Upstream request bounds for one logical call (one ``chat``/
        ``vision_chat`` invocation, i.e. one attempt as seen by the
        per-model ``RetryWrapper`` around this class): the compat-retry loop
        below tries the initial request plus at most one attempt per
        distinct action (``relax_tool_choice``, ``disable_thinking``,
        ``enable_thinking``), so at most 4 requests come from this loop
        itself. Without ``response_format``, ``chat``'s ``call`` adds at
        most 1 more request in total (not per loop iteration) from
        ``_chat_with_prefix_retry``'s DeepSeek function-call-prefix resend,
        since the sanitized messages carry forward into later iterations
        instead of re-triggering the same prefix rejection -- at most 5
        upstream requests per logical call. With ``response_format``
        supplied (``vision_chat`` today), the inner pop-and-retry can instead
        fire independently on every one of the up to 4 loop iterations,
        since each iteration's ``tool_choice``/``thinking`` differs -- at
        most 8 upstream requests per logical call. Under
        ``ModelConfig.max_retries``'s default outer budget of 10 attempts, a
        run whose errors keep alternating between a compat-fixable shape and
        an outer-retryable one can reach at most 50 upstream requests without
        ``response_format`` or 80 with it.
        """
        current_tool_choice = tool_choice
        current_thinking = thinking
        current_messages = messages
        attempted: set[str] = set()

        # ``render`` only models extra_body. Thinking also reaches the request
        # through ``_build_request_messages(messages, thinking=...)``, which
        # ``render`` never sees; that second path is a no-op today only because
        # this class's MRO resolves to ``OpenAILLM._prepare_messages_for_request``
        # (ignores ``thinking``), not ``DeepSeekLLM``'s rewrite. If a class on
        # this MRO ever starts consuming ``thinking`` there, this no-op
        # comparison must model the message body too, not just extra_body.
        def render(candidate_thinking: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            return self._prepare_provider_reasoning_extra_body(
                extra_body=self._prepare_extra_body(
                    dict(kwargs.get("extra_body") or {})
                ),
                thinking=candidate_thinking,
                tools=tools,
                response_format=response_format,
                output_config=output_config,
                is_streaming=False,
            )

        while True:
            sanitized_out: List[List[Dict[str, Any]]] = []
            call_kwargs: Dict[str, Any] = dict(
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=current_tool_choice,
                response_format=response_format,
                thinking=current_thinking,
                output_config=output_config,
                **kwargs,
            )
            if sanitize_messages:
                call_kwargs["sanitized_out"] = sanitized_out
            try:
                return await call(current_messages, **call_kwargs)
            except LLMRetryableError:
                raise
            except _COMPAT_RETRYABLE_ERRORS as exc:
                if retry_on(exc):
                    raise

                adjustment = _next_compat_adjustment(
                    exc,
                    tools=tools,
                    tool_choice=current_tool_choice,
                    thinking=current_thinking,
                    attempted=attempted,
                    render=render,
                )
                if adjustment is None:
                    raise

                current_tool_choice, current_thinking, log_message, action_key = (
                    adjustment
                )
                attempted.add(action_key)
                if sanitized_out:
                    current_messages = sanitized_out[-1]
                logger.info(log_message)

    async def _stream_chat_with_prefix_retry(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        sanitized_out: Optional[List[List[Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming counterpart of ``_chat_with_prefix_retry``.

        ``sanitized_out`` carries the same contract as the non-streaming
        version: when the prefix retry fires, the sanitized messages are
        appended to it so a caller looping over compat errors can reuse them.
        """
        has_yielded = False
        try:
            async for chunk in super().stream_chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=thinking,
                output_config=output_config,
                **kwargs,
            ):
                has_yielded = True
                yield chunk
            return
        except RuntimeError as exc:
            sanitized_messages = self._deepseek_function_prefix_retry_messages(
                exc, messages
            )
            if has_yielded or sanitized_messages is None:
                raise

        if sanitized_out is not None:
            sanitized_out.append(sanitized_messages)

        logger.info(
            "OpenRouter DeepSeek rejected streaming function-call history with an "
            "assistant prefix; retrying once without tool-call prefixes"
        )
        async for chunk in super().stream_chat(
            messages=sanitized_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        ):
            yield chunk

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion through OpenRouter.

        Applies the same OpenRouter provider-compatibility retries as
        ``chat``: a retry may relax a strict ``tool_choice`` down to
        ``"auto"``, or flip ``thinking`` between enabled and disabled, each
        adjustment at most once per call. Once the first chunk has been
        yielded to the caller, no further compat retry happens: a later
        error on that same stream is raised as-is, since replaying it would
        duplicate output already sent.
        """
        async for chunk in self._run_stream_chat_with_compat_retry(
            self._stream_chat_inner,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            kwargs=kwargs,
        ):
            yield chunk

    async def _run_stream_chat_with_compat_retry(
        self,
        call: Callable[..., AsyncIterator[StreamChunk]],
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str | Dict[str, Any]],
        response_format: Optional[Dict[str, Any]],
        thinking: Optional[Dict[str, Any]],
        output_config: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        """Streaming counterpart of ``_run_chat_with_compat_retry``.

        Only retries while nothing has reached the caller yet: once a chunk
        has actually been yielded, replaying the request would duplicate
        output, so any later error is raised as-is. Note that for the
        DeepSeek dict-tool_choice path ``call`` (``_stream_chat_inner``)
        buffers chunks internally until the tool protocol is validated, so an
        inner chunk being produced is not the same thing as one having been
        yielded from here. ``call`` is always ``_stream_chat_inner``, which
        forwards its ``**kwargs`` straight through to
        ``_stream_chat_with_prefix_retry``, so the ``sanitized_out`` list
        built here reaches that function's out-param without either function
        needing to know about it explicitly.
        """
        has_yielded = False
        current_tool_choice = tool_choice
        current_thinking = thinking
        current_messages = messages
        attempted: set[str] = set()

        # ``render`` only models extra_body, not the thinking that also flows
        # into ``_build_request_messages``'s messages -- see the ``render``
        # closure in ``_run_chat_with_compat_retry`` for why that is safe today.
        def render(candidate_thinking: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            return self._prepare_provider_reasoning_extra_body(
                extra_body=self._prepare_extra_body(
                    dict(kwargs.get("extra_body") or {})
                ),
                thinking=candidate_thinking,
                tools=tools,
                response_format=response_format,
                output_config=output_config,
                is_streaming=True,
            )

        while True:
            sanitized_out: List[List[Dict[str, Any]]] = []
            try:
                async for chunk in call(
                    current_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=current_tool_choice,
                    response_format=response_format,
                    thinking=current_thinking,
                    output_config=output_config,
                    sanitized_out=sanitized_out,
                    **kwargs,
                ):
                    has_yielded = True
                    yield chunk
                return
            except LLMRetryableError:
                raise
            except _COMPAT_RETRYABLE_ERRORS as exc:
                if has_yielded:
                    raise
                if retry_on(exc):
                    raise

                adjustment = _next_compat_adjustment(
                    exc,
                    tools=tools,
                    tool_choice=current_tool_choice,
                    thinking=current_thinking,
                    attempted=attempted,
                    render=render,
                )
                if adjustment is None:
                    raise

                current_tool_choice, current_thinking, log_message, action_key = (
                    adjustment
                )
                attempted.add(action_key)
                if sanitized_out:
                    current_messages = sanitized_out[-1]
                logger.info(log_message)

    async def _stream_chat_inner(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        if self._uses_deepseek_tool_protocol:
            tool_choice = _force_single_required_deepseek_tool(tools, tool_choice)
        stream = self._stream_chat_with_prefix_retry(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )
        if not self._uses_deepseek_tool_protocol:
            async for chunk in stream:
                yield chunk
            return
        adapted_stream = adapt_deepseek_stream(stream, tools=tools)

        # A named tool choice cannot validly finish as assistant text. Buffer
        # this narrow stream until DeepSeek's tool protocol has been validated,
        # so a malformed serialized call raises before any chunk escapes and the
        # shared LLM retry wrapper can safely replay the request.
        if isinstance(tool_choice, dict):
            buffered_chunks: list[StreamChunk] = []
            async for chunk in adapted_stream:
                if chunk.is_protocol_error():
                    retry_error = _deepseek_tool_protocol_retry_error(
                        {TOOL_PROTOCOL_ERROR_KEY: chunk.protocol_error}
                    )
                    if retry_error is not None:
                        raise retry_error
                buffered_chunks.append(chunk)
            for chunk in buffered_chunks:
                yield chunk
            return

        async for chunk in adapted_stream:
            if chunk.is_protocol_error():
                retry_error = _deepseek_tool_protocol_retry_error(
                    {TOOL_PROTOCOL_ERROR_KEY: chunk.protocol_error}
                )
                if retry_error is not None:
                    raise retry_error
            yield chunk

    def _prepare_extra_body(self, extra_body: Dict[str, Any]) -> Dict[str, Any]:
        if (
            not get_openrouter_official_providers_only()
            or not self._is_official_openrouter_client()
            or "provider" in extra_body
        ):
            return extra_body

        author = _openrouter_model_author(self._model_name)
        official_providers = _OPENROUTER_OFFICIAL_PROVIDERS_BY_AUTHOR.get(author)
        if not official_providers:
            return extra_body

        return {
            **extra_body,
            "provider": {
                "only": list(official_providers),
                "allow_fallbacks": False,
            },
        }

    def _prepare_provider_reasoning_extra_body(
        self,
        *,
        extra_body: Dict[str, Any],
        thinking: Optional[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
        output_config: Optional[Dict[str, Any]],
        is_streaming: bool,
    ) -> Dict[str, Any]:
        _ = tools, output_config
        updated_extra_body = dict(extra_body)

        if thinking is not None:
            should_enable = thinking.get("type") == "enabled" or thinking.get(
                "enable", False
            )
            should_disable = not should_enable and (
                thinking.get("type") == "disabled" or not thinking.get("enable", False)
            )
        elif is_streaming and response_format:
            should_disable = (
                self.supports_thinking_mode or self._uses_deepseek_tool_protocol
            )
            should_enable = False
        else:
            # DeepSeek-served endpoints can default to thinking mode, and once a
            # response carries reasoning_content they require it to be replayed
            # verbatim on the next request of a tool-call chain — which this
            # client does not do (#1537). Keep thinking off unless the caller
            # asks for it, matching DeepSeekLLM's default. This deliberately
            # ignores supports_thinking_mode: the blocker is the missing
            # replay, not model capability, so a declared thinking_mode
            # ability must not re-enable the failing default before #1537.
            should_disable = self._uses_deepseek_tool_protocol
            should_enable = False

        if should_disable:
            updated_extra_body["reasoning"] = {"enabled": False}
            updated_extra_body["thinking"] = {"type": "disabled"}
        elif should_enable:
            updated_extra_body["reasoning"] = {"enabled": True}
            updated_extra_body["thinking"] = {"type": "enabled"}

        updated_extra_body.pop("enable_thinking", None)
        return updated_extra_body
