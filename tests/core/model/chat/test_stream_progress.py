"""Unit tests for the streaming no-progress guard (#2785).

The guard is pure: it is fed per-delta facts plus the accumulated tool-call
state that ``OpenAICompatibleLLM._parse_stream_chunk`` maintains, and returns
an abort verdict or ``None``. Every "must not abort" case here protects a
legitimate stream shape that shares the same code path (DeepSeek-style long
reasoning, parallel tool calls, a still-open JSON object, keepalive chunks).
"""

from __future__ import annotations

import pytest

from xagent.core.model.chat.stream_progress import (
    NO_PAYLOAD_STREAM_FALLBACK,
    NO_PROGRESS_FINISH_REASON,
    StreamAbort,
    StreamProgressConfig,
    StreamProgressGuard,
)


def _config(**overrides: object) -> StreamProgressConfig:
    base = {
        "empty_delta_limit": 5,
        "degenerate_window": 32,
        "degenerate_max_period": 8,
        "no_payload_timeout": None,
    }
    base.update(overrides)
    return StreamProgressConfig(**base)  # type: ignore[arg-type]


def _tool_calls(*calls: tuple[str, str]) -> dict[str, dict]:
    """``(call_id, arguments)`` pairs in the shape ``_parse_stream_chunk`` keeps."""
    return {
        call_id: {
            "index": position,
            "id": call_id,
            "type": "function",
            "function": {"name": "search", "arguments": arguments},
        }
        for position, (call_id, arguments) in enumerate(calls)
    }


class TestEmptyDeltas:
    def test_consecutive_empty_deltas_reach_the_limit(self) -> None:
        guard = StreamProgressGuard(_config(empty_delta_limit=3))
        assert guard.observe() is None
        assert guard.observe() is None
        verdict = guard.observe()
        assert isinstance(verdict, StreamAbort)
        assert verdict.reason == "empty_deltas"

    def test_any_progress_resets_the_counter(self) -> None:
        """Keepalive chunks interleaved with real deltas never accumulate."""
        guard = StreamProgressGuard(_config(empty_delta_limit=3))
        assert guard.observe() is None
        assert guard.observe() is None
        assert guard.observe(has_content=True) is None
        assert guard.observe() is None
        assert guard.observe() is None
        assert guard.observe(reasoning_texts=("thinking",)) is None
        assert guard.observe() is None
        assert guard.observe() is None
        assert guard.observe(has_finish_reason=True) is None

    def test_empty_reasoning_text_is_not_liveness(self) -> None:
        """A ``reasoning_content: ""`` delta carries nothing; it counts as empty."""
        guard = StreamProgressGuard(_config(empty_delta_limit=2))
        assert guard.observe(reasoning_texts=("",)) is None
        verdict = guard.observe(reasoning_texts=("",))
        assert verdict is not None and verdict.reason == "empty_deltas"

    def test_opaque_reasoning_value_is_liveness(self) -> None:
        """A non-text ``reasoning*`` payload (e.g. a details list) is live."""
        guard = StreamProgressGuard(_config(empty_delta_limit=2))
        for _ in range(10):
            assert guard.observe(reasoning_opaque=True) is None

    def test_usage_counts_as_progress(self) -> None:
        guard = StreamProgressGuard(_config(empty_delta_limit=2))
        assert guard.observe() is None
        assert guard.observe(has_usage=True) is None
        assert guard.observe() is None

    def test_zero_limit_disables_the_check(self) -> None:
        guard = StreamProgressGuard(_config(empty_delta_limit=0))
        for _ in range(500):
            assert guard.observe() is None


class TestDegenerateReasoning:
    def test_whitespace_only_tail_aborts(self) -> None:
        guard = StreamProgressGuard(_config(degenerate_window=16))
        assert guard.observe(reasoning_texts=("Let me think about this.",)) is None
        for _ in range(3):
            assert guard.observe(reasoning_texts=("     ",)) is None
        verdict = guard.observe(reasoning_texts=(" \t\n ",))
        assert verdict is not None
        assert verdict.reason == "reasoning_whitespace"

    def test_whitespace_below_the_window_does_not_abort(self) -> None:
        guard = StreamProgressGuard(_config(degenerate_window=16))
        assert guard.observe(reasoning_texts=("Let me think.",)) is None
        assert guard.observe(reasoning_texts=(" " * 15,)) is None

    def test_periodic_tail_aborts(self) -> None:
        guard = StreamProgressGuard(
            _config(degenerate_window=32, degenerate_max_period=8)
        )
        assert (
            guard.observe(reasoning_texts=("Some genuine reasoning first. ",)) is None
        )
        verdict = None
        for _ in range(10):
            verdict = guard.observe(reasoning_texts=("verify ",))
            if verdict is not None:
                break
        assert verdict is not None
        assert verdict.reason == "reasoning_repetition"
        assert "period=7" in verdict.detail

    def test_period_longer_than_max_period_is_not_repetition(self) -> None:
        """Only short-period loops are degenerate; a long paragraph repeated
        once is not what the check is for and must not trip it."""
        guard = StreamProgressGuard(
            _config(degenerate_window=32, degenerate_max_period=8)
        )
        paragraph = "abcdefghijklmnop"  # period 16 > max_period 8
        for _ in range(6):
            assert guard.observe(reasoning_texts=(paragraph,)) is None

    def test_single_character_run_is_period_one(self) -> None:
        guard = StreamProgressGuard(
            _config(degenerate_window=16, degenerate_max_period=4)
        )
        assert guard.observe(reasoning_texts=("ok ",)) is None
        verdict = guard.observe(reasoning_texts=("=" * 16,))
        assert verdict is not None
        assert verdict.reason == "reasoning_repetition"
        assert "period=1" in verdict.detail

    def test_legitimate_reasoning_never_aborts(self) -> None:
        """Long, non-repetitive reasoning with no content is the DeepSeek /
        Qwen-thinking shape and must stay live indefinitely."""
        guard = StreamProgressGuard(
            _config(degenerate_window=32, degenerate_max_period=8)
        )
        words = [
            "first",
            "consider",
            "the",
            "user",
            "asked",
            "for",
            "a",
            "shift",
            "on",
            "Monday",
            "so",
            "we",
            "need",
            "the",
            "timezone",
            "and",
            "then",
            "call",
            "the",
            "tool",
            "with",
            "those",
            "values",
            ".",
        ]
        for step in range(400):
            text = words[step % len(words)] + (" " if step % 7 else "\n")
            assert guard.observe(reasoning_texts=(text,)) is None

    def test_short_phrase_repeated_a_few_times_does_not_abort(self) -> None:
        """ "let me verify" twice in a row is normal; the window must fill
        entirely with the repetition before the check fires."""
        guard = StreamProgressGuard(
            _config(degenerate_window=64, degenerate_max_period=16)
        )
        assert guard.observe(reasoning_texts=("The answer is 42, but ",)) is None
        assert (
            guard.observe(reasoning_texts=("let me verify. let me verify. ",)) is None
        )
        assert guard.observe(reasoning_texts=("Yes, 42 is right.",)) is None

    def test_production_defaults_keep_varied_reasoning_live(self) -> None:
        """At the shipped thresholds (W=256, P=64) a long chain of distinct
        sentences never trips either reasoning rule."""
        guard = StreamProgressGuard(StreamProgressConfig(empty_delta_limit=0))
        for step in range(2000):
            sentence = (
                f"Step {step}: the constraint {step * 7 % 13} interacts with "
                f"case {step * 11 % 17}, so re-check {step % 5}. "
            )
            assert guard.observe(reasoning_texts=(sentence,)) is None

    def test_production_defaults_catch_a_short_phrase_loop(self) -> None:
        """At the shipped thresholds a 7-char phrase looping to the cap is
        caught once it alone fills the window; a divisor of 64 is not required."""
        guard = StreamProgressGuard(StreamProgressConfig(empty_delta_limit=0))
        assert guard.observe(reasoning_texts=("Let me check the schedule. ",)) is None
        verdict = None
        for _ in range(60):
            verdict = guard.observe(reasoning_texts=("verify ",))
            if verdict is not None:
                break
        assert verdict is not None
        assert verdict.reason == "reasoning_repetition"
        assert "period=7" in verdict.detail

    def test_production_defaults_tolerate_a_phrase_repeated_a_few_times(self) -> None:
        guard = StreamProgressGuard(StreamProgressConfig(empty_delta_limit=0))
        assert (
            guard.observe(reasoning_texts=("The user wants a Monday shift. ",)) is None
        )
        for _ in range(3):
            assert guard.observe(reasoning_texts=("Let me verify that. ",)) is None
        assert guard.observe(reasoning_texts=("Yes, Monday it is.",)) is None

    def test_degenerate_reasoning_on_the_finishing_delta_does_not_abort(self) -> None:
        guard = StreamProgressGuard(_config(degenerate_window=16))
        assert guard.observe(reasoning_texts=("x" * 15,)) is None
        assert (
            guard.observe(reasoning_texts=(" " * 16,), has_finish_reason=True) is None
        )

    def test_period_is_capped_at_half_the_window(self) -> None:
        """An operator setting max_period >= window must not turn "first and
        last character match" into "periodic"."""
        guard = StreamProgressGuard(
            _config(degenerate_window=16, degenerate_max_period=64)
        )
        text = "abcdefghijklmnoa"  # text[15] == text[0]; period 15 would match
        assert guard.observe(reasoning_texts=(text,)) is None

    def test_mirrored_reasoning_text_counts_once(self) -> None:
        """A provider sending the same text under ``reasoning_content`` and
        ``reasoning`` must not double it: the loop's true period is what the
        periodicity check has to see."""
        guard = StreamProgressGuard(
            _config(degenerate_window=32, degenerate_max_period=8)
        )
        assert (
            guard.observe(reasoning_texts=("Some genuine reasoning first. ",) * 2)
            is None
        )
        verdict = None
        for _ in range(10):
            verdict = guard.observe(reasoning_texts=("verify ", "verify "))
            if verdict is not None:
                break
        assert verdict is not None
        assert verdict.reason == "reasoning_repetition"
        assert "period=7" in verdict.detail

    def test_zero_window_disables_reasoning_checks(self) -> None:
        guard = StreamProgressGuard(_config(degenerate_window=0))
        for _ in range(50):
            assert guard.observe(reasoning_texts=(" " * 64,)) is None
            assert guard.observe(reasoning_texts=("ab" * 32,)) is None


class TestToolCallTrailingBytes:
    def test_open_object_growing_does_not_abort(self) -> None:
        """The ``create_shift`` runaway: the object never closes, so the guard
        leaves it to the ``max_tokens`` stop-loss."""
        guard = StreamProgressGuard(_config())
        arguments = '{"staff_ids": ['
        for i in range(200):
            arguments += f"{i},"
            assert (
                guard.observe(accumulated_tool_calls=_tool_calls(("call_1", arguments)))
                is None
            )

    def test_complete_object_alone_does_not_abort(self) -> None:
        """Unchanged state is not tool-call garbage (the empty-delta predicate
        owns that case and is switched off here)."""
        guard = StreamProgressGuard(_config(empty_delta_limit=0))
        calls = _tool_calls(("call_1", '{"q": "x"}'))
        for _ in range(10):
            assert guard.observe(accumulated_tool_calls=calls) is None

    def test_non_whitespace_after_complete_object_aborts_immediately(self) -> None:
        """A JSON object followed by anything but whitespace can never parse;
        the repeated-object loop (trace 4580191) is cut at its first extra byte."""
        guard = StreamProgressGuard(_config())
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"}')))
            is None
        )
        verdict = guard.observe(
            accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"}{"q"'))
        )
        assert verdict is not None
        assert verdict.reason == "tool_call_trailing_bytes"
        assert verdict.tool_call_id == "call_1"
        assert verdict.keep_length == len('{"q": "x"}')

    def test_whitespace_after_complete_object_aborts_at_the_window(self) -> None:
        guard = StreamProgressGuard(_config(degenerate_window=8))
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"}')))
            is None
        )
        assert (
            guard.observe(
                accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"}' + "\n" * 7))
            )
            is None
        )
        verdict = guard.observe(
            accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"}' + "\n\t" * 4))
        )
        assert verdict is not None
        assert verdict.reason == "tool_call_trailing_whitespace"
        assert verdict.tool_call_id == "call_1"
        assert verdict.keep_length == len('{"q": "x"}' + "\n" * 7)

    def test_trailing_whitespace_below_the_window_is_fine(self) -> None:
        """``}\\n`` arriving together is a normal provider tail, and whitespace
        may keep growing right up to ``window - 1`` characters."""
        guard = StreamProgressGuard(_config(degenerate_window=8))
        assert (
            guard.observe(
                accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"}\n'))
            )
            is None
        )
        assert (
            guard.observe(
                accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"}' + "\n" * 7))
            )
            is None
        )
        verdict = guard.observe(
            accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"}' + "\n" * 8))
        )
        assert verdict is not None
        assert verdict.reason == "tool_call_trailing_whitespace"
        assert verdict.keep_length == len('{"q": "x"}' + "\n" * 7)

    def test_closing_and_opening_brace_in_one_delta_is_caught(self) -> None:
        """``}{`` is a single BPE token on several models, so the repeated
        object arrives as ``{"q": "x"}{`` in one delta: the object is complete
        and the extra byte is garbage, in the same observation."""
        guard = StreamProgressGuard(_config())
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"')))
            is None
        )
        verdict = guard.observe(
            accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"}{'))
        )
        assert verdict is not None
        assert verdict.reason == "tool_call_trailing_bytes"
        assert verdict.keep_length == len('{"q": "x"')

    def test_leading_whitespace_before_the_object_is_allowed(self) -> None:
        guard = StreamProgressGuard(_config())
        assert (
            guard.observe(
                accumulated_tool_calls=_tool_calls(("call_1", '\n{"q": "x"}'))
            )
            is None
        )
        verdict = guard.observe(
            accumulated_tool_calls=_tool_calls(("call_1", '\n{"q": "x"}x'))
        )
        assert verdict is not None
        assert verdict.reason == "tool_call_trailing_bytes"

    def test_open_object_is_not_reparsed_without_a_closing_brace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A large open object must not be JSON-parsed on every delta."""
        from xagent.core.model.chat import stream_progress

        calls = {"n": 0}
        original = stream_progress._complete_object_length

        def counting(arguments: str) -> int | None:
            calls["n"] += 1
            return original(arguments)

        monkeypatch.setattr(stream_progress, "_complete_object_length", counting)
        guard = StreamProgressGuard(_config(empty_delta_limit=0))
        arguments = '{"staff_ids": ['
        for i in range(50):
            arguments += f"{i},"
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", arguments)))
        assert calls["n"] == 0
        guard.observe(accumulated_tool_calls=_tool_calls(("call_1", arguments + "1]}")))
        assert calls["n"] == 1

    def test_new_parallel_tool_call_after_a_complete_one_does_not_abort(self) -> None:
        guard = StreamProgressGuard(_config())
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", '{"q": "x"}')))
            is None
        )
        calls = _tool_calls(("call_1", '{"q": "x"}'), ("call_2", '{"q": '))
        assert guard.observe(accumulated_tool_calls=calls) is None
        calls = _tool_calls(("call_1", '{"q": "x"}'), ("call_2", '{"q": "y"}'))
        assert guard.observe(accumulated_tool_calls=calls) is None

    def test_non_object_json_is_not_treated_as_complete(self) -> None:
        """Only a top-level object marks completion; a bare string or number
        prefix must not arm the check (``"12"`` can legitimately grow to ``"123"``)."""
        guard = StreamProgressGuard(_config())
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", "12"))) is None
        )
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", "123"))) is None
        )
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", '"ab"')))
            is None
        )
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", '"ab"c')))
            is None
        )

    def test_zero_window_still_aborts_on_non_whitespace_but_never_on_whitespace(
        self,
    ) -> None:
        guard = StreamProgressGuard(_config(degenerate_window=0))
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", "{}"))) is None
        )
        assert (
            guard.observe(
                accumulated_tool_calls=_tool_calls(("call_1", "{}" + " " * 5000))
            )
            is None
        )
        verdict = guard.observe(
            accumulated_tool_calls=_tool_calls(("call_1", "{}" + " " * 5000 + "x"))
        )
        assert verdict is not None and verdict.reason == "tool_call_trailing_bytes"

    def test_tool_call_progress_resets_the_empty_counter(self) -> None:
        guard = StreamProgressGuard(_config(empty_delta_limit=2))
        assert guard.observe() is None
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", '{"q'))) is None
        )
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", '{"q'))) is None
        )
        # Unchanged accumulated state is not progress.
        verdict = guard.observe(accumulated_tool_calls=_tool_calls(("call_1", '{"q')))
        assert verdict is not None and verdict.reason == "empty_deltas"


class TestNoPayloadTimeout:
    def test_disabled_by_default(self) -> None:
        clock = iter([0.0, 1000.0, 2000.0])
        guard = StreamProgressGuard(_config(), clock=lambda: next(clock))
        assert guard.observe(reasoning_texts=("a",)) is None
        assert guard.observe(reasoning_texts=("b",)) is None

    def test_fires_after_the_timeout_without_payload(self) -> None:
        now = [0.0]
        guard = StreamProgressGuard(
            _config(no_payload_timeout=30.0), clock=lambda: now[0]
        )
        assert guard.observe(reasoning_texts=("a",)) is None
        now[0] = 29.9
        assert guard.observe(reasoning_texts=("b",)) is None
        now[0] = 30.0
        verdict = guard.observe(reasoning_texts=("c",))
        assert verdict is not None
        assert verdict.reason == "no_payload_timeout"

    def test_measured_from_the_first_delta_not_from_construction(self) -> None:
        now = [0.0]
        guard = StreamProgressGuard(
            _config(no_payload_timeout=30.0), clock=lambda: now[0]
        )
        now[0] = 100.0  # first-token latency is not counted
        assert guard.observe(reasoning_texts=("a",)) is None
        now[0] = 129.0
        assert guard.observe(reasoning_texts=("b",)) is None
        now[0] = 130.0
        assert guard.observe(reasoning_texts=("c",)) is not None

    def test_does_not_fire_on_the_finishing_delta(self) -> None:
        """A provider ending the stream itself (finish_reason, then usage) is
        not a stall: aborting there would only lose the usage chunk."""
        now = [0.0]
        guard = StreamProgressGuard(
            _config(no_payload_timeout=1.0), clock=lambda: now[0]
        )
        assert guard.observe(reasoning_texts=("a",)) is None
        now[0] = 500.0
        assert guard.observe(has_finish_reason=True) is None
        assert guard.observe(has_usage=True) is None

    def test_never_fires_once_payload_was_produced(self) -> None:
        now = [0.0]
        guard = StreamProgressGuard(
            _config(no_payload_timeout=1.0), clock=lambda: now[0]
        )
        assert guard.observe(has_content=True) is None
        now[0] = 500.0
        assert guard.observe(reasoning_texts=("still thinking",)) is None
        guard = StreamProgressGuard(
            _config(no_payload_timeout=1.0), clock=lambda: now[0]
        )
        assert (
            guard.observe(accumulated_tool_calls=_tool_calls(("call_1", "{"))) is None
        )
        now[0] = 1000.0
        assert guard.observe(reasoning_texts=("more",)) is None


class TestConfigFromEnv:
    def test_from_env_reads_config_getters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAGENT_LLM_STREAM_EMPTY_DELTA_LIMIT", "7")
        monkeypatch.setenv("XAGENT_LLM_STREAM_DEGENERATE_WINDOW", "99")
        monkeypatch.setenv("XAGENT_LLM_STREAM_DEGENERATE_MAX_PERIOD", "11")
        monkeypatch.setenv("XAGENT_LLM_STREAM_NO_PAYLOAD_ABORT_MODELS", "kimi, other")
        monkeypatch.setenv("XAGENT_LLM_STREAM_NO_PAYLOAD_TIMEOUT_SECONDS", "12.5")

        listed = StreamProgressConfig.from_env(model_name="kimi")
        assert listed.empty_delta_limit == 7
        assert listed.degenerate_window == 99
        assert listed.degenerate_max_period == 11
        assert listed.no_payload_timeout == 12.5

        not_listed = StreamProgressConfig.from_env(model_name="Kimi")
        assert not_listed.no_payload_timeout is None

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "XAGENT_LLM_STREAM_EMPTY_DELTA_LIMIT",
            "XAGENT_LLM_STREAM_DEGENERATE_WINDOW",
            "XAGENT_LLM_STREAM_DEGENERATE_MAX_PERIOD",
            "XAGENT_LLM_STREAM_NO_PAYLOAD_ABORT_MODELS",
            "XAGENT_LLM_STREAM_NO_PAYLOAD_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(name, raising=False)
        config = StreamProgressConfig.from_env(model_name="anything")
        assert config == StreamProgressConfig(
            empty_delta_limit=200,
            degenerate_window=256,
            degenerate_max_period=64,
            no_payload_timeout=None,
        )


class TestWireVocabulary:
    """The values below are written into ``llm_call_end`` trace rows and read
    by external analysis, so the constants are pinned to their literal wire
    values here: renaming a constant must fail a test, not silently change
    what the trace records. Other tests may assert either spelling."""

    def test_finish_reason_value_is_pinned(self) -> None:
        assert NO_PROGRESS_FINISH_REASON == "no_progress"

    def test_stream_fallback_value_is_pinned(self) -> None:
        assert NO_PAYLOAD_STREAM_FALLBACK == "no_payload"
