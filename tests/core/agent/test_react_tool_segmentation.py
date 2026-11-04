"""Inc.2 — ReAct tool-call segmentation (design §4.2.1).

``_next_segment`` is the pure partitioning function that slices a turn's
pending tool calls into consecutive segments:
- control tools (final_answer / send_message / ask_user_question) own a segment;
- non-concurrency-safe tools run serially, one per segment;
- consecutive concurrency-safe tools merge into one concurrent batch;
- a concurrent batch of length 1 degrades to a serial segment.

When the parallel flag is off, every segment is a single serial call (current
behavior, byte-for-byte). This increment only covers the function; it is not
yet wired into the execution loop.
"""

from __future__ import annotations

import pytest

from tests.core.agent.concurrency_harness import (
    FakeTool,
    make_react,
    make_tool_call,
)

# Tool catalog: name -> concurrency_safe
_SAFE_TOOLS = ["s1", "s2", "s3"]
_UNSAFE_TOOLS = ["u1", "u2"]
_CONTROL_TOOLS = ["final_answer", "send_message", "ask_user_question"]


def _build_tools() -> list[FakeTool]:
    tools: list[FakeTool] = []
    for name in _SAFE_TOOLS:
        tools.append(FakeTool(name, concurrency_safe=True))
    for name in _UNSAFE_TOOLS:
        tools.append(FakeTool(name, concurrency_safe=False))
    return tools


def _segments(pattern, names: list[str]) -> list[tuple[str, list[str]]]:
    """Drive _next_segment like the loop would, returning (kind, names) list."""
    tools = _build_tools()
    pending = [make_tool_call(name) for name in names]
    out: list[tuple[str, list[str]]] = []
    guard = 0
    while pending:
        guard += 1
        assert guard < 100, "segmentation did not make progress"
        segment, kind = pattern._next_segment(pending, tools)
        assert segment, "segment must be non-empty"
        out.append((kind, [tc["name"] for tc in segment]))
        pending = pending[len(segment) :]
    return out


def test_two_safe_tools_form_one_concurrent_segment() -> None:
    pattern = make_react(parallel=True)
    assert _segments(pattern, ["s1", "s2"]) == [("concurrent", ["s1", "s2"])]


def test_single_safe_tool_degrades_to_serial() -> None:
    pattern = make_react(parallel=True)
    assert _segments(pattern, ["s1"]) == [("serial", ["s1"])]


def test_mixed_sequence_partitions_into_expected_segments() -> None:
    pattern = make_react(parallel=True)
    # [S S U S C] -> concurrent[S,S], serial[U], serial[S], control[C]
    assert _segments(pattern, ["s1", "s2", "u1", "s3", "final_answer"]) == [
        ("concurrent", ["s1", "s2"]),
        ("serial", ["u1"]),
        ("serial", ["s3"]),
        ("control", ["final_answer"]),
    ]


def test_control_tool_always_owns_its_segment() -> None:
    pattern = make_react(parallel=True)
    # [S C S] -> the lone S degrades to serial, control owns its own segment.
    assert _segments(pattern, ["s1", "send_message", "s2"]) == [
        ("serial", ["s1"]),
        ("control", ["send_message"]),
        ("serial", ["s2"]),
    ]


def test_three_consecutive_safe_tools_merge() -> None:
    pattern = make_react(parallel=True)
    assert _segments(pattern, ["s1", "s2", "s3"]) == [
        ("concurrent", ["s1", "s2", "s3"])
    ]


def test_concurrent_batch_capped_at_max_concurrency() -> None:
    # A batch never grows past the concurrency width, so a mid-turn interrupt is
    # honored after at most one wave. With max_concurrency=2 the trailing lone
    # safe tool falls into its own (serial) segment.
    pattern = make_react(parallel=True, max_concurrency=2)
    assert _segments(pattern, ["s1", "s2", "s3"]) == [
        ("concurrent", ["s1", "s2"]),
        ("serial", ["s3"]),
    ]


def test_capping_splits_long_run_into_multiple_concurrent_batches() -> None:
    # Six consecutive safe tools at max_concurrency=3 split into two concurrent
    # batches of three; each batch boundary is an interrupt checkpoint.
    pattern = make_react(parallel=True, max_concurrency=3)
    assert _segments(pattern, ["s1", "s2", "s3", "s1", "s2", "s3"]) == [
        ("concurrent", ["s1", "s2", "s3"]),
        ("concurrent", ["s1", "s2", "s3"]),
    ]


def test_unsafe_tool_breaks_the_concurrent_batch() -> None:
    pattern = make_react(parallel=True)
    assert _segments(pattern, ["s1", "u1", "s2", "s3"]) == [
        ("serial", ["s1"]),
        ("serial", ["u1"]),
        ("concurrent", ["s2", "s3"]),
    ]


@pytest.mark.parametrize(
    "names",
    [
        ["s1", "s2"],
        ["s1", "s2", "s3"],
        ["s1", "u1", "s2", "final_answer"],
    ],
)
def test_flag_off_makes_everything_serial_length_one(names: list[str]) -> None:
    pattern = make_react(parallel=False)
    segments = _segments(pattern, names)
    assert len(segments) == len(names)
    for (kind, seg_names), original in zip(segments, names):
        assert len(seg_names) == 1
        assert seg_names[0] == original
        # control tools still classify as control even with the flag off.
        expected = "control" if original in _CONTROL_TOOLS else "serial"
        assert kind == expected


def test_unknown_tool_treated_as_not_concurrency_safe() -> None:
    pattern = make_react(parallel=True)
    # 'mystery' is not in the tool catalog -> conservative serial.
    assert _segments(pattern, ["mystery", "s1", "s2"]) == [
        ("serial", ["mystery"]),
        ("concurrent", ["s1", "s2"]),
    ]
