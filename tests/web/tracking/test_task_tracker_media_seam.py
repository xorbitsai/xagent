"""Media rows must survive the TaskTracker -> quota seam.

The whole design bet of the media metering primitives is that media usage rides
the *existing* ``TokenUsage.details`` list, so it reaches quota accounting
through the same path as token usage with no new plumbing and no migration.
That claim is only worth making if the seam is actually exercised: the two
helpers below are the exact functions the metering path and the mid-run quota
gate call, and both are generic over ``details`` in a way that could silently
drop or mangle a media row.
"""

from __future__ import annotations

from xagent.core.model.chat.token_context import (
    MediaCallType,
    TokenUsage,
    add_media_usage,
    add_token_usage,
)
from xagent.web.tracking.task_tracker import _copy_details, _copy_usage


def _mixed_usage() -> TokenUsage:
    usage = TokenUsage()
    usage.add_input_tokens(10, "gpt", "chat", "g1")
    usage.add_output_tokens(4, "gpt", "chat", "g1")
    usage.record_media_call(
        call_type=MediaCallType.ASR, quantity=12.5, model="whisper", model_id="w1"
    )
    return usage


def test_copy_details_preserves_media_rows_verbatim() -> None:
    usage = _mixed_usage()

    copied = _copy_details(usage.details)

    assert len(copied) == 3
    media = [d for d in copied if d.get("type") == "media"]
    assert len(media) == 1
    # Verbatim, not merely present: every billing field has to arrive intact,
    # since this is the payload quota accounting prices.
    original = [d for d in usage.details if d.get("type") == "media"][0]
    assert media[0] == original
    assert media[0]["unit"] == "seconds"
    assert media[0]["quantity"] == 12.5


def test_copy_details_is_a_deep_enough_copy_to_detach_media_rows() -> None:
    # The copy exists so a database worker sees a stable snapshot. A media row
    # mutated after the copy must not change what gets metered.
    usage = _mixed_usage()
    copied = _copy_details(usage.details)

    for row in usage.details:
        if row.get("type") == "media":
            row["quantity"] = 9999.0

    assert [d for d in copied if d.get("type") == "media"][0]["quantity"] == 12.5


def test_copy_usage_keeps_the_media_call_count() -> None:
    # _copy_usage rebuilds TokenUsage from an explicit kwarg list that does not
    # mention media_calls. That is only safe because media_calls is derived from
    # details; if it were ever reintroduced as a stored field, this snapshot
    # would report zero media calls while carrying the rows.
    usage = _mixed_usage()

    snapshot = _copy_usage(usage)

    assert snapshot.media_calls == usage.media_calls == 1
    assert snapshot.llm_calls == usage.llm_calls
    assert len(snapshot.details) == 3


def test_turn_delta_slice_carries_media_rows_added_after_the_baseline() -> None:
    # _turn_delta is a bare slice of details from a baseline index, with no type
    # filter. Media rows recorded mid-run must therefore appear in the delta the
    # quota gate prices -- and rows from before the baseline must not.
    from xagent.core.model.chat.token_context import TokenContextManager

    with TokenContextManager() as manager:
        add_token_usage(input_tokens=5, output_tokens=1, model="gpt", model_id="g1")
        baseline = len(manager.get_usage().details)

        add_media_usage(quantity=3, model="sd", call_type=MediaCallType.GENERATE_IMAGE)
        add_token_usage(input_tokens=7, output_tokens=2, model="gpt", model_id="g1")

        delta = manager.get_usage().details[baseline:]

    kinds = [d.get("type") for d in delta]
    assert kinds.count("media") == 1
    media = [d for d in delta if d.get("type") == "media"][0]
    assert media["unit"] == "images"
    assert media["quantity"] == 3.0
