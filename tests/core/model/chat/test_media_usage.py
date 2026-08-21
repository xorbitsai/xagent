"""Media (non-LLM) usage tracking: image/video/tts/asr/embedding/rerank.

These modalities record usage via ``add_media_usage`` into the same
``TokenUsage.details`` list that LLM tokens use, so they flow through DB
persistence and the quota ``delta_details`` contract without special-casing.
"""

import pytest

from xagent.core.model.chat.token_context import (
    MediaCallType,
    MediaUnit,
    TokenContextManager,
    TokenUsage,
    add_media_usage,
    add_token_usage,
    aggregate_media_usage_by_model,
    aggregate_token_usage_by_model,
)


def test_add_media_usage_appends_media_entry_and_counts_call() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            quantity=2,
            model="sd-xl",
            model_id="m1",
            call_type="generate_image",
            resolution="1K",
        )
        usage = manager.get_usage()

    assert usage.media_calls == 1
    # Media does not count as an LLM call or add tokens.
    assert usage.llm_calls == 0
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert len(usage.details) == 1
    entry = usage.details[0]
    assert entry["type"] == "media"
    assert entry["unit"] == "images"
    assert entry["quantity"] == 2.0
    assert entry["model"] == "sd-xl"
    assert entry["model_id"] == "m1"
    assert entry["call_type"] == "generate_image"
    assert entry["resolution"] == "1K"


def test_add_media_usage_carries_accompanying_tokens() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            quantity=1,
            model="gemini-image",
            call_type="generate_image",
            input_tokens=5,
            output_tokens=3,
        )
        usage = manager.get_usage()

    entry = usage.details[0]
    # Stored under provider_tokens, never "tokens": a consumer that sums the
    # "tokens" key across all entries must not pick up media counts.
    assert "tokens" not in entry
    assert entry["provider_tokens"] == 8
    assert entry["provider_input_tokens"] == 5
    assert entry["provider_output_tokens"] == 3
    assert entry["tokens_estimated"] is False
    # Media token passthrough must NOT inflate the LLM token totals.
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_estimated_tokens_are_flagged() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            quantity=2,
            model="embed",
            call_type="embedding",
            input_tokens=12,
            tokens_estimated=True,
        )
        details = manager.get_usage().details

    assert details[0]["tokens_estimated"] is True
    # The flag survives aggregation so billing can refuse to price an estimate.
    assert aggregate_media_usage_by_model(details)[0]["tokens_estimated"] is True


def test_dirty_quantity_is_coerced_on_free_quantity_units() -> None:
    # A malformed quantity records 0 rather than raising: the provider call
    # happened, so the row must survive as "unmeasured". Uses a duration-billed
    # unit, because REQUESTS additionally pins its quantity to exactly 1 (see
    # test_requests_unit_rejects_any_quantity_but_one) and would reject these.
    with TokenContextManager() as manager:
        add_media_usage(quantity=None, model="x", call_type="asr")  # type: ignore[arg-type]
        add_media_usage(quantity="oops", model="x", call_type="asr")  # type: ignore[arg-type]
        usage = manager.get_usage()

    assert usage.media_calls == 2
    assert all(entry["quantity"] == 0.0 for entry in usage.details)


def test_dirty_quantity_on_requests_unit_is_rejected_not_coerced() -> None:
    # For REQUESTS the permissive path is wrong: coercing a malformed quantity
    # to 0 would record a call whose billable quantity contradicts its own
    # call count. Reject instead, leaving no state behind.
    with TokenContextManager() as manager:
        for bad in (None, "oops", 0):
            with pytest.raises(ValueError, match="exactly one call"):
                add_media_usage(quantity=bad, model="x", call_type="rerank")  # type: ignore[arg-type]
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_to_dict_from_dict_roundtrip_preserves_media() -> None:
    with TokenContextManager() as manager:
        add_token_usage(input_tokens=10, output_tokens=4, model="gpt", model_id="g1")
        add_media_usage(quantity=3.5, model="tts", call_type="asr")
        usage = manager.get_usage()

    data = usage.to_dict()
    assert data["media_calls"] == 1
    assert data["llm_calls"] == 1

    restored = TokenUsage.from_dict(data)
    assert restored.media_calls == 1
    assert restored.llm_calls == 1
    assert restored.input_tokens == 10
    # 2 token entries (one input, one output) + 1 media entry.
    assert len(restored.details) == 3
    assert sum(1 for d in restored.details if d["type"] == "media") == 1


def test_merge_combines_media_calls_and_details() -> None:
    a = TokenUsage()
    a.record_media_call(quantity=1, model="x", call_type="generate_image")

    b = TokenUsage()
    b.record_media_call(quantity=2, model="y", call_type="asr")

    a.merge(b)
    assert a.media_calls == 2
    assert len(a.details) == 2


def test_token_aggregation_ignores_media_entries() -> None:
    with TokenContextManager() as manager:
        add_token_usage(input_tokens=10, output_tokens=5, model="gpt", model_id="g1")
        add_media_usage(quantity=2, model="sd", call_type="generate_image")
        details = manager.get_usage().details

    token_groups = aggregate_token_usage_by_model(details)
    assert len(token_groups) == 1
    assert token_groups[0]["model_name"] == "gpt"
    assert token_groups[0]["input_tokens"] == 10
    assert token_groups[0]["output_tokens"] == 5


def test_media_aggregation_groups_by_model_unit_and_call_type() -> None:
    with TokenContextManager() as manager:
        add_media_usage(quantity=2, model="sd", call_type="generate_image")
        add_media_usage(quantity=3, model="sd", call_type="generate_image")
        add_media_usage(quantity=4, model="tts", call_type="asr")
        # LLM tokens must never appear in the media aggregation.
        add_token_usage(input_tokens=7, output_tokens=2, model="gpt", model_id="g1")
        details = manager.get_usage().details

    media_groups = aggregate_media_usage_by_model(details)
    assert len(media_groups) == 2

    by_unit = {group["unit"]: group for group in media_groups}
    assert by_unit["images"]["quantity"] == 5.0
    assert by_unit["images"]["calls"] == 2
    assert by_unit["images"]["call_type"] == "generate_image"
    assert by_unit["seconds"]["quantity"] == 4.0
    assert by_unit["seconds"]["calls"] == 1


def test_media_aggregation_splits_by_resolution() -> None:
    # Same model+call_type at different resolutions must bill as separate line
    # items, since an image model's price varies by resolution.
    with TokenContextManager() as manager:
        add_media_usage(
            quantity=1,
            model="gemini-image",
            call_type="generate_image",
            resolution="1K",
        )
        add_media_usage(
            quantity=1,
            model="gemini-image",
            call_type="generate_image",
            resolution="4K",
        )
        details = manager.get_usage().details

    groups = aggregate_media_usage_by_model(details)
    assert len(groups) == 2
    by_res = {group["resolution"]: group for group in groups}
    assert set(by_res) == {"1K", "4K"}
    assert by_res["1K"]["calls"] == 1
    assert by_res["4K"]["calls"] == 1


def test_aggregations_tolerate_non_list_and_dirty_entries() -> None:
    assert aggregate_media_usage_by_model(None) == []
    assert aggregate_media_usage_by_model("nope") == []
    # Non-dict junk is ignored; a bare media entry still counts as a call, since
    # a media row's existence is itself the billing signal.
    groups = aggregate_media_usage_by_model([{"type": "media"}, 42, "junk"])
    assert len(groups) == 1
    assert groups[0]["calls"] == 1
    assert groups[0]["quantity"] == 0.0


def test_zero_quantity_media_entries_stay_visible() -> None:
    # A duration-billed call the provider never measured records 0 seconds.
    # That entry must survive aggregation: it is the only evidence the task
    # made a billable provider call, and dropping it would report
    # media_calls=0 (and hide the whole popover) for a task that did.
    with TokenContextManager() as manager:
        add_media_usage(quantity=0, model="tts", call_type="asr")
        add_media_usage(quantity=5, model="tts", call_type="asr")
        details = manager.get_usage().details

    groups = aggregate_media_usage_by_model(details)
    assert len(groups) == 1
    assert groups[0]["quantity"] == 5.0
    assert groups[0]["calls"] == 2  # both calls counted, including the unmeasured one


def test_only_unmeasured_calls_still_surface() -> None:
    # The async-video case: no duration is available yet for any call, so the
    # whole group is zero-quantity. It must still be reported.
    with TokenContextManager() as manager:
        for _ in range(3):
            add_media_usage(quantity=0, model="veo", call_type="video")
        groups = aggregate_media_usage_by_model(manager.get_usage().details)

    assert len(groups) == 1
    assert groups[0]["calls"] == 3
    assert groups[0]["quantity"] == 0.0


def test_unknown_call_type_is_rejected() -> None:
    with TokenContextManager() as manager:
        with pytest.raises(ValueError, match="Unknown media call type"):
            add_media_usage(quantity=1, model="m", call_type="speech")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_enum_members_are_accepted() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            quantity=3,
            model="whisper",
            call_type=MediaCallType.ASR,
        )
        usage = manager.get_usage()

    entry = usage.details[0]
    # Stored as plain strings so details stays JSON-serialisable.
    assert entry["unit"] == "seconds"
    assert entry["call_type"] == "asr"
    assert isinstance(entry["unit"], str)


def test_legacy_positional_construction_still_binds_the_same_fields() -> None:
    # media_calls must NOT be inserted into the historical positional sequence.
    # An existing TokenUsage(input, output, llm_calls, tool_calls, details) call
    # would otherwise bind its 4th argument to media_calls and shift the rest —
    # tool count read as media count, details read as tool count — silently,
    # with no error. The field is appended and keyword-only instead.
    details = [{"type": "input", "tokens": 7}]
    usage = TokenUsage(10, 5, 2, 3, details)

    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.llm_calls == 2
    assert usage.tool_calls == 3
    assert usage.details == details
    assert usage.media_calls == 0

    # media_calls is derived from details, so it cannot be set at all — which
    # is what makes it impossible to drop on a copy or seed path.
    assert TokenUsage(details=[{"type": "media"}]).media_calls == 1

    # Five positional arguments is the maximum: media_calls is a derived
    # property, so there is no sixth field for an old caller to mis-bind.
    with pytest.raises(TypeError):
        TokenUsage(1, 2, 3, 4, [], 5)  # type: ignore[misc]


def test_estimate_media_tokens_accepts_any_iterable_of_strings() -> None:
    # The docstring promises an iterable; a generator previously estimated 0,
    # so an embedding producer passing one would have billed nothing.
    from xagent.core.model.chat.token_context import estimate_media_tokens

    assert estimate_media_tokens(x for x in ["abcd", "efgh"]) == 2
    assert estimate_media_tokens(["abcd", "efgh"]) == 2
    # Non-iterables and non-string members are ignored, never raised on.
    assert estimate_media_tokens(42) == 0
    assert estimate_media_tokens(["abcd", 42, None]) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0), ("a", 1), ("ab", 1), ("abc", 1), ("abcd", 1), ("abcde", 2)],
)
def test_estimate_media_tokens_never_undercounts_short_text(text, expected) -> None:
    # Truncating division estimated 0 for any 1-3 character Latin string, so
    # short non-empty text billed nothing. Empty still estimates 0.
    from xagent.core.model.chat.token_context import estimate_media_tokens

    assert estimate_media_tokens(text) == expected


@pytest.mark.parametrize("bad_quantity", [0, 0.5, 2, 7, -1])
def test_requests_unit_rejects_any_quantity_but_one(bad_quantity) -> None:
    # MediaUnit.REQUESTS is defined as exactly one provider call, so its
    # quantity is not a free variable. Letting another value through would make
    # the billable quantity disagree with the media_calls / aggregate `calls`
    # count derived from the same row, so quota and pricing would read
    # different numbers off one record.
    usage = TokenUsage()
    with pytest.raises(ValueError, match="exactly one call"):
        usage.record_media_call(quantity=bad_quantity, call_type="rerank")

    # Rejected before any mutation: no counter bump, no orphan detail row.
    assert usage.media_calls == 0
    assert usage.details == []


def test_requests_unit_accepts_exactly_one() -> None:
    usage = TokenUsage()
    usage.record_media_call(quantity=1, call_type="rerank")

    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 1.0


def test_requests_rejection_is_swallowed_by_the_wrapper() -> None:
    # Producers route through record_media_usage, whose contract is that an
    # accounting bug never breaks the media call the user asked for.
    from xagent.core.tools.core.media_usage import record_media_usage

    with TokenContextManager() as manager:
        record_media_usage(MediaCallType.RERANK, 3, model="m")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


@pytest.mark.parametrize("bad_tokens", [float("nan"), "not a number", None])
def test_unusable_provider_tokens_zero_out_without_losing_the_row(bad_tokens) -> None:
    # The row must survive a malformed token count: the provider call happened
    # and is billable by its quantity even when the token fields are junk.
    #
    # Overflow IS handled now — see test_overflowing_provider_tokens_keep_the_row.
    # Booleans still coerce to 0/1 in the shared `_coerce_int`, which is
    # deliberate: `add_token_usage` does `if input_tokens:`, so flooring there
    # would flip a real value from "added" to "silently skipped" on the live LLM
    # path. Negatives are clamped at the media boundary instead.
    usage = TokenUsage()
    usage.record_media_call(
        quantity=1,
        call_type="generate_image",
        input_tokens=bad_tokens,
        output_tokens=bad_tokens,
    )

    assert usage.media_calls == 1
    entry = usage.details[0]
    assert entry["quantity"] == 1.0
    assert entry["provider_input_tokens"] == 0
    assert entry["provider_output_tokens"] == 0
    assert entry["provider_tokens"] == 0


def test_valid_provider_tokens_survive() -> None:
    usage = TokenUsage()
    usage.record_media_call(
        quantity=1,
        call_type="generate_image",
        input_tokens=11,
        output_tokens=5,
    )
    entry = usage.details[0]

    assert (entry["provider_input_tokens"], entry["provider_output_tokens"]) == (11, 5)
    assert entry["provider_tokens"] == 16


def test_estimate_media_tokens_never_raises_from_a_hostile_iterable() -> None:
    # The docstring promises malformed input cannot raise, but only TypeError
    # was caught: a custom iterable raising anything else escaped into the
    # accounting path and would break the call being measured.
    from xagent.core.model.chat.token_context import estimate_media_tokens

    class _RaisesMidIteration:
        def __iter__(self):
            yield "abcd"
            raise RuntimeError("iterator exploded")

    class _RaisesImmediately:
        def __iter__(self):
            raise ValueError("cannot iterate")

    assert estimate_media_tokens(_RaisesMidIteration()) == 0
    assert estimate_media_tokens(_RaisesImmediately()) == 0


def test_requests_error_reports_the_value_the_caller_passed() -> None:
    # quantity is coerced before the guard runs, so interpolating the coerced
    # value would report "got 0.0" for a caller who passed -1.
    usage = TokenUsage()
    with pytest.raises(ValueError, match=r"got -1"):
        usage.record_media_call(quantity=-1, call_type="rerank")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # CJK is ~1 token per character; a flat chars/4 heuristic undercounts
        # Chinese by close to 4x. Punctuation and fullwidth forms count too —
        # they appear in essentially every Chinese sentence.
        ("中文", 2),
        ("中文abcd", 3),
        ("中文，你好。", 6),
        ("（全角）", 4),
    ],
)
def test_estimate_media_tokens_counts_cjk(text, expected) -> None:
    from xagent.core.model.chat.token_context import estimate_media_tokens

    assert estimate_media_tokens(text) == expected


def test_media_aggregation_uses_model_id_when_present() -> None:
    # identity = model_id or model_name — the model_id branch had no coverage,
    # so rows keyed by id always fell through to the name branch in tests.
    with TokenContextManager() as manager:
        # Empty-name row FIRST: with the named row first, setdefault seeds the
        # group's model_name and the backfill branch never executes -- the test
        # then passes with that branch deleted.
        add_media_usage(
            quantity=2,
            model="",
            model_id="img-1",
            call_type="generate_image",
        )
        add_media_usage(
            quantity=1,
            model="display-name",
            model_id="img-1",
            call_type="generate_image",
        )
        groups = aggregate_media_usage_by_model(manager.get_usage().details)

    # Both rows share one identity via model_id even though one lacks a name...
    assert len(groups) == 1
    assert groups[0]["model_id"] == "img-1"
    assert groups[0]["quantity"] == 3.0
    # ...and the name is backfilled from whichever row carried it.
    assert groups[0]["model_name"] == "display-name"


@pytest.mark.parametrize("huge", [10**400, -(10**400), "1e400"])
def test_huge_quantity_records_zero_rather_than_dropping_the_row(huge) -> None:
    # float(10**400) raises OverflowError, which _coerce_float did not catch.
    # An uncaught raise here propagates out and the error-swallowing wrapper
    # drops the whole billing row — the opposite of the reject-to-0.0 contract.
    usage = TokenUsage()
    usage.record_media_call(quantity=huge, call_type="generate_image")

    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 0.0


def test_add_media_usage_reports_the_raw_quantity_in_the_requests_error() -> None:
    # The wrapper coerces one layer above record_media_call, so without
    # threading the raw value through, a caller passing -1 saw "got 0.0".
    with TokenContextManager():
        with pytest.raises(ValueError, match=r"got -1"):
            add_media_usage(quantity=-1, model="m", call_type="rerank")


@pytest.mark.parametrize("overflowing", [float("inf"), float("-inf")])
def test_overflowing_provider_tokens_keep_the_row(overflowing) -> None:
    # int(float("inf")) raises OverflowError. Uncaught, it propagates out of the
    # accounting path and the error-swallowing wrapper drops the entire billing
    # row — losing a call that did happen in order to salvage one bad field.
    # Image providers pass prompt_tokens straight from provider JSON, so a
    # malformed payload reaches this boundary unfiltered.
    usage = TokenUsage()
    usage.record_media_call(
        quantity=1,
        call_type="generate_image",
        input_tokens=overflowing,
        output_tokens=overflowing,
    )

    assert usage.media_calls == 1
    entry = usage.details[0]
    assert entry["quantity"] == 1.0
    assert entry["provider_input_tokens"] == 0
    assert entry["provider_output_tokens"] == 0


@pytest.mark.parametrize("unusable", [-1, -0.5, -1000.0, True, float("nan")])
def test_unusable_quantity_records_zero(unusable) -> None:
    # A negative quantity would subtract from a bill, and a bool is a caller bug
    # rather than a count of 1 or 0 — `True` would otherwise bill one image.
    # Recorded as 0 rather than rejected, because the provider call still
    # happened: the row is the evidence, and 0 marks it unmeasured.
    usage = TokenUsage()
    usage.record_media_call(quantity=unusable, call_type="generate_image")

    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 0.0


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), "abc"])
def test_coerce_int_change_is_safe_on_the_llm_token_path(bad) -> None:
    # This PR's one functional change to pre-existing code adds OverflowError to
    # `_coerce_int`, which `add_token_usage`, `extract_cached_input_tokens` and
    # `aggregate_token_usage_by_model` all use on the live LLM path. At base
    # `_coerce_int(float("inf"))` raised; it must now record 0 without raising.
    with TokenContextManager() as manager:
        add_token_usage(
            input_tokens=bad, output_tokens=bad, model="m", call_type="chat"
        )  # type: ignore[arg-type]
        usage = manager.get_usage()

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


@pytest.mark.parametrize("bad_call_type", ["not-a-type", "IMAGES", "", None])
def test_unknown_call_type_is_rejected_and_required(bad_call_type) -> None:
    # call_type is the only identity now — the unit is derived from it — so an
    # unknown or missing value has nothing to derive from and must be rejected
    # rather than silently skipping enforcement, which is what an optional
    # call_type did in the earlier design.
    usage = TokenUsage()
    with pytest.raises(ValueError):
        usage.record_media_call(call_type=bad_call_type, quantity=1)  # type: ignore[arg-type]

    assert usage.media_calls == 0
    assert usage.details == []


def test_unit_cannot_be_passed_at_all() -> None:
    # The whole point of the redesign: a wrong (unit, modality) pair is not
    # rejected, it is unrepresentable.
    usage = TokenUsage()
    with pytest.raises(TypeError):
        usage.record_media_call(  # type: ignore[call-arg]
            unit="seconds", call_type="generate_image", quantity=1
        )


@pytest.mark.parametrize("call_type", list(MediaCallType))
def test_every_call_type_carries_a_unit_and_records_it(call_type) -> None:
    # Parametrised from the enum itself, not a hand-written list: a new member
    # added without a unit fails here rather than silently skipping enforcement.
    usage = TokenUsage()
    usage.record_media_call(
        call_type=call_type, quantity=1 if call_type.unit is MediaUnit.REQUESTS else 2
    )

    assert usage.details[0]["unit"] == call_type.unit.value
    assert isinstance(call_type.unit, MediaUnit)


@pytest.mark.parametrize("truthy_non_bool", ["no", "false", "0", 1, [0], object()])
def test_tokens_estimated_only_accepts_a_real_true(truthy_non_bool) -> None:
    # bool() is not enough here: bool("no") and bool("false") are both True, so
    # a provider adapter forwarding a string flag would mark measured counts as
    # estimated and let them through billing as guesses. Only an actual True
    # sets the flag.
    usage = TokenUsage()
    usage.record_media_call(
        call_type=MediaCallType.ASR, quantity=1, tokens_estimated=truthy_non_bool
    )

    assert usage.details[0]["tokens_estimated"] is False


def test_tokens_estimated_true_still_sets_the_flag() -> None:
    usage = TokenUsage()
    usage.record_media_call(
        call_type=MediaCallType.ASR, quantity=1, tokens_estimated=True
    )

    assert usage.details[0]["tokens_estimated"] is True


def test_negative_provider_tokens_are_floored_not_subtracted() -> None:
    # A provider returning a negative token count must not credit the tenant
    # back tokens it never spent. Floored at the media boundary only, so the
    # LLM path's `if input_tokens:` keeps seeing real values.
    usage = TokenUsage()
    usage.record_media_call(
        call_type=MediaCallType.TTS,
        quantity=10,
        input_tokens=-500,
        output_tokens=-7,
    )
    entry = usage.details[0]

    assert entry["provider_input_tokens"] == 0
    assert entry["provider_output_tokens"] == 0
    assert entry["provider_tokens"] == 0
    # The row survives as evidence the call happened.
    assert entry["quantity"] == 10.0


def test_model_identity_is_stripped_so_padding_does_not_split_billing() -> None:
    # " sd " and "sd" are the same model; leaving the padding in produces two
    # aggregation groups and two invoice lines for one model.
    usage = TokenUsage()
    usage.record_media_call(
        call_type=MediaCallType.GENERATE_IMAGE,
        quantity=1,
        model=" sd ",
        model_id=" s1 ",
    )
    usage.record_media_call(
        call_type=MediaCallType.GENERATE_IMAGE, quantity=1, model="sd", model_id="s1"
    )

    assert {d["model"] for d in usage.details} == {"sd"}
    assert {d["model_id"] for d in usage.details} == {"s1"}
    assert len(aggregate_media_usage_by_model(usage.details)) == 1


@pytest.mark.parametrize("malformed", [None, "not-a-list", {"a": 1}, 7])
def test_from_dict_coerces_a_malformed_details_field(malformed) -> None:
    # media_calls is derived by iterating details, so a persisted
    # `details: null` -- harmless while the counter was a stored field --
    # would now raise on every read of media_calls, to_dict and merge.
    # Coerce at the read boundary instead of making each consumer defensive.
    restored = TokenUsage.from_dict(
        {"input_tokens": 3, "media_calls": 9, "details": malformed}
    )

    assert restored.details == []
    assert restored.media_calls == 0
    assert restored.input_tokens == 3
    # The three operations that would otherwise raise.
    assert restored.to_dict()["media_calls"] == 0
    TokenUsage().merge(restored)
    restored.record_media_call(call_type=MediaCallType.ASR, quantity=1)
    assert restored.media_calls == 1


def test_from_dict_keeps_a_real_details_list() -> None:
    # The coercion must not eat valid rows.
    restored = TokenUsage.from_dict(
        {
            "details": [
                {"type": "media", "unit": "seconds", "quantity": 2.0},
                {"type": "input", "tokens": 5},
            ]
        }
    )

    assert len(restored.details) == 2
    assert restored.media_calls == 1


def test_fullwidth_ascii_is_classified_per_character_not_per_block() -> None:
    # U+FF01-FF5E mixes two billing rates in one Unicode block, so this is
    # asserted over the whole block rather than by example: an earlier revision
    # narrowed the range to fix fullwidth *letters* over-counting 4x and
    # silently regressed fullwidth *punctuation* (（）) in the same edit.
    from xagent.core.model.chat.token_context import estimate_media_tokens

    misclassified = []
    for code in range(0xFF01, 0xFF5F):
        char = chr(code)
        # Every char in this block is the fullwidth form of an ASCII char.
        is_alnum = chr(code - 0xFEE0).isalnum()
        # CJK rate: one token per character. Latin rate: four chars per token.
        billed_per_char = estimate_media_tokens(char * 4) == 4
        if is_alnum == billed_per_char:
            misclassified.append((hex(code), chr(code - 0xFEE0)))

    assert misclassified == []


@pytest.mark.parametrize(
    "char,expected_cjk",
    [
        ("｟", True),  # FULLWIDTH LEFT WHITE PARENTHESIS
        ("｠", True),  # FULLWIDTH RIGHT WHITE PARENTHESIS
        ("｡", True),  # HALFWIDTH IDEOGRAPHIC FULL STOP
        ("Ａ", False),  # FULLWIDTH LATIN CAPITAL A
        ("０", False),  # FULLWIDTH DIGIT ZERO
    ],
)
def test_fullwidth_range_boundaries(char, expected_cjk) -> None:
    # U+FF5F/FF60 sat in the gap between the two ranges and billed as Latin.
    from xagent.core.model.chat.token_context import estimate_media_tokens

    assert (estimate_media_tokens(char * 4) == 4) is expected_cjk


@pytest.mark.parametrize("boolean", [True, False])
def test_boolean_provider_tokens_are_not_counted_as_one(boolean) -> None:
    # bool is an int subclass, so _coerce_int(True) is 1 and a provider handing
    # back a JSON boolean would bill a token. `quantity` at this same boundary
    # already rejects booleans; tokens must agree.
    usage = TokenUsage()
    usage.record_media_call(
        call_type=MediaCallType.TTS,
        quantity=10,
        input_tokens=boolean,
        output_tokens=boolean,
    )
    entry = usage.details[0]

    assert entry["provider_input_tokens"] == 0
    assert entry["provider_output_tokens"] == 0
    assert entry["provider_tokens"] == 0
    # The row still records the measured call.
    assert entry["quantity"] == 10.0


def test_resolution_is_stripped_so_padding_does_not_split_billing() -> None:
    # resolution is part of the aggregate key, so ' 1K ' and '1K' would become
    # two billable line items for one resolution tier -- and the padded row
    # would miss an exact price-table join.
    usage = TokenUsage()
    usage.record_media_call(
        call_type=MediaCallType.GENERATE_IMAGE,
        quantity=1,
        model="sd",
        resolution=" 1K ",
    )
    usage.record_media_call(
        call_type=MediaCallType.GENERATE_IMAGE,
        quantity=1,
        model="sd",
        resolution="1K",
    )

    assert {d["resolution"] for d in usage.details} == {"1K"}
    assert len(aggregate_media_usage_by_model(usage.details)) == 1
