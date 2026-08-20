"""Media (non-LLM) usage tracking: image/video/tts/asr/embedding/rerank.

These modalities record usage via ``add_media_usage`` into the same
``TokenUsage.details`` list that LLM tokens use, so they flow through DB
persistence and the quota ``delta_details`` contract without special-casing.
"""

import sys
import threading

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
            unit="images",
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
            unit="images",
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
            unit="texts",
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
        add_media_usage(unit="seconds", quantity=None, model="x", call_type="asr")  # type: ignore[arg-type]
        add_media_usage(unit="seconds", quantity="oops", model="x", call_type="asr")  # type: ignore[arg-type]
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
                add_media_usage(
                    unit="requests", quantity=bad, model="x", call_type="rerank"
                )  # type: ignore[arg-type]
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_to_dict_from_dict_roundtrip_preserves_media() -> None:
    with TokenContextManager() as manager:
        add_token_usage(input_tokens=10, output_tokens=4, model="gpt", model_id="g1")
        add_media_usage(unit="seconds", quantity=3.5, model="tts", call_type="asr")
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
    a.record_media_call(
        unit="images", quantity=1, model="x", call_type="generate_image"
    )

    b = TokenUsage()
    b.record_media_call(unit="seconds", quantity=2, model="y", call_type="asr")

    a.merge(b)
    assert a.media_calls == 2
    assert len(a.details) == 2


def test_token_aggregation_ignores_media_entries() -> None:
    with TokenContextManager() as manager:
        add_token_usage(input_tokens=10, output_tokens=5, model="gpt", model_id="g1")
        add_media_usage(
            unit="images", quantity=2, model="sd", call_type="generate_image"
        )
        details = manager.get_usage().details

    token_groups = aggregate_token_usage_by_model(details)
    assert len(token_groups) == 1
    assert token_groups[0]["model_name"] == "gpt"
    assert token_groups[0]["input_tokens"] == 10
    assert token_groups[0]["output_tokens"] == 5


def test_media_aggregation_groups_by_model_unit_and_call_type() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            unit="images", quantity=2, model="sd", call_type="generate_image"
        )
        add_media_usage(
            unit="images", quantity=3, model="sd", call_type="generate_image"
        )
        add_media_usage(unit="seconds", quantity=4, model="tts", call_type="asr")
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
            unit="images",
            quantity=1,
            model="gemini-image",
            call_type="generate_image",
            resolution="1K",
        )
        add_media_usage(
            unit="images",
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
        add_media_usage(unit="seconds", quantity=0, model="tts", call_type="asr")
        add_media_usage(unit="seconds", quantity=5, model="tts", call_type="asr")
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
            add_media_usage(unit="seconds", quantity=0, model="veo", call_type="video")
        groups = aggregate_media_usage_by_model(manager.get_usage().details)

    assert len(groups) == 1
    assert groups[0]["calls"] == 3
    assert groups[0]["quantity"] == 0.0


@pytest.mark.parametrize("bad_unit", ["image", "second", "tokens", "IMAGES", ""])
def test_unknown_unit_is_rejected(bad_unit: str) -> None:
    # A typo'd unit mints a new billing dimension that the aggregator will
    # happily key off, and a written usage record cannot be repaired
    # retroactively. The write boundary is the last point it is still fixable.
    with TokenContextManager() as manager:
        with pytest.raises(ValueError, match="Unknown media unit"):
            add_media_usage(unit=bad_unit, quantity=1, model="m", call_type="tts")
        usage = manager.get_usage()

    # A rejected call must leave no partial state behind: media_calls is
    # incremented before the detail entry is appended, so validating late would
    # record a call with no matching entry.
    assert usage.media_calls == 0
    assert usage.details == []


def test_unknown_call_type_is_rejected() -> None:
    with TokenContextManager() as manager:
        with pytest.raises(ValueError, match="Unknown media call type"):
            add_media_usage(unit="seconds", quantity=1, model="m", call_type="speech")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_empty_call_type_is_allowed() -> None:
    # call_type is optional metadata rather than a billing dimension of its
    # own, so omitting it stays legal while a typo does not.
    with TokenContextManager() as manager:
        add_media_usage(unit="requests", quantity=1, model="m")
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["call_type"] == ""


def test_none_call_type_is_normalised_to_empty() -> None:
    with TokenContextManager() as manager:
        add_media_usage(unit="requests", quantity=1, model="m", call_type=None)
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["call_type"] == ""


def test_none_unit_is_rejected_with_clear_error() -> None:
    with TokenContextManager() as manager:
        with pytest.raises(ValueError, match="Media unit cannot be None"):
            add_media_usage(unit=None, quantity=1, model="m")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_enum_members_are_accepted() -> None:
    with TokenContextManager() as manager:
        add_media_usage(
            unit=MediaUnit.SECONDS,
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


def test_concurrent_snapshots_never_see_a_torn_counter_and_rows_pair() -> None:
    # What the lock actually protects is the PAIRING of the counter with its
    # detail rows, not `+=` itself. On CPython 3.12 a bare `self.n += 1` does not
    # measurably lose increments even at 8x200000 threads-iterations, so a test
    # that only counts increments passes with the lock entirely removed — an
    # earlier version of this test did exactly that.
    #
    # record_media_call bumps the counter and appends the row as one operation.
    # Without the lock a concurrent reader lands between them and observes
    # media_calls disagreeing with the number of media rows. Mutation-verified:
    # replacing _lock with contextlib.nullcontext() makes this fail in 30/30
    # runs, typically within the first few hundred writes.
    usage = TokenUsage()
    torn: list[tuple[int, int]] = []
    stop = threading.Event()

    # Required, not decoration: at the default switch interval the window is too
    # narrow to hit, and this test passes with the lock removed. At 1e-9 the
    # unlocked variant tears in 10/10 runs.
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-9)

    def writer() -> None:
        for _ in range(3000):
            usage.record_media_call(
                unit="images", quantity=1, call_type="generate_image"
            )
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            snap = usage.to_dict()
            media = sum(1 for d in snap["details"] if d.get("type") == "media")
            if snap["media_calls"] != media:
                torn.append((snap["media_calls"], media))
                return

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    finally:
        sys.setswitchinterval(previous_interval)

    assert not torn, f"observed torn counter/rows pairs: {torn[:5]}"
    assert usage.media_calls == 3000
    assert sum(1 for d in usage.details if d.get("type") == "media") == 3000


def test_concurrent_merge_loses_no_counts() -> None:
    # Each thread merges repeatedly and all start together at a barrier, so the
    # read-modify-write in merge() genuinely overlaps. One merge per thread of a
    # one-row source would very likely pass against an unlocked merge() too.
    target = TokenUsage()
    workers, merges_per_worker = 8, 100

    source = TokenUsage()
    source.record_media_call(unit="seconds", quantity=2, call_type="asr")

    barrier = threading.Barrier(workers)

    def merge_many() -> None:
        barrier.wait(timeout=10)
        for _ in range(merges_per_worker):
            target.merge(source)

    threads = [threading.Thread(target=merge_many) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    for thread in threads:
        assert not thread.is_alive(), "merge deadlocked under contention"

    expected = workers * merges_per_worker
    assert target.media_calls == expected
    media_rows = [d for d in target.details if d.get("type") == "media"]
    assert len(media_rows) == expected
    # The counter and its rows must agree exactly — a lost += shows up here.
    assert target.media_calls == len(media_rows)


def test_usage_survives_asdict_deepcopy_and_pickle() -> None:
    # The mutation lock is not picklable, so as a dataclass field it would break
    # every generic consumer of this object (dataclasses.asdict walks
    # __dataclass_fields__ and ignores __getstate__).
    import copy
    import dataclasses
    import pickle

    usage = TokenUsage()
    usage.record_media_call(unit="images", quantity=2, call_type="generate_image")

    assert dataclasses.asdict(usage)["media_calls"] == 1
    assert copy.deepcopy(usage).media_calls == 1
    revived = pickle.loads(pickle.dumps(usage))
    assert revived.media_calls == 1
    # The revived object gets a working lock of its own, not a shared one.
    revived.record_media_call(unit="images", quantity=1, call_type="generate_image")
    assert revived.media_calls == 2


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

    # And it is reachable by keyword, where it cannot be confused with anything.
    assert TokenUsage(media_calls=4).media_calls == 4

    # The private lock must not occupy a positional slot either: six positional
    # arguments is the documented maximum (five fields plus self).
    with pytest.raises(TypeError):
        TokenUsage(1, 2, 3, 4, [], 5)  # type: ignore[misc]


def test_concurrent_merges_in_both_directions_do_not_deadlock() -> None:
    # merge() snapshots the source under its own lock and releases it before
    # taking the target's; holding both at once would deadlock on opposing
    # merges. A plain thread start/join can be scheduled so the two never
    # actually overlap, which would pass even against a both-locks-held
    # implementation — so a Barrier forces them into the window.
    a = TokenUsage()
    b = TokenUsage()
    per_side = 50
    for _ in range(per_side):
        a.record_media_call(unit="images", quantity=1, call_type="generate_image")
        b.record_media_call(unit="seconds", quantity=1, call_type="asr")

    barrier = threading.Barrier(2)

    def merge_after_barrier(target: TokenUsage, source: TokenUsage) -> None:
        barrier.wait(timeout=10)
        target.merge(source)

    threads = [
        threading.Thread(target=merge_after_barrier, args=(a, b)),
        threading.Thread(target=merge_after_barrier, args=(b, a)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    for thread in threads:
        assert not thread.is_alive(), "merge deadlocked in opposing directions"

    # Exact counts, not lower bounds: a lower bound would pass while rows were
    # dropped or a counter drifted from its detail rows. Each side ends with its
    # own 50 plus whatever snapshot it read of the other — between 50 and 100
    # extra depending on interleaving — but the scalar and the rows must always
    # agree with each other, and neither side may lose its own rows.
    for usage, own_unit in ((a, "images"), (b, "seconds")):
        media_rows = [d for d in usage.details if d.get("type") == "media"]
        assert usage.media_calls == len(media_rows), "counter drifted from rows"
        assert len(media_rows) >= 2 * per_side
        assert len(media_rows) <= 3 * per_side
        own_rows = [d for d in media_rows if d["unit"] == own_unit]
        assert len(own_rows) >= per_side, "a side lost its own rows"


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
        usage.record_media_call(
            unit=MediaUnit.REQUESTS, quantity=bad_quantity, call_type="rerank"
        )

    # Rejected before any mutation: no counter bump, no orphan detail row.
    assert usage.media_calls == 0
    assert usage.details == []


def test_requests_unit_accepts_exactly_one() -> None:
    usage = TokenUsage()
    usage.record_media_call(unit=MediaUnit.REQUESTS, quantity=1, call_type="rerank")

    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 1.0


def test_other_units_keep_free_quantities() -> None:
    # The REQUESTS constraint must not leak into duration/count-billed units.
    usage = TokenUsage()
    usage.record_media_call(unit=MediaUnit.SECONDS, quantity=2.5, call_type="asr")
    usage.record_media_call(unit=MediaUnit.IMAGES, quantity=4, call_type="edit_image")

    assert [d["quantity"] for d in usage.details] == [2.5, 4.0]


def test_requests_rejection_is_swallowed_by_the_wrapper() -> None:
    # Producers route through record_media_usage, whose contract is that an
    # accounting bug never breaks the media call the user asked for.
    from xagent.core.tools.core.media_usage import record_media_usage

    with TokenContextManager() as manager:
        record_media_usage(
            MediaUnit.REQUESTS, 3, model="m", call_type=MediaCallType.RERANK
        )
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


@pytest.mark.parametrize(
    "bad_tokens",
    [True, False, -100, float("inf"), float("-inf"), float("nan"), "abc"],
)
def test_malformed_provider_tokens_sanitize_without_losing_the_row(bad_tokens) -> None:
    # int(float("inf")) raises OverflowError, which was not caught — so a
    # malformed provider payload propagated out of the accounting path and
    # dropped the whole billable media row. Negatives and booleans were
    # persisted as-is, letting a provider bug subtract from a bill.
    usage = TokenUsage()
    usage.record_media_call(
        unit="images",
        quantity=1,
        call_type="generate_image",
        input_tokens=bad_tokens,
        output_tokens=bad_tokens,
    )

    # The row survives — the call happened and is billable by quantity ...
    assert usage.media_calls == 1
    entry = usage.details[0]
    assert entry["quantity"] == 1.0
    # ... with only the unusable token fields zeroed.
    assert entry["provider_input_tokens"] == 0
    assert entry["provider_output_tokens"] == 0
    assert entry["provider_tokens"] == 0


def test_valid_provider_tokens_survive() -> None:
    usage = TokenUsage()
    usage.record_media_call(
        unit="images",
        quantity=1,
        call_type="generate_image",
        input_tokens=11,
        output_tokens=5,
    )
    entry = usage.details[0]

    assert (entry["provider_input_tokens"], entry["provider_output_tokens"]) == (11, 5)
    assert entry["provider_tokens"] == 16


def test_snapshot_is_taken_under_the_lock() -> None:
    # An external copier (TaskTracker._copy_usage) reading fields one by one can
    # interleave with a concurrent write and see a counter without its matching
    # detail row. snapshot() takes the lock once so the pair always agrees.
    usage = TokenUsage()
    torn: list[tuple[int, int]] = []
    stop = threading.Event()

    def writer() -> None:
        for _ in range(400):
            usage.record_media_call(
                unit="images", quantity=1, call_type="generate_image"
            )
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            snap = usage.snapshot()
            media = sum(1 for d in snap.details if d.get("type") == "media")
            if snap.media_calls != media:
                torn.append((snap.media_calls, media))

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not torn, f"snapshot() returned torn state: {torn[:5]}"
    assert usage.media_calls == 400


def test_snapshot_detaches_from_the_source() -> None:
    usage = TokenUsage()
    usage.record_media_call(unit="images", quantity=1, call_type="generate_image")
    snap = usage.snapshot()

    usage.record_media_call(unit="images", quantity=1, call_type="generate_image")

    # The snapshot must not see writes that landed after it was taken.
    assert snap.media_calls == 1
    assert len(snap.details) == 1
    assert snap.details is not usage.details


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


def test_copy_copy_does_not_share_details_under_a_different_lock() -> None:
    # The default shallow copy routes through __getstate__/__setstate__, which
    # gives the new instance a fresh lock while still referencing the SAME
    # details list — two objects each believing they hold exclusive access,
    # mutating one list under two different locks. __copy__ routes to snapshot()
    # so copy.copy is a supported fork rather than a trap.
    import copy

    usage = TokenUsage()
    usage.record_media_call(unit="images", quantity=1, call_type="generate_image")
    clone = copy.copy(usage)

    assert clone.details is not usage.details
    assert clone._lock is not usage._lock

    clone.record_media_call(unit="images", quantity=1, call_type="generate_image")
    assert len(usage.details) == 1, "the original saw a write through its copy"
    assert len(clone.details) == 2
    assert usage.media_calls == 1


def test_deepcopy_also_routes_through_snapshot() -> None:
    import copy

    usage = TokenUsage()
    usage.record_media_call(unit="seconds", quantity=3, call_type="asr")
    clone = copy.deepcopy(usage)

    assert clone.details is not usage.details
    assert clone.media_calls == 1
    clone.record_media_call(unit="seconds", quantity=1, call_type="asr")
    assert usage.media_calls == 1


def test_to_dict_does_not_leak_the_live_detail_dicts() -> None:
    # to_dict takes the lock, but a new outer list around the same inner dicts
    # lets a caller mutate the live usage object through the returned payload,
    # entirely outside that lock.
    usage = TokenUsage()
    usage.record_media_call(unit="images", quantity=2, call_type="generate_image")

    payload = usage.to_dict()
    payload["details"][0]["quantity"] = 999
    payload["details"].append({"type": "media", "unit": "images"})

    assert usage.details[0]["quantity"] == 2.0
    assert len(usage.details) == 1


@pytest.mark.parametrize("negative", [-1, -5, -100])
def test_counters_ignore_negative_increments(negative) -> None:
    # A negative count would drive the counter below the number of matching
    # detail rows — the same counter/rows divergence the REQUESTS constraint
    # exists to prevent.
    usage = TokenUsage()
    usage.increment_tool_calls(negative)

    assert usage.tool_calls == 0

    # Positive increments still work.
    usage.increment_tool_calls(2)
    assert usage.tool_calls == 2


def test_merge_does_not_alias_the_source_detail_dicts() -> None:
    # Sharing the inner dicts would leave merged rows aliased to the source's,
    # so mutating one usage object silently rewrites the other's billing rows.
    # Found by self-review as the sibling of the to_dict leak.
    target = TokenUsage()
    source = TokenUsage()
    source.record_media_call(unit="images", quantity=1, call_type="generate_image")

    target.merge(source)
    target.details[0]["quantity"] = 999

    assert source.details[0]["quantity"] == 1.0
    assert target.media_calls == 1


def test_detail_tail_returns_only_the_tail_atomically() -> None:
    # The narrow read the delta path needs: copying the whole cumulative list to
    # use a small tail costs O(total usage) on a per-chunk quota path.
    usage = TokenUsage()
    for _ in range(5):
        usage.record_media_call(unit="images", quantity=1, call_type="generate_image")
    usage.increment_tool_calls(4)

    tail, tool_calls = usage.detail_tail(3)

    assert len(tail) == 2
    assert tool_calls == 4
    # Rows are detached, so a caller cannot mutate live state through them.
    tail[0]["quantity"] = 999
    assert usage.details[3]["quantity"] == 1.0

    # A start past the end is empty rather than an error.
    assert usage.detail_tail(99) == ([], 4)


def test_constructor_and_from_dict_detach_and_filter_details() -> None:
    # details is persisted as free-form JSON, so a legacy row may not be a dict;
    # and storing the caller's list by reference would let them mutate rows
    # outside the lock. Both are normalised at construction.
    caller_list = [{"type": "media", "unit": "images"}, "junk", None, 42]

    usage = TokenUsage(details=caller_list)  # type: ignore[arg-type]
    assert len(usage.details) == 1

    caller_list[0]["unit"] = "MUTATED"  # type: ignore[index]
    assert usage.details[0]["unit"] == "images"

    revived = TokenUsage.from_dict({"details": [{"type": "media"}, "bad"]})
    assert len(revived.details) == 1


def test_requests_error_reports_the_value_the_caller_passed() -> None:
    # quantity is coerced before the guard runs, so interpolating the coerced
    # value would report "got 0.0" for a caller who passed -1.
    usage = TokenUsage()
    with pytest.raises(ValueError, match=r"got -1"):
        usage.record_media_call(unit="requests", quantity=-1, call_type="rerank")


def test_lock_is_reentrant() -> None:
    # A plain Lock makes any future nested acquisition a self-deadlock, resting
    # on manual discipline; RLock removes the trap.
    usage = TokenUsage()
    with usage._lock:
        with usage._lock:
            usage.increment_tool_calls(1)
    assert usage.tool_calls == 1


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


def test_coerce_int_change_applies_to_the_llm_token_path() -> None:
    # _coerce_int is shared with add_token_usage, so its hardening (bool -> 0,
    # negatives -> 0) changes behaviour for every LLM adapter, not just media.
    with TokenContextManager() as manager:
        add_token_usage(
            input_tokens=True,  # type: ignore[arg-type]
            output_tokens=-5,
            model="m",
            call_type="chat",
        )
        usage = manager.get_usage()

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_media_aggregation_uses_model_id_when_present() -> None:
    # identity = model_id or model_name — the model_id branch had no coverage,
    # so rows keyed by id always fell through to the name branch in tests.
    with TokenContextManager() as manager:
        add_media_usage(
            unit="images",
            quantity=1,
            model="display-name",
            model_id="img-1",
            call_type="generate_image",
        )
        add_media_usage(
            unit="images",
            quantity=2,
            model="",
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
    usage.record_media_call(unit="images", quantity=huge, call_type="generate_image")

    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 0.0


def test_add_media_usage_reports_the_raw_quantity_in_the_requests_error() -> None:
    # The wrapper coerces one layer above record_media_call, so without
    # threading the raw value through, a caller passing -1 saw "got 0.0".
    with TokenContextManager():
        with pytest.raises(ValueError, match=r"got -1"):
            add_media_usage(unit="requests", quantity=-1, model="m", call_type="rerank")


@pytest.mark.parametrize(
    ("unit", "call_type"),
    [
        ("images", "video"),  # video bills in seconds
        ("seconds", "tts"),  # tts bills in characters
        ("images", "asr"),
        ("requests", "embedding"),  # embedding bills per text
    ],
)
def test_unit_must_match_the_modality(unit, call_type) -> None:
    # "The unit is a property of the modality" was stated in three docstrings and
    # violated by six call sites in this file. MEDIA_UNIT_BY_CALL_TYPE turns the
    # invariant into an enforced constraint: a (model, unit) price table is only
    # usable if one modality never reports two units.
    usage = TokenUsage()
    with pytest.raises(ValueError, match="bills in"):
        usage.record_media_call(unit=unit, quantity=1, call_type=call_type)

    assert usage.media_calls == 0
    assert usage.details == []


@pytest.mark.parametrize(
    ("call_type", "unit"),
    [
        ("generate_image", "images"),
        ("edit_image", "images"),
        ("video", "seconds"),
        ("asr", "seconds"),
        ("music", "seconds"),
        ("sound_effect", "seconds"),
        ("tts", "characters"),
        ("embedding", "texts"),
        ("rerank", "requests"),
    ],
)
def test_every_call_type_accepts_its_own_unit(call_type, unit) -> None:
    # The mapping must cover every MediaCallType member, or a legitimate
    # producer would be rejected.
    usage = TokenUsage()
    usage.record_media_call(unit=unit, quantity=1, call_type=call_type)

    assert usage.details[0]["unit"] == unit


def test_unit_is_unconstrained_when_call_type_is_omitted() -> None:
    # call_type is optional metadata; with none given there is no modality to
    # check the unit against.
    usage = TokenUsage()
    usage.record_media_call(unit="images", quantity=1)

    assert usage.details[0]["unit"] == "images"
