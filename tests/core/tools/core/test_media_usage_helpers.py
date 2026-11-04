"""Unit stability for duration-billed media tools.

The billed unit must depend on the modality, never on how complete the
provider's response happened to be — a price table keyed on (model, unit) is
unusable if the same model sometimes reports "seconds" and sometimes
"requests".
"""

import pytest

from xagent.core.model.chat.token_context import (
    MediaCallType,
    MediaUnit,
    TokenContextManager,
    aggregate_media_usage_by_model,
)
from xagent.core.tools.core.media_usage import (
    coerce_duration,
    record_media_seconds,
    record_media_usage,
    resolve_billing_model,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (12.5, 12.5),
        ("3", 3.0),
        (0, None),  # zero is "no usable duration", not a real measurement
        (-5, None),
        (None, None),
        (True, None),  # bool must not sneak through as 1.0
        ("nonsense", None),
    ],
)
def test_coerce_duration(value, expected) -> None:
    assert coerce_duration(value) == expected


def test_record_media_seconds_keeps_unit_stable_when_duration_missing() -> None:
    # Same model, one call with a duration and one without: both must report
    # seconds so billing sees a single line item, not two different units.
    with TokenContextManager() as manager:
        record_media_seconds(30.0, call_type="video", model="veo")
        record_media_seconds(None, call_type="video", model="veo")
        details = manager.get_usage().details

    assert [entry["unit"] for entry in details] == ["seconds", "seconds"]
    # The unmeasured call records 0 seconds and is deliberately KEPT in the
    # rollup: it is the only evidence that a billable provider call happened.
    # It contributes nothing to quantity but still counts toward calls.
    assert [entry["quantity"] for entry in details] == [30.0, 0.0]
    groups = aggregate_media_usage_by_model(details)
    assert len(groups) == 1
    assert groups[0]["unit"] == "seconds"
    assert groups[0]["quantity"] == 30.0
    assert groups[0]["calls"] == 2


def test_record_media_seconds_warns_when_unmeasured(caplog) -> None:
    with caplog.at_level("WARNING"):
        with TokenContextManager():
            record_media_seconds(None, call_type="video", model="veo")
    assert "unmeasured" in caplog.text


def test_record_media_usage_never_raises() -> None:
    # Accounting must never break the underlying media call.
    with TokenContextManager() as manager:
        record_media_usage("video", None, model="m")  # type: ignore[arg-type]
        assert manager.get_usage().media_calls == 1


def test_resolve_billing_model_prefers_real_identities_over_the_fallback() -> None:
    """`_configured_model_id`-style lookups return Optional[str]; passing that
    through str() records a model literally named "None"."""

    class _Model:
        model_name = "elevenlabs-music-v1"

    # None id -> falls back to the provider's own name, not "None".
    assert resolve_billing_model(None, _Model()) == "elevenlabs-music-v1"
    assert resolve_billing_model("", _Model()) == "elevenlabs-music-v1"
    # A real configured id always wins.
    assert resolve_billing_model("cfg-id", _Model()) == "cfg-id"
    # Nothing identifies the model: an explicit fallback, never None/"None".
    assert resolve_billing_model(None, None) == "default"

    # Placeholder names on the model are not treated as identities.
    class _Placeholder:
        model_name = "None"

    assert resolve_billing_model(None, _Placeholder()) == "default"


def test_record_media_usage_swallows_recording_errors(monkeypatch) -> None:
    # The best-effort guarantee is the whole point of this wrapper: a metering
    # bug must never break the media call the user asked for. The existing
    # quantity=None case coerces to 0 and records fine, so it would still pass
    # with the try/except deleted — this one would not.
    import xagent.core.tools.core.media_usage as mu

    def _boom(**kwargs: object) -> None:
        raise RuntimeError("recording backend exploded")

    monkeypatch.setattr(mu, "add_media_usage", _boom)

    with TokenContextManager() as manager:
        # Must not raise.
        mu.record_media_usage(MediaCallType.GENERATE_IMAGE, 1, model="m")
        usage = manager.get_usage()

    # And must leave no partial state behind.
    assert usage.media_calls == 0
    assert usage.details == []


def test_record_media_usage_swallows_invalid_call_type() -> None:
    # A typo'd call_type raises ValueError from the validator; the wrapper drops
    # the record with a warning rather than propagating into the media call.
    # Note "tts" is now a *valid* call_type -- the unit it used to be checked
    # against no longer exists -- so the invalid value has to be a real typo.
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_usage("ttts", 1, model="m")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_record_media_seconds_swallows_invalid_call_type() -> None:
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_seconds(3.0, call_type="not-a-call-type", model="m")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_record_media_seconds_ignores_non_finite_duration() -> None:
    # inf > 0 is True, so without the finiteness guard this would record a
    # non-JSON-serialisable quantity.
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_seconds(
            mu.coerce_duration(float("inf")), model="m", call_type=MediaCallType.ASR
        )
        entry = manager.get_usage().details[0]

    # Treated as "no duration reported": recorded as 0 seconds, unit unchanged.
    assert entry["unit"] == "seconds"
    assert entry["quantity"] == 0.0


def test_record_media_usage_forwards_optional_billing_metadata() -> None:
    # The wrapper must mirror the core primitive's optional fields: passing them
    # previously raised TypeError during argument binding, before the try block.
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_usage(
            MediaCallType.GENERATE_IMAGE,
            1,
            model="m",
            resolution="2K",
            input_tokens=7,
            output_tokens=3,
            tokens_estimated=True,
        )
        entry = manager.get_usage().details[0]

    assert entry["resolution"] == "2K"
    assert entry["provider_tokens"] == 10
    assert entry["tokens_estimated"] is True


def test_resolve_billing_model_survives_raising_descriptor() -> None:
    # Identity resolution runs before record_media_usage's try/except, so a
    # provider whose model_name property raises would otherwise break the call.
    import xagent.core.tools.core.media_usage as mu

    class _Hostile:
        @property
        def model_name(self) -> str:
            raise RuntimeError("descriptor exploded")

        @property
        def model(self) -> str:
            return "real-name"

    assert mu.resolve_billing_model(None, _Hostile()) == "real-name"


@pytest.mark.parametrize(
    "call_type",
    [c for c in MediaCallType if c.unit is not MediaUnit.SECONDS],
)
def test_record_media_seconds_drops_non_duration_modalities(call_type, caplog) -> None:
    # Handing this a modality billed in images or characters used to record a
    # row whose quantity was a duration but whose unit said otherwise -- a
    # silently mispriced row. Dropped and logged instead.
    #
    # Deliberately not raised: producers call this from inside their own
    # `try/except Exception` (music_tool), whose handler returns
    # `success: False`, so raising would turn a generated-and-billed media call
    # into a reported failure.
    import xagent.core.tools.core.media_usage as mu

    with caplog.at_level("ERROR"):
        with TokenContextManager() as manager:
            mu.record_media_seconds(1.5, model="m", call_type=call_type)
            assert manager.get_usage().details == []

    assert "record_media_seconds" in caplog.text
    assert call_type.value in caplog.text


@pytest.mark.parametrize(
    "call_type", [c for c in MediaCallType if c.unit is MediaUnit.SECONDS]
)
def test_record_media_seconds_accepts_every_duration_modality(call_type) -> None:
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_seconds(1.5, model="m", call_type=call_type)
        details = manager.get_usage().details

    assert len(details) == 1
    assert details[0]["unit"] == "seconds"
    assert details[0]["quantity"] == 1.5


@pytest.mark.parametrize(
    "overflowing",
    [
        # float() of a too-large int raises OverflowError, not ValueError. A
        # 400-digit integer literal is exactly what json.loads yields for an
        # oversized JSON number in a provider response.
        10**400,
        -(10**400),
        # These convert fine and are caught by the isfinite guard instead.
        float("inf"),
        float("-inf"),
        float("nan"),
        "1e400",
    ],
)
def test_coerce_duration_survives_non_finite_values(overflowing) -> None:
    # int(float("inf")) raises OverflowError, not ValueError; an ASR provider
    # reporting a non-finite duration must degrade to an unmeasured row rather
    # than break the transcription call it is measuring.
    import xagent.core.tools.core.media_usage as mu

    # None, not 0.0: the helper's documented contract distinguishes "provider
    # reported no duration" from "provider reported zero seconds".
    assert mu.coerce_duration(overflowing) is None

    with TokenContextManager() as manager:
        mu.record_media_seconds(
            mu.coerce_duration(overflowing), model="m", call_type=MediaCallType.ASR
        )
        details = manager.get_usage().details

    assert len(details) == 1
    assert details[0]["quantity"] == 0.0


@pytest.mark.parametrize("empty", [None, ""])
def test_record_media_seconds_leaves_empty_call_types_to_the_primitive(empty) -> None:
    # An absent call_type has no modality to check the unit against, so the
    # duration guard must not claim a mismatch. Validation belongs to
    # add_media_usage, which produces the message listing valid options -- and
    # the wrapper swallows it so a media call is never broken by accounting.
    import xagent.core.tools.core.media_usage as mu

    assert mu._resolved_call_type(empty) is None

    with TokenContextManager() as manager:
        mu.record_media_seconds(1.5, model="m", call_type=empty)

        # Swallowed, not raised, and no half-written row left behind.
        assert manager.get_usage().details == []
