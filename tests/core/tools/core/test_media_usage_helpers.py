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
        record_media_seconds(30.0, model="veo", call_type="video")
        record_media_seconds(None, model="veo", call_type="video")
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
            record_media_seconds(None, model="veo", call_type="video")
    assert "unmeasured" in caplog.text


def test_record_media_usage_never_raises() -> None:
    # Accounting must never break the underlying media call.
    with TokenContextManager() as manager:
        record_media_usage("seconds", None, model="m", call_type="video")  # type: ignore[arg-type]
        assert manager.get_usage().media_calls == 1


def test_resolve_billing_model_never_returns_a_placeholder() -> None:
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
        mu.record_media_usage(
            MediaUnit.IMAGES, 1, model="m", call_type=MediaCallType.GENERATE_IMAGE
        )
        usage = manager.get_usage()

    # And must leave no partial state behind.
    assert usage.media_calls == 0
    assert usage.details == []


def test_record_media_usage_swallows_invalid_unit(monkeypatch) -> None:
    # A typo'd unit raises ValueError from the validator; the wrapper drops the
    # record with a warning rather than propagating into the media call.
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_usage("not-a-unit", 1, model="m", call_type="tts")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_record_media_seconds_swallows_invalid_call_type() -> None:
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_seconds(3.0, model="m", call_type="not-a-call-type")
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
            MediaUnit.IMAGES,
            1,
            model="m",
            call_type=MediaCallType.GENERATE_IMAGE,
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
