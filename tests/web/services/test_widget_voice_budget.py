"""Widget voice daily budgets and the fail-closed synthesis rate gate."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone

import pytest
from redis.exceptions import RedisError

from tests.shared.widget_voice_env import clear_widget_voice_env
from xagent.web.services import share_rate_limit, widget_voice_budget
from xagent.web.services.widget_voice_budget import (
    ASR_BUCKETS,
    TTS_BUCKETS,
    VoiceBudgetExhausted,
    VoiceBudgetUnavailable,
    refund,
    refund_sync,
    reserve_asr,
    reserve_asr_sync,
    reserve_tts_sync,
)
from xagent.web.services.writer_workers import active_workers

SETTINGS = {
    "tts-visitor": "XAGENT_WIDGET_VOICE_TTS_VISITOR_DAILY_CHARS",
    "tts-entity": "XAGENT_WIDGET_VOICE_TTS_ENTITY_DAILY_CHARS",
    "tts-site": "XAGENT_WIDGET_VOICE_TTS_SITE_DAILY_CHARS",
}


@pytest.fixture(autouse=True)
def memory_budget(monkeypatch):
    clear_widget_voice_env(monkeypatch)
    monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
    monkeypatch.setattr(
        widget_voice_budget, "_memory_store", widget_voice_budget._MemoryStore()
    )
    return widget_voice_budget


def _counts() -> dict[str, int]:
    counts = widget_voice_budget._memory_store._counts
    return {key.split(":")[3]: value for key, value in counts.items()}


async def test_cancelled_reserve_tts_settles_before_cancellation_propagates(
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_reserve(entity_key, visitor_key, chars):
        entered.set()
        release.wait(5)
        reserve_tts_sync(entity_key, visitor_key, chars)
        finished.set()

    monkeypatch.setattr(widget_voice_budget, "reserve_tts_sync", blocking_reserve)
    task = asyncio.ensure_future(widget_voice_budget.reserve_tts("agent:1", "v", 4))
    assert await asyncio.to_thread(entered.wait, 5)
    assert active_workers() == 1
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()  # still waiting for the reservation to settle
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
    assert active_workers() == 0
    assert _counts() == {"tts-visitor": 4, "tts-entity": 4, "tts-site": 4}


def test_tts_buckets_are_the_three_synthesis_layers():
    assert len(TTS_BUCKETS) == 3
    assert {bucket.name for bucket in TTS_BUCKETS} == set(SETTINGS)


@pytest.mark.parametrize("bucket", [b.name for b in TTS_BUCKETS])
def test_each_bucket_refuses_at_its_limit(monkeypatch, bucket):
    monkeypatch.setenv(SETTINGS[bucket], "10")
    reserve_tts_sync("agent:1", "1.2.3.4", 10)
    with pytest.raises(VoiceBudgetExhausted):
        reserve_tts_sync("agent:1", "1.2.3.4", 1)
    assert _counts()[bucket] == 10


@pytest.mark.parametrize(
    ("over", "rolled_back"),
    [("tts-site", ["tts-visitor", "tts-entity"]), ("tts-entity", ["tts-visitor"])],
)
def test_later_bucket_refusal_rolls_back_earlier_ones(monkeypatch, over, rolled_back):
    monkeypatch.setenv(SETTINGS[over], "5")
    reserve_tts_sync("agent:1", "1.2.3.4", 5)
    before = _counts()
    with pytest.raises(VoiceBudgetExhausted):
        reserve_tts_sync("agent:1", "1.2.3.4", 3)
    after = _counts()
    for bucket in [*rolled_back, over]:
        assert after[bucket] == before[bucket]


def test_zero_site_budget_refuses_everything(monkeypatch):
    monkeypatch.setenv(SETTINGS["tts-site"], "0")
    with pytest.raises(VoiceBudgetExhausted):
        reserve_tts_sync("agent:1", "v", 1)


def _unset(monkeypatch, settings: dict[str, str]) -> None:
    for name in settings.values():
        monkeypatch.delenv(name, raising=False)


def test_unset_visitor_and_entity_budgets_have_no_limit(monkeypatch):
    _unset(monkeypatch, SETTINGS)
    # One visitor reserves 250,000 characters in a day.
    for _ in range(5):
        reserve_tts_sync("agent:1", "1.2.3.4", 50_000)
    assert _counts() == {
        "tts-visitor": 250_000,
        "tts-entity": 250_000,
        "tts-site": 250_000,
    }


def test_unset_site_budget_refuses_at_a_million_characters(monkeypatch):
    _unset(monkeypatch, SETTINGS)
    for visitor in range(4):
        reserve_tts_sync("agent:1", f"v{visitor}", 250_000)
    before = dict(widget_voice_budget._memory_store._counts)
    with pytest.raises(VoiceBudgetExhausted) as refused:
        reserve_tts_sync("agent:1", "v0", 1)
    assert str(refused.value) == "tts-site"
    assert widget_voice_budget._memory_store._counts == before
    assert _counts()["tts-site"] == 1_000_000


@pytest.mark.parametrize("bucket", ["tts-visitor", "tts-entity"])
def test_zero_visitor_or_entity_budget_refuses_everything(monkeypatch, bucket):
    _unset(monkeypatch, SETTINGS)
    monkeypatch.setenv(SETTINGS[bucket], "0")
    with pytest.raises(VoiceBudgetExhausted) as refused:
        reserve_tts_sync("agent:1", "v", 1)
    assert str(refused.value) == bucket
    assert set(_counts().values()) <= {0}


@pytest.mark.parametrize("next_day", [False, True])
def test_budget_window_is_the_utc_calendar_day(monkeypatch, next_day):
    monkeypatch.setenv(SETTINGS["tts-entity"], "10")
    day = {"value": "20261114"}
    monkeypatch.setattr(widget_voice_budget, "_utc_day", lambda: day["value"])
    reserve_tts_sync("agent:1", "v", 8)
    if next_day:
        day["value"] = "20261115"
        reserve_tts_sync("agent:1", "v", 8)
        counts = widget_voice_budget._memory_store._counts
        assert counts["xagent:widget-voice:20261115:tts-entity:agent:1"] == 8
    else:
        with pytest.raises(VoiceBudgetExhausted):
            reserve_tts_sync("agent:1", "v", 8)


def test_late_prior_day_reservation_keeps_the_new_day_counts(monkeypatch):
    monkeypatch.setenv(SETTINGS["tts-site"], "10")
    day = {"value": "20261117"}
    monkeypatch.setattr(widget_voice_budget, "_utc_day", lambda: day["value"])
    reserve_tts_sync("agent:1", "v", 10)
    day["value"] = "20261116"
    reserve_tts_sync("agent:1", "v", 5)
    day["value"] = "20261117"
    with pytest.raises(VoiceBudgetExhausted):
        reserve_tts_sync("agent:1", "v", 1)
    counts = widget_voice_budget._memory_store._counts
    assert counts["xagent:widget-voice:20261117:tts-site:all"] == 10
    assert counts["xagent:widget-voice:20261116:tts-site:all"] == 5


def test_memory_budget_drops_counters_two_days_old(monkeypatch):
    _unset(monkeypatch, SETTINGS)
    day = {"value": "20261130"}
    monkeypatch.setattr(widget_voice_budget, "_utc_day", lambda: day["value"])
    reserve_tts_sync("agent:1", "v", 3)
    day["value"] = "20261201"
    reserve_tts_sync("agent:1", "v", 3)
    keys = widget_voice_budget._memory_store._counts
    assert "xagent:widget-voice:20261130:tts-site:all" in keys
    assert "xagent:widget-voice:20261201:tts-site:all" in keys
    day["value"] = "20261202"
    reserve_tts_sync("agent:1", "v", 3)
    assert not [k for k in keys if ":20261130:" in k]
    assert "xagent:widget-voice:20261201:tts-site:all" in keys
    assert "xagent:widget-voice:20261202:tts-site:all" in keys


def test_budget_day_is_taken_in_utc(monkeypatch):
    class _Clock:
        @staticmethod
        def now(tz):
            assert tz is timezone.utc
            return datetime(2026, 11, 14, 23, 59, tzinfo=timezone.utc)

    monkeypatch.setattr(widget_voice_budget, "datetime", _Clock)
    assert widget_voice_budget._utc_day() == "20261114"


class _FakeRedis:
    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.values: dict[str, int] = {}
        self._ops: list[tuple[str, int]] = []

    def pipeline(self):
        self._ops = []
        return self

    def incrby(self, key, amount):
        self._ops.append((key, amount))

    def expire(self, key, seconds):
        assert seconds == 48 * 3600

    def execute(self):
        if self.fail == "increment":
            raise RedisError("down")
        key, amount = self._ops[0]
        self.values[key] = self.values.get(key, 0) + amount
        return [self.values[key], True]

    def decrby(self, key, amount):
        if self.fail == "rollback":
            raise RedisError("down")
        self.values[key] -= amount


@pytest.mark.parametrize("failure", ["init", "increment", "rollback", None])
def test_configured_redis_never_falls_back_to_memory(monkeypatch, failure):
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://budget.invalid:6379/0")
    fake = _FakeRedis(failure)

    def client(_url):
        if failure == "init":
            raise ValueError("bad url")
        return fake

    monkeypatch.setattr(widget_voice_budget, "_redis_client", client)
    if failure == "rollback":
        monkeypatch.setenv(SETTINGS["tts-site"], "0")
        with pytest.raises(VoiceBudgetExhausted):
            reserve_tts_sync("agent:1", "v", 4)
    elif failure is not None:
        with pytest.raises(VoiceBudgetUnavailable):
            reserve_tts_sync("agent:1", "v", 4)
    else:
        reserve_tts_sync("agent:1", "v", 4)
        assert sorted(fake.values.values()) == [4, 4, 4]
    assert widget_voice_budget._memory_store._counts == {}


@pytest.mark.parametrize(
    ("setting", "attribute", "default"),
    [
        (
            "XAGENT_WIDGET_VOICE_TTS_IP_RATE_LIMIT",
            "_widget_voice_tts_ip_limit",
            "120/minute",
        ),
        (
            "XAGENT_WIDGET_VOICE_TTS_RATE_LIMIT",
            "_widget_voice_tts_entity_limit",
            "600/minute",
        ),
    ],
)
def test_unparsable_synthesis_rate_limit_uses_its_default(
    monkeypatch, setting, attribute, default
):
    from limits import parse

    monkeypatch.setenv(setting, "often")
    share_rate_limit.reset_share_rate_limiter()
    limiter = share_rate_limit.get_share_rate_limiter()
    assert getattr(limiter, attribute) == parse(default)
    share_rate_limit.reset_share_rate_limiter()


@pytest.mark.parametrize("phase", ["test", "hit"])
def test_widget_voice_rate_limit_fails_closed(monkeypatch, phase):
    share_rate_limit.reset_share_rate_limiter()
    limiter = share_rate_limit.get_share_rate_limiter()

    def broken(*_args):
        raise RedisError("storage down")

    monkeypatch.setattr(limiter._limiter, phase, broken)
    assert limiter.allow_widget_voice_tts("agent:1", "1.2.3.4") is False
    share_rate_limit.reset_share_rate_limiter()


def test_widget_voice_rate_limit_pairs_ip_and_entity(monkeypatch):
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_IP_RATE_LIMIT", "1/minute")
    share_rate_limit.reset_share_rate_limiter()
    limiter = share_rate_limit.get_share_rate_limiter()
    assert limiter.allow_widget_voice_tts("agent:1", "1.2.3.4") is True
    assert limiter.allow_widget_voice_tts("agent:1", "1.2.3.4") is False
    assert limiter.allow_widget_voice_tts("agent:1", "5.6.7.8") is True
    share_rate_limit.reset_share_rate_limiter()


ASR_SETTINGS = {
    "asr-visitor": "XAGENT_WIDGET_VOICE_ASR_VISITOR_DAILY_SECONDS",
    "asr-entity": "XAGENT_WIDGET_VOICE_ASR_ENTITY_DAILY_SECONDS",
    "asr-site": "XAGENT_WIDGET_VOICE_ASR_SITE_DAILY_SECONDS",
}


def test_budget_buckets_are_three_layers_of_synthesis_and_transcription():
    assert len(TTS_BUCKETS + ASR_BUCKETS) == 6
    assert {bucket.name for bucket in ASR_BUCKETS} == set(ASR_SETTINGS)
    assert [bucket.limit() for bucket in ASR_BUCKETS] == [None, None, 86_400]


@pytest.mark.parametrize("bucket", [b.name for b in ASR_BUCKETS])
def test_each_transcription_bucket_refuses_at_its_limit(monkeypatch, bucket):
    monkeypatch.setenv(ASR_SETTINGS[bucket], "1200")
    reserve_asr_sync("agent:1", "1.2.3.4", 600)
    reserve_asr_sync("agent:1", "1.2.3.4", 600)
    with pytest.raises(VoiceBudgetExhausted):
        reserve_asr_sync("agent:1", "1.2.3.4", 600)
    assert _counts()[bucket] == 1200


@pytest.mark.parametrize(
    ("over", "rolled_back"),
    [("asr-site", ["asr-visitor", "asr-entity"]), ("asr-entity", ["asr-visitor"])],
)
def test_later_transcription_bucket_refusal_rolls_back(monkeypatch, over, rolled_back):
    monkeypatch.setenv(ASR_SETTINGS[over], "700")
    reserve_asr_sync("agent:1", "1.2.3.4", 600)
    before = _counts()
    with pytest.raises(VoiceBudgetExhausted):
        reserve_asr_sync("agent:1", "1.2.3.4", 600)
    after = _counts()
    for bucket in [*rolled_back, over]:
        assert after[bucket] == before[bucket]


def test_unset_visitor_and_entity_transcription_budgets_have_no_limit(monkeypatch):
    _unset(monkeypatch, ASR_SETTINGS)
    # One visitor reserves 15,000 seconds in a day.
    for _ in range(25):
        reserve_asr_sync("agent:1", "1.2.3.4", 600)
    assert _counts() == {
        "asr-visitor": 15_000,
        "asr-entity": 15_000,
        "asr-site": 15_000,
    }


def test_unset_site_transcription_budget_refuses_at_a_day_of_seconds(monkeypatch):
    _unset(monkeypatch, ASR_SETTINGS)
    for session in range(144):
        reserve_asr_sync("agent:1", f"v{session % 4}", 600)
    before = dict(widget_voice_budget._memory_store._counts)
    with pytest.raises(VoiceBudgetExhausted) as refused:
        reserve_asr_sync("agent:1", "v0", 1)
    assert str(refused.value) == "asr-site"
    assert widget_voice_budget._memory_store._counts == before
    assert _counts()["asr-site"] == 86_400


@pytest.mark.parametrize("bucket", ["asr-visitor", "asr-entity"])
def test_zero_visitor_or_entity_transcription_budget_refuses_everything(
    monkeypatch, bucket
):
    _unset(monkeypatch, ASR_SETTINGS)
    monkeypatch.setenv(ASR_SETTINGS[bucket], "0")
    with pytest.raises(VoiceBudgetExhausted) as refused:
        reserve_asr_sync("agent:1", "v", 1)
    assert str(refused.value) == bucket
    assert set(_counts().values()) <= {0}


@pytest.mark.parametrize("raw", ["", "  ", "-5", "lots"])
@pytest.mark.parametrize("index", [0, 1], ids=["asr-visitor", "asr-entity"])
def test_blank_or_bad_visitor_and_entity_transcription_budgets_mean_no_limit(
    monkeypatch, caplog, index, raw
):
    bucket = ASR_BUCKETS[index]
    monkeypatch.setenv(ASR_SETTINGS[bucket.name], raw)
    assert bucket.limit() is None
    warned = f"Invalid {ASR_SETTINGS[bucket.name]}={raw!r}" in caplog.text
    assert warned == bool(raw.strip())


@pytest.mark.parametrize("next_day", [False, True])
def test_transcription_budget_window_is_the_utc_calendar_day(monkeypatch, next_day):
    monkeypatch.setenv(ASR_SETTINGS["asr-entity"], "1000")
    day = {"value": "20261114"}
    monkeypatch.setattr(widget_voice_budget, "_utc_day", lambda: day["value"])
    reserve_asr_sync("agent:1", "v", 600)
    if next_day:
        day["value"] = "20261115"
        reserve_asr_sync("agent:1", "v", 600)
        assert _counts()["asr-entity"] == 600
    else:
        with pytest.raises(VoiceBudgetExhausted):
            reserve_asr_sync("agent:1", "v", 600)


def test_refund_gives_back_on_the_keys_it_reserved_under(monkeypatch):
    day = {"value": "20261114"}
    monkeypatch.setattr(widget_voice_budget, "_utc_day", lambda: day["value"])
    reservation = reserve_asr_sync("agent:1", "v", 600)
    day["value"] = "20261115"
    refund_sync(reservation, 540)
    counts = widget_voice_budget._memory_store._counts
    assert sorted(counts.values()) == [60, 60, 60]
    assert all(":20261114:" in key for key in counts)
    # A refund never gives back more than was reserved, nor a negative amount.
    refund_sync(reservation, 10_000)
    refund_sync(reservation, -5)
    assert sorted(counts.values()) == [-540, -540, -540]


def test_refund_after_midnight_lands_on_the_key_it_reserved_under(monkeypatch):
    day = {"value": "20261114"}
    monkeypatch.setattr(widget_voice_budget, "_utc_day", lambda: day["value"])
    reservation = reserve_asr_sync("agent:1", "v", 600)
    day["value"] = "20261115"
    # A later reservation on the new day does not drop the previous day.
    reserve_asr_sync("agent:1", "w", 100)
    refund_sync(reservation, 540)
    counts = widget_voice_budget._memory_store._counts
    previous = {k: v for k, v in counts.items() if ":20261114:" in k}
    assert sorted(previous.values()) == [60, 60, 60]
    assert sum(1 for k in counts if ":20261115:" in k) == 3


def test_refund_storage_failure_only_logs(monkeypatch, caplog):
    reservation = reserve_asr_sync("agent:1", "v", 600)

    def broken(*_args):
        raise ConnectionError("storage down")

    monkeypatch.setattr(widget_voice_budget._memory_store, "decrby", broken)
    refund_sync(reservation, 600)
    assert sorted(_counts().values()) == [600, 600, 600]
    assert "Widget voice budget refund failed" in caplog.text


async def test_cancelled_transcription_reservation_is_given_back(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = widget_voice_budget.reserve_asr_sync

    def slow(*args):
        entered.set()
        release.wait(2)
        return real(*args)

    monkeypatch.setattr(widget_voice_budget, "reserve_asr_sync", slow)
    task = asyncio.create_task(reserve_asr("agent:1", "v", 600))
    await asyncio.to_thread(entered.wait, 2)
    # The reservation runs as a tracked writer, counted until its thread ends.
    assert active_workers() == 1
    task.cancel()
    await asyncio.sleep(0.01)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert active_workers() == 0
    assert sorted(_counts().values()) == [0, 0, 0]


async def test_cancelled_refund_settles_before_cancellation_propagates(monkeypatch):
    reservation = await reserve_asr("agent:1", "v", 600)
    entered, release = threading.Event(), threading.Event()
    real = widget_voice_budget.refund_sync

    def slow(*args):
        entered.set()
        release.wait(2)
        real(*args)

    monkeypatch.setattr(widget_voice_budget, "refund_sync", slow)
    task = asyncio.create_task(refund(reservation, 590))
    await asyncio.to_thread(entered.wait, 2)
    assert active_workers() == 1
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()  # still waiting for the refund to land
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert active_workers() == 0
    assert sorted(_counts().values()) == [10, 10, 10]


async def test_async_refund_settles_the_reservation():
    reservation = await reserve_asr("agent:1", "v", 600)
    await refund(reservation, 590)
    assert sorted(_counts().values()) == [10, 10, 10]


@pytest.mark.parametrize(
    "method", ["allow_widget_voice_connect_ip", "allow_widget_voice_connect_entity"]
)
def test_widget_voice_connect_gates_fail_closed(monkeypatch, method):
    share_rate_limit.reset_share_rate_limiter()
    limiter = share_rate_limit.get_share_rate_limiter()

    def broken(*_args):
        raise RedisError("storage down")

    monkeypatch.setattr(limiter._limiter, "hit", broken)
    assert getattr(limiter, method)("agent:1") is False
    share_rate_limit.reset_share_rate_limiter()


def test_widget_voice_connect_gates_count_ip_and_entity_apart(monkeypatch):
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_CONNECT_IP_RATE_LIMIT", "1/minute")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_CONNECT_RATE_LIMIT", "2/minute")
    share_rate_limit.reset_share_rate_limiter()
    limiter = share_rate_limit.get_share_rate_limiter()
    assert limiter.allow_widget_voice_connect_ip("1.2.3.4") is True
    assert limiter.allow_widget_voice_connect_ip("1.2.3.4") is False
    assert limiter.allow_widget_voice_connect_ip("5.6.7.8") is True
    assert limiter.allow_widget_voice_connect_entity("agent:1") is True
    assert limiter.allow_widget_voice_connect_entity("agent:1") is True
    assert limiter.allow_widget_voice_connect_entity("agent:1") is False
    assert limiter.allow_widget_voice_connect_entity("agent:2") is True
    share_rate_limit.reset_share_rate_limiter()


def test_unset_connect_ip_gate_admits_120_attempts_a_minute(monkeypatch):
    monkeypatch.delenv("XAGENT_WIDGET_VOICE_CONNECT_IP_RATE_LIMIT", raising=False)
    share_rate_limit.reset_share_rate_limiter()
    limiter = share_rate_limit.get_share_rate_limiter()
    assert all(limiter.allow_widget_voice_connect_ip("1.2.3.4") for _ in range(120))
    assert limiter.allow_widget_voice_connect_ip("1.2.3.4") is False
    share_rate_limit.reset_share_rate_limiter()


@pytest.mark.parametrize(
    ("setting", "attribute", "default"),
    [
        (
            "XAGENT_WIDGET_VOICE_CONNECT_IP_RATE_LIMIT",
            "_widget_voice_connect_ip_limit",
            "120/minute",
        ),
        (
            "XAGENT_WIDGET_VOICE_CONNECT_RATE_LIMIT",
            "_widget_voice_connect_entity_limit",
            "120/minute",
        ),
    ],
)
def test_unparsable_connect_rate_limit_uses_its_default(
    monkeypatch, setting, attribute, default
):
    from limits import parse

    monkeypatch.setenv(setting, "often")
    share_rate_limit.reset_share_rate_limiter()
    limiter = share_rate_limit.get_share_rate_limiter()
    assert getattr(limiter, attribute) == parse(default)
    share_rate_limit.reset_share_rate_limiter()


ALL_BUDGET_SETTINGS = {**SETTINGS, **ASR_SETTINGS}


DEFAULT_LIMITS = {
    "tts-visitor": None,
    "tts-entity": None,
    "tts-site": 1_000_000,
    "asr-visitor": None,
    "asr-entity": None,
    "asr-site": 86_400,
}


@pytest.mark.parametrize("raw", ["", "  "])
def test_blank_budget_settings_use_their_defaults(monkeypatch, raw):
    # A deployment that leaves a variable empty passes an empty string.
    for setting in ALL_BUDGET_SETTINGS.values():
        monkeypatch.setenv(setting, raw)
    buckets = TTS_BUCKETS + ASR_BUCKETS
    assert len(ALL_BUDGET_SETTINGS) == len(buckets) == 6
    assert {b.name: b.limit() for b in buckets} == DEFAULT_LIMITS


def test_budget_settings_ignore_surrounding_whitespace(monkeypatch):
    for setting in ALL_BUDGET_SETTINGS.values():
        monkeypatch.setenv(setting, " 7 ")
    assert {b.limit() for b in TTS_BUCKETS + ASR_BUCKETS} == {7}


VOICE_RATE_LIMITS = {
    "XAGENT_WIDGET_VOICE_TTS_IP_RATE_LIMIT": (
        "_widget_voice_tts_ip_limit",
        "120/minute",
    ),
    "XAGENT_WIDGET_VOICE_TTS_RATE_LIMIT": (
        "_widget_voice_tts_entity_limit",
        "600/minute",
    ),
    "XAGENT_WIDGET_VOICE_CONNECT_IP_RATE_LIMIT": (
        "_widget_voice_connect_ip_limit",
        "120/minute",
    ),
    "XAGENT_WIDGET_VOICE_CONNECT_RATE_LIMIT": (
        "_widget_voice_connect_entity_limit",
        "120/minute",
    ),
}


@pytest.mark.parametrize("raw", ["", "  "])
def test_blank_voice_rate_limits_use_their_defaults(monkeypatch, raw):
    from limits import parse

    for setting in VOICE_RATE_LIMITS:
        monkeypatch.setenv(setting, raw)
    share_rate_limit.reset_share_rate_limiter()
    limiter = share_rate_limit.get_share_rate_limiter()
    for attribute, default in VOICE_RATE_LIMITS.values():
        assert getattr(limiter, attribute) == parse(default)
    share_rate_limit.reset_share_rate_limiter()


def test_voice_rate_limits_ignore_surrounding_whitespace(monkeypatch):
    from limits import parse

    for setting in VOICE_RATE_LIMITS:
        monkeypatch.setenv(setting, " 3/minute ")
    share_rate_limit.reset_share_rate_limiter()
    limiter = share_rate_limit.get_share_rate_limiter()
    for attribute, _default in VOICE_RATE_LIMITS.values():
        assert getattr(limiter, attribute) == parse("3/minute")
    share_rate_limit.reset_share_rate_limiter()
