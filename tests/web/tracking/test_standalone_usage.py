"""The usage sink for work that has no TaskTracker.

Recording usage is only half of metering: something has to bind a TokenUsage
and hand it to the quota hook. Entry points such as ``/speech/transcribe`` have
no TaskTracker, so without this their recorded usage lands in a throwaway
object.
"""

import asyncio
from typing import Any, Optional

import pytest

from xagent.core.model.chat.token_context import (
    MediaCallType,
    add_media_usage,
    get_token_usage,
)
from xagent.web.tracking.standalone_usage import usage_scope


def _async_return(value: Any):
    """An async callable ignoring its args and returning ``value``."""

    async def _fn(*_a: Any, **_k: Any) -> Any:
        return value

    return _fn


class _FakeSession:
    """Stand-in for the short-lived compatibility Session ``_report`` opens.

    ``in_transaction`` is configurable and the calls are recorded in order:
    with it hard-coded to False the rollback branch is unreachable, so
    deleting the rollback from _report would leave every test green.
    """

    def __init__(self, in_transaction: bool = False) -> None:
        self._in_transaction = in_transaction
        self.rolled_back = False
        self.closed = False
        self.calls: list[str] = []

    def in_transaction(self) -> bool:
        return self._in_transaction

    def rollback(self) -> None:
        self.rolled_back = True
        self._in_transaction = False
        self.calls.append("rollback")

    def close(self) -> None:
        self.closed = True
        self.calls.append("close")


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch):
    """Capture what reaches quota_hooks.record_usage."""
    calls: list[dict[str, Any]] = []
    session = _FakeSession()

    def _record(db: Any, user_id: Any, details: list, actions: int) -> None:
        calls.append(
            {"db": db, "user_id": user_id, "details": details, "actions": actions}
        )

    from xagent.web.models import database
    from xagent.web.services import quota_hooks

    monkeypatch.setattr(quota_hooks, "record_usage", _record)
    # _report short-circuits before touching a session when no hook is
    # installed, so the stock configuration would otherwise report nothing.
    monkeypatch.setattr(quota_hooks, "has_usage_record_hook", lambda: True)
    monkeypatch.setattr(database, "get_session_local", lambda: lambda: session)
    return calls, session


def test_usage_recorded_in_scope_reaches_the_quota_hook(captured) -> None:
    calls, _ = captured
    with usage_scope(42):
        add_media_usage(MediaCallType.ASR, 3, model="asr-1")

    assert len(calls) == 1
    assert calls[0]["user_id"] == 42
    assert calls[0]["details"][0]["call_type"] == "asr"
    # The unit is derived from the call type, never passed in.
    assert calls[0]["details"][0]["unit"] == "seconds"
    assert calls[0]["details"][0]["quantity"] == 3.0
    # These paths make provider calls, not agent tool calls.
    assert calls[0]["actions"] == 0


def test_scope_reports_even_when_the_body_raises(captured) -> None:
    """A provider call that already happened is billable regardless of what
    fails afterwards."""
    calls, _ = captured
    with pytest.raises(RuntimeError):
        with usage_scope(7):
            add_media_usage(MediaCallType.TTS, 12, model="t")
            raise RuntimeError("downstream failure")

    assert len(calls) == 1
    assert calls[0]["details"][0]["call_type"] == "tts"
    assert calls[0]["details"][0]["unit"] == "characters"


def test_no_usage_means_no_hook_call(captured) -> None:
    calls, _ = captured
    with usage_scope(1):
        pass
    assert calls == []


def test_compatibility_session_is_disposed(captured) -> None:
    """The hook manages its own durability; the session handed to it must be
    closed by us and never left holding a transaction."""
    calls, session = captured
    with usage_scope(5):
        add_media_usage(MediaCallType.ASR, 2, model="e")

    assert session.closed is True
    assert calls[0]["db"] is session


def test_active_transaction_is_rolled_back_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hook owns its own durability and must not be left holding one.

    With in_transaction() hard-coded False the rollback branch is unreachable,
    so removing the rollback from _report would leave the suite green.
    """
    from xagent.web.models import database
    from xagent.web.services import quota_hooks

    session = _FakeSession(in_transaction=True)
    monkeypatch.setattr(quota_hooks, "record_usage", lambda *a, **k: None)
    monkeypatch.setattr(quota_hooks, "has_usage_record_hook", lambda: True)
    monkeypatch.setattr(database, "get_session_local", lambda: lambda: session)

    with usage_scope(1):
        add_media_usage(MediaCallType.ASR, 1, model="m")

    assert session.rolled_back is True
    # Ordering matters: rolling back after close would raise on a real Session.
    assert session.calls == ["rollback", "close"]


def test_active_transaction_is_rolled_back_when_the_hook_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finally-branch must dispose the session on the failure path too."""
    from xagent.web.models import database
    from xagent.web.services import quota_hooks

    session = _FakeSession(in_transaction=True)

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("quota backend down")

    monkeypatch.setattr(quota_hooks, "record_usage", _boom)
    monkeypatch.setattr(quota_hooks, "has_usage_record_hook", lambda: True)
    monkeypatch.setattr(database, "get_session_local", lambda: lambda: session)

    with usage_scope(1):
        add_media_usage(MediaCallType.ASR, 1, model="m")

    assert session.rolled_back is True
    assert session.calls == ["rollback", "close"]


def test_inactive_transaction_is_closed_without_rollback(captured) -> None:
    """The common case: nothing pending, so no rollback is issued."""
    _calls, session = captured
    with usage_scope(1):
        add_media_usage(MediaCallType.ASR, 1, model="m")

    assert session.rolled_back is False
    assert session.calls == ["close"]


def test_scope_restores_the_previous_context(captured) -> None:
    """A nested scope must not leak its usage object into the outer one."""
    _calls, _ = captured
    with usage_scope(1) as outer:
        add_media_usage(MediaCallType.TTS, 1, model="a")
        with usage_scope(2):
            add_media_usage(MediaCallType.TTS, 1, model="b")
        # Back on the outer usage, and the inner call did not land here.
        assert get_token_usage() is outer
        assert len(outer.details) == 1


def test_none_user_id_is_tolerated(captured) -> None:
    """record_usage no-ops on a missing user; the scope must not raise."""
    calls, _ = captured
    user_id: Optional[int] = None
    with usage_scope(user_id):
        add_media_usage(MediaCallType.ASR, 1, model="m")
    assert calls[0]["user_id"] is None


def test_hook_failure_does_not_break_the_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metering must never break the operation it measures."""
    from xagent.web.models import database
    from xagent.web.services import quota_hooks

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("quota backend down")

    monkeypatch.setattr(quota_hooks, "record_usage", _boom)
    monkeypatch.setattr(quota_hooks, "has_usage_record_hook", lambda: True)
    monkeypatch.setattr(database, "get_session_local", lambda: _FakeSession)

    with usage_scope(1):
        add_media_usage(MediaCallType.ASR, 1, model="m")
    # No exception escaped.


def test_no_hook_installed_skips_the_session_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stock configuration has no hook, so _report must not pay for a pool
    checkout per transcription."""
    from xagent.web.models import database
    from xagent.web.services import quota_hooks

    checkouts: list[int] = []

    monkeypatch.setattr(quota_hooks, "has_usage_record_hook", lambda: False)
    monkeypatch.setattr(
        database, "get_session_local", lambda: checkouts.append(1) or _FakeSession
    )

    with usage_scope(1):
        add_media_usage(MediaCallType.ASR, 1, model="m")

    assert checkouts == []


def test_transcribe_endpoint_reports_asr_usage_for_the_authenticated_owner(
    captured, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the real endpoint and observe the quota hook.

    Asserting that the literal "usage_scope(" appears in the source would pass
    with the wrong user, the wrong scope boundary, or no recording at all. This
    observes what actually reaches the hook: owner, quantity, unit and that it
    is reported exactly once.
    """
    calls, _ = captured

    from xagent.core.model.asr import adapter
    from xagent.core.model.asr.base import ASRResult
    from xagent.web.api import model as model_api

    class _FakeASR:
        model_name = "asr-provider"

        async def transcribe(self, audio: Any, **kwargs: Any) -> ASRResult:
            _ = (audio, kwargs)
            # A real ASRResult, not a dict: record_asr_usage reads
            # raw_response/segments off ASRResult, so a dict meters as 0s.
            return ASRResult(text="hello", raw_response={"duration": 12.5})

    class _DBModel:
        # Mirrors the real DBModel: `id` is the integer PK and `model_id` the
        # configured string id. The endpoint returns both model_id and
        # model_name, so a fake carrying only one of them fails on the
        # response build rather than on anything this test is about.
        id = 1
        model_id = "configured-asr-id"
        model_name = "asr-provider"
        category = "speech"

    class _User:
        id = 4242

    class _Upload:
        filename = "clip.wav"
        content_type = "audio/wav"

        async def read(self, *_a: Any, **_k: Any) -> bytes:
            return b"RIFFfake"

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        model_api, "_resolve_asr_model_for_transcription", lambda *a, **k: _DBModel()
    )
    monkeypatch.setattr(adapter, "get_asr_model_instance", lambda *a, **k: _FakeASR())
    monkeypatch.setattr(
        model_api,
        "_read_transcribe_upload_with_size_limit",
        _async_return(b"RIFFfake"),
    )

    response = asyncio.run(
        model_api.transcribe_speech_input(
            file=_Upload(),
            language=None,
            model_id=None,
            db=object(),
            user=_User(),
        )
    )

    # Assert the response too: it is built from db_model after the scope
    # closes, so a fake that diverges from the real schema shows up here
    # instead of as an AttributeError deep in the endpoint.
    assert response["text"] == "hello"
    assert response["model_id"] == "configured-asr-id"
    assert response["model_name"] == "asr-provider"

    # Exactly one report, for the authenticated owner, with the ASR row.
    assert len(calls) == 1, calls
    assert calls[0]["user_id"] == 4242
    media = [d for d in calls[0]["details"] if d.get("type") == "media"]
    assert len(media) == 1, media
    assert media[0]["call_type"] == "asr"
    assert media[0]["unit"] == "seconds"
    assert media[0]["quantity"] == 12.5
    assert media[0]["model"] == "asr-provider"


def test_transcribe_falls_back_to_configured_id_for_a_placeholder_db_name(
    captured, monkeypatch: pytest.MonkeyPatch
) -> None:
    """model_name is nullable=False but not constrained to be meaningful.

    A row whose name is empty or literally "default" must not be billed under
    that string -- the module invariants forbid placeholder identities -- so
    the resolver falls back to the configured id.
    """
    calls, _ = captured

    from xagent.core.model.asr import adapter
    from xagent.core.model.asr.base import ASRResult
    from xagent.web.api import model as model_api

    class _FakeASR:
        model_name = "asr-provider"

        async def transcribe(self, audio: Any, **kwargs: Any) -> ASRResult:
            _ = (audio, kwargs)
            return ASRResult(text="hello", raw_response={"duration": 3.0})

    class _PlaceholderNameDBModel:
        id = 1
        model_id = "configured-asr-id"
        model_name = "default"
        category = "speech"

    class _User:
        id = 7

    class _Upload:
        filename = "clip.wav"
        content_type = "audio/wav"

        async def read(self, *_a: Any, **_k: Any) -> bytes:
            return b"RIFFfake"

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        model_api,
        "_resolve_asr_model_for_transcription",
        lambda *a, **k: _PlaceholderNameDBModel(),
    )
    monkeypatch.setattr(adapter, "get_asr_model_instance", lambda *a, **k: _FakeASR())
    monkeypatch.setattr(
        model_api,
        "_read_transcribe_upload_with_size_limit",
        _async_return(b"RIFFfake"),
    )

    asyncio.run(
        model_api.transcribe_speech_input(
            file=_Upload(),
            language=None,
            model_id=None,
            db=object(),
            user=_User(),
        )
    )

    media = [d for d in calls[0]["details"] if d.get("type") == "media"]
    assert media[0]["model"] == "configured-asr-id"
    assert media[0]["model"] != "default"
