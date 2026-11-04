import pytest

from xagent.core.tools.adapters.vibe.db_session import tool_session_scope


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_scope_opens_one_session_and_closes_it() -> None:
    made: list[_FakeSession] = []

    def factory() -> _FakeSession:
        s = _FakeSession()
        made.append(s)
        return s

    with tool_session_scope(factory) as db:
        assert db is made[0]
        assert db.closed is False
    assert len(made) == 1
    assert made[0].closed is True


def test_scope_closes_session_on_exception() -> None:
    made: list[_FakeSession] = []

    def factory() -> _FakeSession:
        s = _FakeSession()
        made.append(s)
        return s

    with pytest.raises(RuntimeError, match="boom"):
        with tool_session_scope(factory):
            raise RuntimeError("boom")
    assert made[0].closed is True
