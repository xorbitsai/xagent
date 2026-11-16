"""Widget voice tests start from the code defaults of every voice setting."""

from __future__ import annotations

import os

import pytest

WIDGET_VOICE_ENV_PREFIX = "XAGENT_WIDGET_VOICE_"


def clear_widget_voice_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every ``XAGENT_WIDGET_VOICE_*`` variable for the current test.

    The test session loads a developer ``.env`` over the shell environment, so
    a value set there would otherwise decide what these tests assert. Tests
    set the values they depend on themselves.
    """
    for name in [n for n in os.environ if n.startswith(WIDGET_VOICE_ENV_PREFIX)]:
        monkeypatch.delenv(name)
