"""Keep real password hashing inexpensive in ordinary web tests."""

from types import SimpleNamespace

import bcrypt
import pytest

from xagent.web.api import auth as auth_api


@pytest.fixture(autouse=True)
def fast_password_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replace only auth's module binding: API-key verification shares the real
    # bcrypt library and has its own production-cost dummy verification policy.
    monkeypatch.setattr(
        auth_api,
        "bcrypt",
        SimpleNamespace(
            gensalt=lambda: bcrypt.gensalt(rounds=4),
            hashpw=bcrypt.hashpw,
            checkpw=bcrypt.checkpw,
        ),
    )
