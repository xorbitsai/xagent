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


@pytest.fixture(autouse=True)
def isolate_password_recovery_attempts():
    """Each web test owns its recovery budgets, including cross-suite runs."""
    from xagent.web.api import auth

    with auth._password_recovery_lock:
        auth._password_recovery_attempts.clear()
        auth._password_recovery_last_sweep = 0
    yield
    with auth._password_recovery_lock:
        auth._password_recovery_attempts.clear()
        auth._password_recovery_last_sweep = 0


@pytest.fixture(autouse=True, scope="function")
def _reset_command_classifier():
    """Keep a test's installed command classifier from leaking into later tests."""
    from xagent.web.services.task_execution_admission import reset_command_classifier

    yield
    reset_command_classifier()
