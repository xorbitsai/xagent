import pytest


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
