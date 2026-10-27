"""Fixtures that apply the tenant-key guard to every test of a session."""

from collections.abc import Iterator

import pytest

from tests.shared import fairness_key_guard


@pytest.fixture(autouse=True, scope="session")
def fairness_key_guard_installed() -> Iterator[None]:
    """Refuse any statement that rewrites a command row's tenant key."""
    fairness_key_guard.install()
    yield


@pytest.fixture(autouse=True, scope="function")
def fairness_key_guard_check(request) -> Iterator[None]:
    """Fail a test whose code was refused a write of the tenant key.

    The guard raises where the statement runs, but production code may catch
    that error, so the record it keeps is checked when the test ends.
    """
    fairness_key_guard.current_test = request.node.nodeid
    yield
    fairness_key_guard.current_test = ""
    refused = fairness_key_guard.take_violations()
    assert not refused, (
        "statements rewrote the tenant key of a task command: "
        + "; ".join(f"{test or 'outside a test'}: {sql}" for test, sql, _ in refused)
    )


def pytest_sessionfinish(session, exitstatus):
    """Report a refused write that no test reported (a late background thread)."""
    refused = fairness_key_guard.take_violations()
    if refused:
        print(
            "\nstatements rewrote the tenant key of a task command after the last "
            "test that could report them:\n"
            + "\n".join(
                f"  {test or 'outside a test'}: {sql}" for test, sql, _ in refused
            )
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
