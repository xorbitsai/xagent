from __future__ import annotations

import logging

import pytest

from xagent.web.services import conversation_storage_rollout as csr
from xagent.web.services import ops_signals
from xagent.web.services.ops_signals import EXECUTION_EVENTS_UNOBSERVED

ENTRIES = "XAGENT_EXECUTION_EVENT_STORAGE_ENTRIES"
USER_IDS = "XAGENT_EXECUTION_EVENT_STORAGE_USER_IDS"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ENTRIES, raising=False)
    monkeypatch.delenv(USER_IDS, raising=False)
    ops_signals.clear_degradation(EXECUTION_EVENTS_UNOBSERVED)
    yield
    ops_signals.clear_degradation(EXECUTION_EVENTS_UNOBSERVED)


def test_empty_config_is_all_v1(monkeypatch):
    policy = csr.load_conversation_storage_policy()
    assert policy.entries == frozenset()
    assert policy.owner_user_ids is None
    for entry in csr.STORAGE_ENTRIES:
        assert (
            csr.new_task_conversation_storage_version(entry=entry, owner_user_id=1) == 1
        )


def test_valid_policy_parses(monkeypatch):
    monkeypatch.setenv(ENTRIES, " SDK , trigger ")
    monkeypatch.setenv(USER_IDS, "3, 5")
    policy = csr.load_conversation_storage_policy()
    assert policy.entries == frozenset({"sdk", "trigger"})
    assert policy.owner_user_ids == frozenset({3, 5})


def test_unknown_entry_rejected(monkeypatch):
    monkeypatch.setenv(ENTRIES, "sdk,bogus")
    with pytest.raises(csr.ConversationStorageConfigError, match="bogus"):
        csr.load_conversation_storage_policy()


def test_duplicate_entry_cites_original_spelling(monkeypatch):
    monkeypatch.setenv(ENTRIES, "sdk,a2a,SDK")
    with pytest.raises(csr.ConversationStorageConfigError, match="Duplicate.*'sdk'"):
        csr.load_conversation_storage_policy()


@pytest.mark.parametrize("value", ["abc", "0", "-1", "1,x", "1.5"])
def test_invalid_user_ids_rejected(monkeypatch, value):
    monkeypatch.setenv(ENTRIES, "sdk")
    monkeypatch.setenv(USER_IDS, value)
    with pytest.raises(csr.ConversationStorageConfigError):
        csr.load_conversation_storage_policy()


def test_config_error_is_value_error():
    assert issubclass(csr.ConversationStorageConfigError, ValueError)


def test_selection_matrix(monkeypatch):
    select = csr.new_task_conversation_storage_version
    monkeypatch.setenv(ENTRIES, "sdk,workforce")
    assert select(entry="sdk", owner_user_id=1) == 2
    assert select(entry="a2a", owner_user_id=1) == 1

    # Allowlist.
    monkeypatch.setenv(USER_IDS, "7")
    assert select(entry="sdk", owner_user_id=7) == 2
    assert select(entry="sdk", owner_user_id=8) == 1
    monkeypatch.delenv(USER_IDS)

    # via is a conjunction with entry.
    assert select(entry="workforce", owner_user_id=1, via="sdk") == 2
    assert select(entry="workforce", owner_user_id=1, via="widget") == 1
    assert select(entry="workforce", owner_user_id=1, via=None) == 2
    monkeypatch.setenv(ENTRIES, "sdk")
    assert select(entry="workforce", owner_user_id=1, via="sdk") == 1


def test_selection_reads_env_every_call(monkeypatch):
    select = csr.new_task_conversation_storage_version
    assert select(entry="sdk", owner_user_id=1) == 1
    monkeypatch.setenv(ENTRIES, "sdk")
    assert select(entry="sdk", owner_user_id=1) == 2


def test_unknown_entry_or_via_is_value_error(monkeypatch):
    monkeypatch.setenv(ENTRIES, "sdk")
    with pytest.raises(ValueError):
        csr.new_task_conversation_storage_version(entry="nope", owner_user_id=1)
    with pytest.raises(ValueError):
        csr.new_task_conversation_storage_version(
            entry="sdk", owner_user_id=1, via="nope"
        )


def test_startup_logs_one_info_line(monkeypatch, caplog):
    monkeypatch.setenv(ENTRIES, "trigger,sdk")
    monkeypatch.setenv(USER_IDS, "1,2,3")
    with caplog.at_level(logging.INFO, logger=csr.logger.name):
        policy = csr.validate_conversation_storage_rollout_at_startup()
    assert policy.entries == frozenset({"sdk", "trigger"})
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert lines == [
        "Conversation storage rollout configured: "
        "entries=['sdk', 'trigger'] owner_user_ids=3"
    ]


def test_startup_logs_all_when_no_allowlist(caplog):
    with caplog.at_level(logging.INFO, logger=csr.logger.name):
        csr.validate_conversation_storage_rollout_at_startup()
    assert "entries=[] owner_user_ids=all" in caplog.text


def test_startup_rejects_invalid_config(monkeypatch):
    monkeypatch.setenv(ENTRIES, "bogus")
    with pytest.raises(csr.ConversationStorageConfigError):
        csr.validate_conversation_storage_rollout_at_startup()


def test_unobserved_warns_and_registers(monkeypatch, caplog):
    sentinel = "https://secret-endpoint.invalid/otlp-sentinel"
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", sentinel)
    monkeypatch.setenv("XAGENT_OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", sentinel)
    monkeypatch.setenv(ENTRIES, "web_chat,sdk")
    # The user allowlist does not matter for the signal.
    monkeypatch.setenv(USER_IDS, "1")
    with caplog.at_level(logging.WARNING, logger=csr.logger.name):
        csr.report_unobserved_execution_events(telemetry_started=False)
    assert "sentinel" not in caplog.text
    assert "['sdk', 'web_chat']" in caplog.text
    assert ops_signals.active_degradations()[EXECUTION_EVENTS_UNOBSERVED] == (
        "entries=['sdk', 'web_chat']"
    )


def test_unobserved_silent_when_telemetry_started(monkeypatch, caplog):
    monkeypatch.setenv(ENTRIES, "sdk")
    with caplog.at_level(logging.WARNING, logger=csr.logger.name):
        csr.report_unobserved_execution_events(telemetry_started=True)
    assert caplog.text == ""
    assert EXECUTION_EVENTS_UNOBSERVED not in ops_signals.active_degradations()


def test_unobserved_silent_when_no_entries(caplog):
    with caplog.at_level(logging.WARNING, logger=csr.logger.name):
        csr.report_unobserved_execution_events(telemetry_started=False)
    assert caplog.text == ""
    assert EXECUTION_EVENTS_UNOBSERVED not in ops_signals.active_degradations()
