"""Startup configuration parsing and fail-fast validation.

Covers env parsing, startup validation, and vocabulary pairings, plus
``Task.source`` normalization -- the current no-strip/no-lower spec.
"""

from __future__ import annotations

import dataclasses
import logging
import re

import pytest

import xagent.config as config
import xagent.web.services.interaction_rollout as ir
from xagent.web.models.task_interaction import (
    INTERACTION_ORIGIN_VOCABULARY,
    TASK_SOURCE_LITERALS,
    TaskInteractionRequest,
    normalize_interaction_origin,
)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv(config.INTERACTION_PROTOCOL_MODE, raising=False)
    monkeypatch.delenv(config.INTERACTION_NATIVE_SOURCES, raising=False)
    monkeypatch.setattr(ir, "_policy", None)
    ir._counters.clear()
    yield
    monkeypatch.setattr(ir, "_policy", None)
    ir._counters.clear()


# ---------------------------------------------------------------------------
# T-C: config parsing + startup validation
# ---------------------------------------------------------------------------


def test_tc1_defaults_to_legacy_with_empty_sources_and_version_1():
    policy = ir.validate_interaction_rollout_at_startup()
    assert policy.mode == "legacy"
    assert policy.native_sources == frozenset()
    assert policy.native_protocol_version == 1


@pytest.mark.parametrize(
    "raw,expected",
    [("legacy", "legacy"), (" READ ", "read"), ("Native", "native")],
)
def test_tc2_valid_modes_parse_with_whitespace_and_case(monkeypatch, raw, expected):
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, raw)
    if expected == "native":
        monkeypatch.setenv(config.INTERACTION_NATIVE_SOURCES, "sdk")
    policy = ir.validate_interaction_rollout_at_startup()
    assert policy.mode == expected


def test_tc3_invalid_mode_raises_with_original_value_and_legal_values(monkeypatch):
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "prod")
    with pytest.raises(ir.InteractionRolloutConfigError) as exc:
        ir.validate_interaction_rollout_at_startup()
    message = str(exc.value)
    assert "prod" in message
    assert "legacy" in message
    assert "read" in message
    assert "native" in message


def test_tc4_all_seven_gating_sources_parse_with_case_and_whitespace(monkeypatch):
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "native")
    monkeypatch.setenv(
        config.INTERACTION_NATIVE_SOURCES,
        " Internal, SDK ,a2a,TRIGGER,widget,Shared_Link,Channel",
    )
    policy = ir.validate_interaction_rollout_at_startup()
    assert policy.native_sources == frozenset(
        {"internal", "sdk", "a2a", "trigger", "widget", "shared_link", "channel"}
    )


def test_tc5_unknown_source_and_duplicate_source_raise_distinct_messages(monkeypatch):
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "native")
    monkeypatch.setenv(config.INTERACTION_NATIVE_SOURCES, "sdk,bogus")
    with pytest.raises(ir.InteractionRolloutConfigError) as exc_unknown:
        ir.validate_interaction_rollout_at_startup()
    unknown_message = str(exc_unknown.value)

    monkeypatch.setattr(ir, "_policy", None)
    monkeypatch.setenv(config.INTERACTION_NATIVE_SOURCES, "sdk,sdk")
    with pytest.raises(ir.InteractionRolloutConfigError) as exc_dup:
        ir.validate_interaction_rollout_at_startup()
    dup_message = str(exc_dup.value)

    assert "bogus" in unknown_message
    assert "sdk" in dup_message
    assert unknown_message != dup_message
    assert "more than once" in dup_message
    assert "more than once" not in unknown_message


def test_tc6_native_mode_with_empty_sources_raises(monkeypatch):
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "native")
    with pytest.raises(ir.InteractionRolloutConfigError):
        ir.validate_interaction_rollout_at_startup()


def test_tc7_legacy_with_nonempty_sources_does_not_raise_but_logs_info(
    monkeypatch, caplog
):
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "legacy")
    monkeypatch.setenv(config.INTERACTION_NATIVE_SOURCES, "sdk")
    with caplog.at_level(
        logging.INFO, logger="xagent.web.services.interaction_rollout"
    ):
        policy = ir.validate_interaction_rollout_at_startup()
    assert policy.mode == "legacy"
    assert policy.native_sources == frozenset({"sdk"})
    assert any(
        "no effect until mode is switched to native" in r.message
        for r in caplog.records
    )


def test_tc8_blank_entries_are_skipped_not_raised(monkeypatch):
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "native")
    monkeypatch.setenv(config.INTERACTION_NATIVE_SOURCES, "sdk,,a2a")
    policy = ir.validate_interaction_rollout_at_startup()
    assert policy.native_sources == frozenset({"sdk", "a2a"})


def test_tc9_policy_is_frozen():
    policy = ir.validate_interaction_rollout_at_startup()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.mode = "native"  # type: ignore[misc]


def test_tc12_idempotent_returns_same_object_identity(monkeypatch):
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "read")
    first = ir.validate_interaction_rollout_at_startup()
    second = ir.validate_interaction_rollout_at_startup()
    assert first is second


def test_tc13_uninitialized_singleton_raises_runtime_error_not_lazy_parse():
    with pytest.raises(RuntimeError):
        ir.get_interaction_rollout_policy()


def test_tc14_trigger_in_allowlist_warns_but_does_not_raise(monkeypatch, caplog):
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "native")
    monkeypatch.setenv(config.INTERACTION_NATIVE_SOURCES, "trigger")
    with caplog.at_level(
        logging.WARNING, logger="xagent.web.services.interaction_rollout"
    ):
        policy = ir.validate_interaction_rollout_at_startup()
    assert policy.native_sources == frozenset({"trigger"})
    assert any("no interactive responder" in r.message for r in caplog.records)


def test_tc10a_origin_vocabulary_equals_check_constraint_in_list():
    check = next(
        c
        for c in TaskInteractionRequest.__table__.constraints
        if getattr(c, "name", None) == "ck_task_interaction_requests_origin"
    )
    values = frozenset(re.findall(r"'([^']+)'", check.sqltext.text))
    assert INTERACTION_ORIGIN_VOCABULARY == values


def test_tc10b_task_source_literals_is_a_strict_subset_of_vocabulary():
    assert TASK_SOURCE_LITERALS <= INTERACTION_ORIGIN_VOCABULARY
    assert TASK_SOURCE_LITERALS != INTERACTION_ORIGIN_VOCABULARY
    assert INTERACTION_ORIGIN_VOCABULARY - TASK_SOURCE_LITERALS == {"internal"}


def test_tc10c_gating_sources_is_origin_vocabulary_plus_channel_only():
    assert ir.INTERACTION_GATING_SOURCES >= INTERACTION_ORIGIN_VOCABULARY
    assert ir.INTERACTION_GATING_SOURCES - INTERACTION_ORIGIN_VOCABULARY == {"channel"}


# ---------------------------------------------------------------------------
# T-S: Task.source normalization -- LIVE spec, no strip/lower
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, ""])
def test_ts1_falsy_normalizes_to_internal(value):
    assert normalize_interaction_origin(value) == "internal"


def test_ts2_whitespace_only_is_unknown_not_internal():
    assert normalize_interaction_origin("   ") is None


@pytest.mark.parametrize(
    "value", ["internal", "sdk", "a2a", "trigger", "widget", "shared_link"]
)
def test_ts3_exact_vocabulary_matches_pass_through(value):
    assert normalize_interaction_origin(value) == value


@pytest.mark.parametrize("value", [" sdk", "SDK", "sdk "])
def test_ts4_near_matches_are_unknown_not_coerced(value):
    assert normalize_interaction_origin(value) is None


def test_ts5_unrecognized_free_text_is_unknown_not_internal():
    assert normalize_interaction_origin("something-else") is None


def test_ts6_env_side_source_list_still_strips_and_lowers(monkeypatch):
    monkeypatch.setenv(config.INTERACTION_NATIVE_SOURCES, " SDK , a2a")
    assert config.get_interaction_native_sources() == ["sdk", "a2a"]
