"""Behavior tests for ``task_clarification_draft.py``.

Pure-function coverage (guard order, payload construction, idempotency key
derivation) needs no database at all: every argument
``resolve_publishable_clarification`` takes is an in-memory value the
caller already holds. Four tests are the exceptions, each opening a
private SQLite database the same way the interaction-staging suite does:
the cross-run anchor scenario, which stages a real interaction row through
``interaction_handoff`` to confirm the downstream primitive -- not this
module -- is what degrades a cross-run mismatch; and three payload/legacy-
reader shape comparisons (a ``send_message`` draft, a draft with
interactions, and an empty-form ``ask_user_question`` draft where the two
readers are known to still disagree), which build their drafts through
the real ``draft_from_waiting_request`` and call the real
``get_latest_waiting_question`` against a persisted chat message, rather
than checking this module's own output against itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_interaction_schema_shared import (
    make_task,
    make_trace_event,
    make_user,
)
from xagent.core.agent import clarification as clarification_module
from xagent.core.agent.clarification import (
    ClarificationDraft,
    ClarificationRequestItem,
    draft_from_waiting_request,
)
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.services import ops_signals
from xagent.web.services.chat_history_service import (
    get_latest_waiting_question,
    persist_assistant_message,
)
from xagent.web.services.task_clarification_draft import (
    CLARIFICATION_REQUEST_TTL,
    FailClosed,
    NotApplicable,
    Publishable,
    build_clarification_payload,
    clarification_idempotency_key,
    parse_clarification_payload,
    resolve_publishable_clarification,
)
from xagent.web.services.task_command_transport import COMMAND_ID_PATTERN
from xagent.web.services.task_interaction_staging import (
    InteractionAnchor,
    interaction_handoff,
)
from xagent.web.services.task_lease_service import TaskLease


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _draft(
    *, message: str = "what is your favorite color?", **overrides: Any
) -> ClarificationDraft:
    """Build a :class:`ClarificationDraft` for guard/payload tests that do
    not care where a draft came from.

    ``turn_marker`` is computed with the real ``_compose_turn_marker`` from
    whatever ``turn_message_count`` / ``origin_step_id`` / ``requests``
    this call ends up with (defaults or overrides), rather than pinned to a
    literal string: a hand-written marker can drift from what
    ``_compose_turn_marker``'s length-prefixed encoding actually produces
    (component lengths are on the *filtered* value, not letter-counted by
    eye) without any test here noticing.
    """

    values: dict[str, Any] = {
        "source": "send_message",
        "message": message,
        "message_type": "info",
        "interactions": (),
        "requests": (ClarificationRequestItem("respond", "call-1", "call-1"),),
        "origin_step_id": "step-1",
        "origin_execution_id": "exec-1",
        "turn_message_count": 3,
    }
    values.update(overrides)
    if "turn_marker" not in values:
        values["turn_marker"] = clarification_module._compose_turn_marker(
            turn_message_count=values["turn_message_count"],
            origin_step_id=values["origin_step_id"],
            requests=values["requests"],
        )
    return ClarificationDraft(**values)


def _task(**overrides: Any) -> Task:
    values: dict[str, Any] = {
        "id": 1,
        "runner_id": "runner-a",
        "run_id": "run-a",
        "lease_attempt_id": "attempt-a",
    }
    values.update(overrides)
    return Task(**values)


def _lease(**overrides: Any) -> TaskLease:
    values: dict[str, Any] = {
        "task_id": 1,
        "runner_id": "runner-a",
        "run_id": "run-a",
        "attempt_id": "attempt-a",
    }
    values.update(overrides)
    return TaskLease(**values)


def _anchor(**overrides: Any) -> InteractionAnchor:
    values: dict[str, Any] = {
        "trace_event_id": 42,
        "resume_event_id": "resume-event-1",
        "resume_execution_id": "resume-exec-1",
        "resume_run_partition": "run-a",
    }
    values.update(overrides)
    return InteractionAnchor(**values)


@pytest.fixture(autouse=True)
def _reset_ops_signals():
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)
    yield
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)


# ---------------------------------------------------------------------------
# The four-way pairing between run status and draft presence
# ---------------------------------------------------------------------------


def test_waiting_with_draft_is_publishable() -> None:
    result = {"status": "waiting_for_user", "clarification_draft": _draft()}
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, Publishable)
    assert resolution.fail_closed_reason is None


def test_waiting_without_draft_fails_closed_as_missing_draft() -> None:
    result = {"status": "waiting_for_user"}
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, FailClosed)
    assert resolution.fail_closed_reason == "missing_draft"


def test_non_waiting_with_draft_fails_closed_as_draft_status_mismatch() -> None:
    result = {"status": "completed", "clarification_draft": _draft()}
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, FailClosed)
    assert resolution.fail_closed_reason == "draft_status_mismatch"


def test_non_waiting_without_draft_is_not_applicable() -> None:
    result = {"status": "completed"}
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, NotApplicable)
    assert resolution.reason is None
    assert resolution.fail_closed_reason is None


# ---------------------------------------------------------------------------
# The remaining guards, each in isolation
# ---------------------------------------------------------------------------


def test_unfenced_lease_fails_closed() -> None:
    result = {"status": "waiting_for_user", "clarification_draft": _draft()}
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(run_id=None), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, FailClosed)
    assert resolution.fail_closed_reason == "unfenced_lease"


def test_ownership_changed_fails_closed() -> None:
    result = {"status": "waiting_for_user", "clarification_draft": _draft()}
    resolution = resolve_publishable_clarification(
        result,
        task=_task(runner_id="someone-else"),
        lease=_lease(),
        anchor=_anchor(),
        now=_now(),
    )
    assert isinstance(resolution, FailClosed)
    assert resolution.fail_closed_reason == "ownership_changed"


def test_attempt_mismatch_fails_closed() -> None:
    result = {"status": "waiting_for_user", "clarification_draft": _draft()}
    resolution = resolve_publishable_clarification(
        result,
        task=_task(lease_attempt_id="a-later-attempt"),
        lease=_lease(),
        anchor=_anchor(),
        now=_now(),
    )
    assert isinstance(resolution, FailClosed)
    assert resolution.fail_closed_reason == "attempt_mismatch"


def test_guards_run_most_specific_failure_first() -> None:
    """When ownership *and* attempt identity are both wrong on the same
    input, the ownership check must win -- it comes first in the guard
    sequence because it is the more specific, less data-exposing failure.
    A resolver that checked attempt identity before ownership would report
    the wrong classification here even though both are genuinely wrong."""

    result = {"status": "waiting_for_user", "clarification_draft": _draft()}
    resolution = resolve_publishable_clarification(
        result,
        task=_task(runner_id="someone-else", lease_attempt_id="a-later-attempt"),
        lease=_lease(),
        anchor=_anchor(),
        now=_now(),
    )
    assert isinstance(resolution, FailClosed)
    assert resolution.fail_closed_reason == "ownership_changed"


def test_none_attempt_id_skips_the_attempt_guard() -> None:
    """``lease.attempt_id is None`` cannot prove attempt identity at all and
    must be treated as "skip this check", never as "matches" -- the same
    reading ``interaction_handoff`` uses for the identical sentinel."""

    result = {"status": "waiting_for_user", "clarification_draft": _draft()}
    resolution = resolve_publishable_clarification(
        result,
        task=_task(lease_attempt_id="whatever-is-on-the-row"),
        lease=_lease(attempt_id=None),
        anchor=_anchor(),
        now=_now(),
    )
    assert isinstance(resolution, Publishable)


def test_missing_draft_guard_wins_over_unfenced_lease_guard() -> None:
    """Guard 1 (waiting-status vs. draft-presence pairing) must run before
    guard 2 (lease fencing). A waiting run with neither a draft nor a
    fenced lease has to classify as ``missing_draft`` -- whose caller
    contract is "don't discard the round" -- not ``unfenced_lease``, whose
    contract is "discard the round wholesale". Swapping the two guards'
    order would still return a ``FailClosed`` here, but with the wrong
    reason and therefore the wrong consequence for the caller."""

    result = {"status": "waiting_for_user"}
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(run_id=None), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, FailClosed)
    assert resolution.fail_closed_reason == "missing_draft"


def test_missing_anchor_is_not_applicable_not_fail_closed() -> None:
    result = {"status": "waiting_for_user", "clarification_draft": _draft()}
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=None, now=_now()
    )
    assert isinstance(resolution, NotApplicable)
    assert resolution.reason == "no_anchor"


def test_cross_run_anchor_mismatch_still_resolves_as_publishable() -> None:
    """A cross-run anchor is not rejected by this module -- that check was
    deliberately moved to ``interaction_handoff``, which owns a dedicated
    exception and degradation signal for exactly this mismatch."""

    result = {"status": "waiting_for_user", "clarification_draft": _draft()}
    resolution = resolve_publishable_clarification(
        result,
        task=_task(),
        lease=_lease(),
        anchor=_anchor(resume_run_partition="a-different-run"),
        now=_now(),
    )
    assert isinstance(resolution, Publishable)


# ---------------------------------------------------------------------------
# Payload construction and parsing
# ---------------------------------------------------------------------------


def test_payload_round_trip_matches_the_legacy_reader_shape_for_a_send_message_draft(
    tmp_path: Path,
) -> None:
    """``parse_clarification_payload`` must return the exact shape the real
    ``get_latest_waiting_question`` (``chat_history_service.py``) returns
    for the legacy transcript path -- checked against that real function,
    not against this module's own return-type promise.

    ``send_message`` is the source that actually reaches this comparison
    with no interactions: its waiting request never carries an
    ``"interactions"`` key at all (see the ``send_message`` branch of
    ``ReActPattern._handle_control_tool`` in ``react.py``), so
    ``draft_from_waiting_request`` always builds it with an empty
    ``interactions`` tuple, and the websocket outbound-message path that
    persists the matching chat row (``websocket.py``, the
    ``expect_response`` branch) never passes an ``interactions`` keyword
    either, leaving the column ``NULL``. Both readers must call that "no
    interactions", i.e. ``None`` -- never ``[]``, which would claim a form
    existed when none did.
    """

    engine = _engine(tmp_path)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        user_id = make_user(db)
        task_id = make_task(db, user_id=user_id)

        # No question persisted yet: the legacy reader's "nothing found"
        # branch must agree in shape with the parser's "empty payload"
        # branch.
        no_question = get_latest_waiting_question(db, task_id)
        empty_payload_result = parse_clarification_payload({})
        assert no_question == (None, None)
        assert empty_payload_result == (None, None)

        waiting_request = {
            "tool_call_id": "call-1",
            "tool_name": "send_message",
            "message": "what is your favorite color?",
            "message_type": "info",
            "task_text": "pick a color",
            "message_count": 3,
        }
        draft = draft_from_waiting_request(
            waiting_request, execution_id="exec-1", step_id="step-1"
        )
        assert draft is not None
        assert draft.source == "send_message"
        assert draft.interactions == ()

        persist_assistant_message(
            db,
            task_id=task_id,
            user_id=user_id,
            content=draft.message,
            message_type="question",
        )

        real_question, real_interactions = get_latest_waiting_question(db, task_id)
        payload = build_clarification_payload(draft)
        parsed_question, parsed_interactions = parse_clarification_payload(payload)

        assert type(real_question) is type(parsed_question)
        assert isinstance(real_question, str) and isinstance(parsed_question, str)
        assert real_interactions is None
        assert parsed_interactions is None
    finally:
        db.close()
        engine.dispose()


def test_payload_round_trip_matches_the_legacy_reader_shape_for_a_draft_with_interactions(
    tmp_path: Path,
) -> None:
    """The ``ask_user_question`` source is the one whose waiting request
    always carries a populated ``"interactions"`` key (see the
    ``ask_user_question`` branch of ``ReActPattern._handle_control_tool``
    in ``react.py``), and the websocket outbound-message path persists
    that same list into the chat row's ``interactions`` column. Both
    readers must return that list back as a list of dicts, not merely
    each checked in isolation against a type it is guaranteed to satisfy
    by construction."""

    engine = _engine(tmp_path)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        user_id = make_user(db)
        task_id = make_task(db, user_id=user_id)

        interactions = [{"prompt": "pick one"}]
        waiting_request = {
            "tool_call_id": "call-1",
            "tool_name": "ask_user_question",
            "message": "pick one",
            "message_type": "question",
            "interactions": interactions,
            "task_text": "pick one",
            "message_count": 5,
        }
        draft = draft_from_waiting_request(
            waiting_request, execution_id="exec-1", step_id="step-1"
        )
        assert draft is not None
        assert draft.source == "ask_user_question"
        assert draft.interactions == ({"prompt": "pick one"},)

        persist_assistant_message(
            db,
            task_id=task_id,
            user_id=user_id,
            content=draft.message,
            message_type="question",
            interactions=interactions,
        )

        real_question, real_interactions = get_latest_waiting_question(db, task_id)
        payload = build_clarification_payload(draft)
        parsed_question, parsed_interactions = parse_clarification_payload(payload)

        assert type(real_question) is type(parsed_question)
        assert type(real_interactions) is type(parsed_interactions)
        assert isinstance(real_question, str) and isinstance(parsed_question, str)
        assert isinstance(real_interactions, list) and isinstance(
            parsed_interactions, list
        )
        assert all(isinstance(item, dict) for item in real_interactions)
        assert all(isinstance(item, dict) for item in parsed_interactions)
        assert real_interactions == parsed_interactions == [{"prompt": "pick one"}]
    finally:
        db.close()
        engine.dispose()


def test_payload_round_trip_diverges_from_the_legacy_reader_for_an_empty_form_ask_user_question(
    tmp_path: Path,
) -> None:
    """This pins a known, un-reconciled gap between the two readers, not an
    ideal contract they are supposed to share.

    An empty-form ``ask_user_question`` call is legal: the source is
    classified by whether its waiting request carries an ``"interactions"``
    key at all, not by whether that key's value is non-empty (see
    ``draft_from_waiting_request``). Carried through the real path --
    ``_normalize_ask_user_interactions`` in ``react.py`` turns a missing or
    empty ``interactions`` argument into ``[]``, and ``websocket.py``'s
    outbound-message handler passes that ``[]`` straight through to
    ``persist_assistant_message`` because it is a list, not ``None`` -- the
    persisted row's ``interactions`` column ends up holding an actual empty
    list, not ``NULL``. ``get_latest_waiting_question`` therefore returns
    ``[]`` for this row. ``parse_clarification_payload`` still normalizes
    every empty case to ``None``, so the two readers disagree here: ``[]``
    on the legacy side, ``None`` on this module's side. That is today's
    actual behavior, asserted as such -- not a bug this test is waiting to
    catch.
    """

    engine = _engine(tmp_path)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        user_id = make_user(db)
        task_id = make_task(db, user_id=user_id)

        waiting_request = {
            "tool_call_id": "call-1",
            "tool_name": "ask_user_question",
            "message": "anything else?",
            "message_type": "question",
            "interactions": [],
            "task_text": "anything else?",
            "message_count": 5,
        }
        draft = draft_from_waiting_request(
            waiting_request, execution_id="exec-1", step_id="step-1"
        )
        assert draft is not None
        assert draft.source == "ask_user_question"
        assert draft.interactions == ()

        # Mirrors the real websocket outbound-message path: metadata's
        # "interactions" is a list ([]), so it is passed through as [],
        # not None -- the persisted column ends up [], not NULL.
        persist_assistant_message(
            db,
            task_id=task_id,
            user_id=user_id,
            content=draft.message,
            message_type="question",
            interactions=[],
        )

        real_question, real_interactions = get_latest_waiting_question(db, task_id)
        payload = build_clarification_payload(draft)
        parsed_question, parsed_interactions = parse_clarification_payload(payload)

        assert isinstance(real_question, str) and isinstance(parsed_question, str)
        assert real_interactions == []
        assert parsed_interactions is None
    finally:
        db.close()
        engine.dispose()


def test_idempotency_key_matches_the_command_id_pattern() -> None:
    draft = _draft()
    key = clarification_idempotency_key(draft)
    assert COMMAND_ID_PATTERN.fullmatch(key) is not None
    assert key.startswith("clarification.")
    assert len(key) == 46


def test_idempotency_key_ignores_message_content() -> None:
    """The key is derived only from ``turn_marker``; changing ``message``
    (including via truncation) must never change it."""

    base = _draft(message="short")
    changed = _draft(message="a completely different, much longer question")
    assert clarification_idempotency_key(base) == clarification_idempotency_key(changed)


def test_oversized_question_is_truncated_and_the_idempotency_key_is_unaffected() -> (
    None
):
    long_question = "q" * (32 * 1024)
    draft = _draft(message=long_question)
    payload = build_clarification_payload(draft)
    assert payload["question_truncated"] is True
    assert len(payload["question"].encode("utf-8")) <= 16 * 1024
    assert payload["question"] != long_question

    key_before = clarification_idempotency_key(draft)
    truncated_variant = _draft(message=long_question[: len(long_question) // 2])
    key_after = clarification_idempotency_key(truncated_variant)
    assert key_before == key_after


def test_oversized_multi_byte_question_truncates_on_a_character_boundary() -> None:
    """Every character here encodes to 3 UTF-8 bytes, so a byte-slice that
    ignores character boundaries would land mid-character. ``errors="ignore"``
    in ``_truncate_to_byte_limit`` drops a partial trailing character rather
    than raising, but nothing pinned that the result stays valid UTF-8 with
    no stray replacement character -- this test is that pin."""

    long_question = "你" * (10 * 1024)  # ~30 KiB, over the 16 KiB limit
    draft = _draft(message=long_question)
    payload = build_clarification_payload(draft)

    assert payload["question_truncated"] is True
    encoded = payload["question"].encode("utf-8")
    assert len(encoded) <= 16 * 1024
    # Round-trips cleanly and introduces no U+FFFD replacement character --
    # a mid-character byte slice would either raise here or leave one.
    assert encoded.decode("utf-8") == payload["question"]
    assert "�" not in payload["question"]


def test_oversized_interactions_are_dropped_entirely() -> None:
    huge_interactions = tuple({"field": "x" * 2000, "index": i} for i in range(64))
    draft = _draft(interactions=huge_interactions)
    payload = build_clarification_payload(draft)
    assert payload["interactions_dropped"] is True
    assert payload["interactions"] == []
    assert payload["question"] == draft.message


def test_request_leaf_control_characters_are_stripped_from_the_payload() -> None:
    """``requests`` is built from ``ClarificationRequestItem.to_dict()``,
    whose string fields (``tool_name``, ``tool_call_id``,
    ``interaction_id``) come from the same untrusted tool-call surface as
    ``interactions``' free-text leaves and must go through the same
    ``_clean_leaves`` filter, not straight into the payload."""

    dirty_requests = (
        ClarificationRequestItem("tool\x07one", "call\x00-1", "call\x00-1"),
    )
    draft = _draft(requests=dirty_requests)
    payload = build_clarification_payload(draft)
    assert payload["requests"] == [
        {
            "tool_name": "toolone",
            "tool_call_id": "call-1",
            "interaction_id": "call-1",
        }
    ]

    # The idempotency key is derived only from turn_marker (computed from
    # the *raw*, unfiltered requests), so filtering the payload's requests
    # leaves must not move it.
    key_before = clarification_idempotency_key(draft)
    build_clarification_payload(draft)
    assert clarification_idempotency_key(draft) == key_before


def test_message_type_control_characters_are_stripped_from_the_payload() -> None:
    """``message_type`` reaches ``ClarificationDraft`` by way of
    ``react.py``'s ``send_message`` handler --
    ``str(args.get("message_type", "info"))``, a raw LLM tool-call
    argument with no runtime validation against the tool schema's declared
    ``enum``. A JSON-schema ``enum`` is a hint the model is told to follow,
    not a constraint this code enforces, so ``message_type`` needs the
    same control-character filtering ``question`` already gets, not a
    straight pass-through."""

    draft = _draft(message_type="info\x07\x00BEL-AND-NUL")
    payload = build_clarification_payload(draft)
    assert payload["message_type"] == "infoBEL-AND-NUL"

    # Same independence check as the requests-leaf test above: filtering
    # message_type must not move the idempotency key, since that key is
    # derived only from turn_marker.
    key_before = clarification_idempotency_key(draft)
    build_clarification_payload(draft)
    assert clarification_idempotency_key(draft) == key_before


def test_interaction_leaf_control_characters_are_stripped_through_full_payload_assembly() -> (
    None
):
    """``test_character_domain_matches_the_marker_cleaning_domain`` pins
    ``_strip_control_characters`` in isolation; this pins the same
    filtering as it actually runs inside ``build_clarification_payload``
    for a form's ``interactions`` leaves."""

    dirty_interactions = ({"prompt": "pick\x07one", "id": "opt\x00-1"},)
    draft = _draft(interactions=dirty_interactions)
    payload = build_clarification_payload(draft)
    assert payload["interactions"] == [{"prompt": "pickone", "id": "opt-1"}]


def test_payload_still_too_large_after_truncation_is_not_applicable() -> None:
    """Only ``tool_waiting`` -- the multi-tool waiting point -- can
    realistically produce thousands of ``requests`` entries;
    ``send_message`` and ``ask_user_question`` always contribute exactly
    one (see ``ClarificationDraft.requests``'s docstring), so a
    ``send_message`` draft with 6000 requests is not a shape production
    can ever hand this function."""

    many_requests = tuple(
        ClarificationRequestItem(f"tool-{i}", f"call-{i}", f"call-{i}")
        for i in range(6000)
    )
    draft = _draft(source="tool_waiting", requests=many_requests)
    result = {"status": "waiting_for_user", "clarification_draft": draft}
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, NotApplicable)
    assert resolution.reason == "payload_too_large"


def test_question_emptied_by_character_filtering_is_not_applicable() -> None:
    draft = _draft(message="\x00\x01\x02\x1f")
    result = {"status": "waiting_for_user", "clarification_draft": draft}
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, NotApplicable)
    assert resolution.reason == "empty_question"


def test_whitespace_only_question_is_not_applicable() -> None:
    """A question made only of whitespace -- including a tab -- survives
    ``_strip_control_characters`` unchanged, since ``\\n``, ``\\r``, and
    ``\\t`` are the three exceptions that filter keeps. The guard must
    still treat it as empty rather than let a blank question through as
    ``Publishable``."""

    draft = _draft(message="   \t  \n  ")
    result = {"status": "waiting_for_user", "clarification_draft": draft}
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, NotApplicable)
    assert resolution.reason == "empty_question"


def test_character_domain_matches_the_marker_cleaning_domain() -> None:
    """This module's own character filter and
    ``clarification.py``'s marker-cleaning filter cannot share code (the
    core layer never imports the web layer), so both copies must agree on
    every probe string here or the two have silently drifted apart."""

    from xagent.web.services.task_clarification_draft import (
        _strip_control_characters,
    )

    probes = [
        "plain text",
        "line one\nline two",
        "tab\tseparated",
        "carriage\rreturn",
        "null\x00byte",
        "unit separator\x1f",
        "bell\x07character",
        "",
    ]
    for probe in probes:
        assert _strip_control_characters(probe) == clarification_module._marker_clean(
            probe
        ), probe


def test_expires_at_is_now_plus_the_ttl_constant() -> None:
    result = {"status": "waiting_for_user", "clarification_draft": _draft()}
    now = _now()
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=now
    )
    assert isinstance(resolution, Publishable)
    assert resolution.expires_at == now + CLARIFICATION_REQUEST_TTL


# ---------------------------------------------------------------------------
# The multiple-drafts degradation signal
# ---------------------------------------------------------------------------


def test_single_waiting_step_does_not_register_the_multiple_drafts_signal() -> None:
    result = {
        "status": "waiting_for_user",
        "clarification_draft": _draft(),
        "clarification_superseded_step_ids": [],
    }
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, Publishable)
    assert (
        ops_signals.CLARIFICATION_MULTIPLE_DRAFTS
        not in ops_signals.active_degradations()
    )


def test_superseded_steps_register_the_multiple_drafts_signal_and_it_stays_up() -> None:
    """Registering the signal is not undone by the very publish that
    reports it: a successful publish this round does not erase the fact
    that a sibling step's draft was discarded this round, so the signal
    must still be registered once ``resolve_publishable_clarification``
    returns."""

    result = {
        "status": "waiting_for_user",
        "clarification_draft": _draft(),
        "clarification_superseded_step_ids": ["step-2"],
    }
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, Publishable)
    assert (
        ops_signals.CLARIFICATION_MULTIPLE_DRAFTS in ops_signals.active_degradations()
    )


def test_interrupted_winner_with_superseded_steps_registers_the_multiple_drafts_signal() -> (
    None
):
    """When an interrupt wins a wakeup that also superseded a losing
    waiting step, ``AgentExecutionAdapter`` puts
    ``clarification_superseded_step_ids`` on the *interrupted* result, not
    a waiting one (see the ``"interrupted"`` branch of
    ``AgentExecutionAdapter._normalize_result`` in ``execution_adapter.py``).
    The signal has to register here even though the guard sequence
    classifies this round ``NotApplicable(None)`` well before it would
    otherwise touch ``clarification_superseded_step_ids`` -- registering
    must not be gated behind reaching the ``Publishable`` path."""

    result = {
        "status": "interrupted",
        "clarification_superseded_step_ids": ["step-2"],
    }
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, NotApplicable)
    assert resolution.reason is None
    assert (
        ops_signals.CLARIFICATION_MULTIPLE_DRAFTS in ops_signals.active_degradations()
    )


def test_waiting_with_superseded_steps_and_no_anchor_registers_the_multiple_drafts_signal() -> (
    None
):
    """Same point as the interrupted-winner case above, for the other
    degrade-not-fail exit: a waiting round with no resume anchor still
    reports a sibling step having been superseded, even though it never
    reaches the ``Publishable`` branch that used to be the only place this
    registered."""

    result = {
        "status": "waiting_for_user",
        "clarification_draft": _draft(),
        "clarification_superseded_step_ids": ["step-2"],
    }
    resolution = resolve_publishable_clarification(
        result, task=_task(), lease=_lease(), anchor=None, now=_now()
    )
    assert isinstance(resolution, NotApplicable)
    assert resolution.reason == "no_anchor"
    assert (
        ops_signals.CLARIFICATION_MULTIPLE_DRAFTS in ops_signals.active_degradations()
    )


def test_missing_draft_registers_and_a_later_publish_clears_it() -> None:
    missing_result = {"status": "waiting_for_user"}
    resolve_publishable_clarification(
        missing_result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert ops_signals.CLARIFICATION_DRAFT_MISSING in ops_signals.active_degradations()

    clean_result = {"status": "waiting_for_user", "clarification_draft": _draft()}
    resolution = resolve_publishable_clarification(
        clean_result, task=_task(), lease=_lease(), anchor=_anchor(), now=_now()
    )
    assert isinstance(resolution, Publishable)
    assert (
        ops_signals.CLARIFICATION_DRAFT_MISSING not in ops_signals.active_degradations()
    )


# ---------------------------------------------------------------------------
# Cross-run anchor: this module lets it through; the staging primitive
# downstream is what actually degrades it.
# ---------------------------------------------------------------------------


def _engine(tmp_path: Path):
    db_path = tmp_path / "clarification_draft.db"
    engine = create_engine(f"sqlite:///{db_path}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    return engine


def test_cross_run_anchor_mismatch_degrades_downstream_in_the_staging_primitive(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        user_id = make_user(db)
        task_id = make_task(db, user_id=user_id)
        anchor_trace_event_id = make_trace_event(db, task_id=task_id)

        task = db.get(Task, task_id)
        task.runner_id = "runner-a"
        task.run_id = "run-a"
        db.flush([task])

        lease = TaskLease(
            task_id=task_id, runner_id="runner-a", run_id="run-a", attempt_id=None
        )
        draft = _draft()
        result = {"status": "waiting_for_user", "clarification_draft": draft}
        mismatched_anchor = _anchor(
            trace_event_id=anchor_trace_event_id, resume_run_partition="a-different-run"
        )

        resolution = resolve_publishable_clarification(
            result, task=task, lease=lease, anchor=mismatched_anchor, now=_now()
        )
        assert isinstance(resolution, Publishable)

        with interaction_handoff(
            db, lease, task=task, anchor=mismatched_anchor, now=_now()
        ) as handoff:
            handoff.stage(
                kind="clarification",
                protocol_version=1,
                request_payload=resolution.request_payload,
                request_idempotency_key=resolution.request_idempotency_key,
                expires_at=resolution.expires_at,
            )
        db.rollback()

        assert (
            ops_signals.INTERACTION_RUN_PARTITION_MISMATCH_DEGRADED
            in ops_signals.active_degradations()
        )
        assert (
            ops_signals.INTERACTION_HANDOFF_DEGRADED
            not in ops_signals.active_degradations()
        )
    finally:
        db.close()
        engine.dispose()
