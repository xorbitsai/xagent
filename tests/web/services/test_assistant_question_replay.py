"""Pairing rules that keep one replayed event per assistant question.

A waiting question is written twice by one producer: once as a TraceEvent
carrying the raw prompt, and once as a transcript row whose content has the
rendered interaction list appended. Replay must emit one of them, not both.
"""

from xagent.core.agent.transcript import build_assistant_transcript_content
from xagent.web.services.assistant_question_replay import (
    ReplayQuestionRow,
    ReplayTraceQuestion,
    plan_assistant_question_replay,
    plan_snapshot_question_replay,
)

INTERACTIONS = [
    {"type": "confirm", "label": "Proceed?", "default": True},
]


def _trace(event_id: str, message: str = "Round 1: please confirm", **kwargs):
    return ReplayTraceQuestion(
        event_id=event_id,
        event_type=kwargs.pop("event_type", "agent_message"),
        message=message,
        interactions=kwargs.pop("interactions", INTERACTIONS),
    )


def _row(row_id: int, message: str = "Round 1: please confirm", **kwargs):
    return ReplayQuestionRow(
        row_id=row_id,
        content=kwargs.pop(
            "content",
            build_assistant_transcript_content(message, INTERACTIONS),
        ),
        message_type=kwargs.pop("message_type", "question"),
        source_event_id=kwargs.pop("source_event_id", None),
    )


def test_row_supersedes_its_own_trace_event_by_source_event_id():
    plan = plan_assistant_question_replay(
        rows=[_row(10, source_event_id="ea62bd91")],
        traces=[_trace("ea62bd91")],
    )

    assert plan.superseded_trace_event_ids == frozenset({"ea62bd91"})
    assert plan.trace_event_id_by_row_id == {10: "ea62bd91"}


def test_legacy_row_without_source_event_id_pairs_by_derived_transcript_text():
    # The row predates ``source_event_id``. Its content is still exactly what
    # ``build_assistant_transcript_content`` produced from the trace payload,
    # so the pair is recoverable without matching rendered text loosely.
    plan = plan_assistant_question_replay(
        rows=[_row(10)],
        traces=[_trace("ea62bd91")],
    )

    assert plan.superseded_trace_event_ids == frozenset({"ea62bd91"})
    assert plan.trace_event_id_by_row_id == {10: "ea62bd91"}


def test_two_rounds_asking_the_same_question_pair_one_to_one():
    # Separate rounds routinely repeat a question verbatim. Each row must claim
    # its own trace event rather than both collapsing onto the first.
    plan = plan_assistant_question_replay(
        rows=[_row(10), _row(12)],
        traces=[_trace("ea62bd91"), _trace("79652fd2")],
    )

    assert plan.trace_event_id_by_row_id == {10: "ea62bd91", 12: "79652fd2"}
    assert plan.superseded_trace_event_ids == frozenset({"ea62bd91", "79652fd2"})


def test_source_event_id_wins_over_an_earlier_text_identical_trace_event():
    plan = plan_assistant_question_replay(
        rows=[_row(12, source_event_id="79652fd2")],
        traces=[_trace("ea62bd91"), _trace("79652fd2")],
    )

    assert plan.trace_event_id_by_row_id == {12: "79652fd2"}
    assert plan.superseded_trace_event_ids == frozenset({"79652fd2"})


def test_a_row_whose_trace_event_is_absent_supersedes_nothing():
    # The trace row can be missing (pruned, or filtered out of the public
    # scope). The transcript row still replays; it just keeps its own id.
    plan = plan_assistant_question_replay(rows=[_row(10)], traces=[])

    assert plan.superseded_trace_event_ids == frozenset()
    assert plan.trace_event_id_by_row_id == {}


def test_non_question_rows_are_left_to_the_existing_content_dedupe():
    # Final answers are already deduped the other way round (row dropped,
    # trace kept) and carry streaming identity this pairing must not disturb.
    plan = plan_assistant_question_replay(
        rows=[_row(10, message_type="assistant_response")],
        traces=[_trace("ea62bd91")],
    )

    assert plan.superseded_trace_event_ids == frozenset()
    assert plan.trace_event_id_by_row_id == {}


def test_relabelled_superseded_question_rows_still_pair():
    plan = plan_assistant_question_replay(
        rows=[_row(10, message_type="question_superseded", source_event_id="ea62bd91")],
        traces=[_trace("ea62bd91")],
    )

    assert plan.trace_event_id_by_row_id == {10: "ea62bd91"}


def test_non_agent_message_trace_events_are_never_claimed():
    plan = plan_assistant_question_replay(
        rows=[_row(10, source_event_id="final-1")],
        traces=[_trace("final-1", event_type="ai_message")],
    )

    assert plan.superseded_trace_event_ids == frozenset()
    assert plan.trace_event_id_by_row_id == {}


def test_question_without_interactions_pairs_on_the_unchanged_message():
    plan = plan_assistant_question_replay(
        rows=[_row(10, content="Which region?", message_type="question")],
        traces=[_trace("ea62bd91", message="Which region?", interactions=None)],
    )

    assert plan.trace_event_id_by_row_id == {10: "ea62bd91"}


def test_normalize_message_is_applied_before_deriving_transcript_text():
    # Persistence reconciles file references into the message *before* building
    # the transcript row, so the raw trace text can differ by exactly that
    # rewrite. Callers pass the same reconciliation in.
    stored = build_assistant_transcript_content(
        "See [report](file:abc123)", INTERACTIONS
    )
    plan = plan_assistant_question_replay(
        rows=[_row(10, content=stored)],
        traces=[_trace("ea62bd91", message="See [report](file:stale)")],
        normalize_message=lambda text: text.replace("file:stale", "file:abc123"),
    )

    assert plan.trace_event_id_by_row_id == {10: "ea62bd91"}


def test_a_blank_source_event_id_falls_back_to_derivation():
    plan = plan_assistant_question_replay(
        rows=[_row(10, source_event_id="   ")],
        traces=[_trace("ea62bd91")],
    )

    assert plan.trace_event_id_by_row_id == {10: "ea62bd91"}


def test_a_source_event_id_naming_a_missing_trace_event_claims_nothing():
    # Not a derivation fallback: the row states which event it came from, and
    # that event is not in this snapshot. Guessing a text-identical neighbour
    # would hand the row a different round's identity.
    plan = plan_assistant_question_replay(
        rows=[_row(10, source_event_id="pruned")],
        traces=[_trace("ea62bd91")],
    )

    assert plan.superseded_trace_event_ids == frozenset()
    assert plan.trace_event_id_by_row_id == {}


class _Row:
    def __init__(
        self,
        id,
        content,
        message_type="question",
        role="assistant",
        source_event_id=None,
    ):
        self.id = id
        self.content = content
        self.message_type = message_type
        self.role = role
        self.source_event_id = source_event_id


class _Event:
    def __init__(self, event_id, event_type="agent_message"):
        self.event_id = event_id
        self.event_type = event_type


def _question_data(
    message="Round 1: please confirm", interactions=INTERACTIONS, **extra
):
    data = {"message": message, "metadata": {"interactions": interactions}}
    data.update(extra)
    return data


def test_snapshot_adapter_pairs_row_and_trace_from_orm_shapes():
    plan = plan_snapshot_question_replay(
        chat_messages=[
            _Row(
                10,
                build_assistant_transcript_content(
                    "Round 1: please confirm", INTERACTIONS
                ),
            ),
        ],
        trace_events=[_Event("ea62bd91")],
        trace_data_by_event_id={"ea62bd91": _question_data()},
    )

    assert plan.trace_event_id_by_row_id == {10: "ea62bd91"}


def test_snapshot_adapter_ignores_user_rows():
    plan = plan_snapshot_question_replay(
        chat_messages=[
            _Row(
                10, "Round 1: please confirm", message_type="user_message", role="user"
            )
        ],
        trace_events=[_Event("ea62bd91")],
        trace_data_by_event_id={"ea62bd91": _question_data()},
    )

    assert plan.trace_event_id_by_row_id == {}


def test_snapshot_adapter_never_lets_an_audit_only_event_claim_a_row():
    # The audit-only twin must not absorb the claim and strand the real one.
    plan = plan_snapshot_question_replay(
        chat_messages=[
            _Row(
                10,
                build_assistant_transcript_content(
                    "Round 1: please confirm", INTERACTIONS
                ),
            ),
        ],
        trace_events=[_Event("audit-1"), _Event("ea62bd91")],
        trace_data_by_event_id={
            "audit-1": _question_data(__audit_only__=True),
            "ea62bd91": _question_data(),
        },
    )

    assert plan.trace_event_id_by_row_id == {10: "ea62bd91"}


def test_snapshot_adapter_skips_events_with_no_normalized_payload():
    plan = plan_snapshot_question_replay(
        chat_messages=[
            _Row(
                10,
                build_assistant_transcript_content(
                    "Round 1: please confirm", INTERACTIONS
                ),
            ),
        ],
        trace_events=[_Event("missing")],
        trace_data_by_event_id={},
    )

    assert plan.trace_event_id_by_row_id == {}


def test_a_legacy_row_never_claims_a_later_round_s_trace_when_its_own_is_absent():
    # Round 1's trace can be missing from the snapshot (filtered out of the
    # public scope, audit-only, or pruned) while round 2's is present. Letting
    # row 1 claim round 2's event id hands it another round's identity, and on
    # the client the two questions then collapse into one bubble -- the #2250
    # failure, reintroduced. Claim nothing instead: replaying both is the
    # pre-fix behaviour, and it never loses a turn.
    plan = plan_assistant_question_replay(
        rows=[_row(10), _row(12)],
        traces=[_trace("79652fd2")],
    )

    assert plan.superseded_trace_event_ids == frozenset()
    assert plan.trace_event_id_by_row_id == {}


def test_one_row_facing_two_identical_traces_claims_neither():
    # Ambiguous the other way round: nothing says which trace is this row's.
    plan = plan_assistant_question_replay(
        rows=[_row(10)],
        traces=[_trace("ea62bd91"), _trace("79652fd2")],
    )

    assert plan.trace_event_id_by_row_id == {}


def test_an_explicit_claim_still_lets_its_legacy_neighbour_derive():
    # The symmetry gate counts only what is left unclaimed, so a row carrying
    # source_event_id must not starve a legacy row asking the same question.
    plan = plan_assistant_question_replay(
        rows=[_row(10, source_event_id="ea62bd91"), _row(12)],
        traces=[_trace("ea62bd91"), _trace("79652fd2")],
    )

    assert plan.trace_event_id_by_row_id == {10: "ea62bd91", 12: "79652fd2"}
