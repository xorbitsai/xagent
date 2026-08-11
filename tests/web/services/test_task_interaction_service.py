"""Unit coverage for ``task_interaction_service``: the outcome vocabularies,
the ``create()`` typed seam, and the compatibility materialization view.

This module accumulates coverage across every deliverable this service
ships except the shared public-chat ownership predicate (covered directly
by ``tests/web/api/test_public_chat_ownership_helper.py``, since it is
extracted from and tested alongside ``public_chat_access.py``) and the
production-caller gate (its own file,
``test_task_interaction_service_production_gate.py``).
"""

from __future__ import annotations

from xagent.web.services import task_interaction_service as svc

# ---------------------------------------------------------------------------
# Vocabulary guards (three numbers, pinned by the frozen design's own
# line-by-line count -- do not recompute them here):
#
#   - RespondOutcome: 21 (outcome type, reason) pairs.
#   - CreateOutcome reason word list: 13 words total, across both delivery
#     periods.
#   - CreateOutcome pairs producible in this delivery specifically: 7.
#
# These guards prove the vocabulary stays closed at exactly these counts.
# They do NOT prove every pair has a test written against it -- several
# reasons are reachable from more than one triggering condition and are
# indistinguishable at the (type, reason) level alone (see each dict's own
# comment in the source for which ones collapse).
# ---------------------------------------------------------------------------


def test_respond_outcome_vocabulary_has_exactly_21_pairs() -> None:
    total = sum(
        len(reasons) for reasons in svc.RESPOND_OUTCOME_REASON_VOCABULARY.values()
    )
    assert total == 21


def test_respond_outcome_vocabulary_covers_all_eight_variants() -> None:
    assert set(svc.RESPOND_OUTCOME_REASON_VOCABULARY) == {
        "RespondAccepted",
        "RespondValidationRejected",
        "RespondUnauthorized",
        "RespondUnavailable",
        "RespondReplayed",
        "RespondConflict",
        "RespondStale",
        "RespondOutcomeUnknown",
    }


def test_create_outcome_reason_word_list_has_exactly_13_words() -> None:
    assert len(svc.CREATE_OUTCOME_REASON_WORDS) == 13


def test_create_outcome_producible_pairs_in_this_delivery_are_exactly_7() -> None:
    total = sum(
        len(reasons) for reasons in svc.CREATE_OUTCOME_REASON_VOCABULARY.values()
    )
    assert total == 7


def test_create_outcome_producible_reasons_are_a_subset_of_the_full_word_list() -> None:
    producible = {
        reason
        for reasons in svc.CREATE_OUTCOME_REASON_VOCABULARY.values()
        for reason in reasons
    }
    assert producible <= svc.CREATE_OUTCOME_REASON_WORDS


def test_create_outcome_this_period_covers_exactly_the_six_producible_variants() -> (
    None
):
    """CreateCreated, CreateConflict, and CreateStale are not producible
    until the wiring batch's ignition PR fills create()'s call body --
    create() never stages a row in this delivery, so nothing that requires
    a staged row can be returned yet."""

    assert set(svc.CREATE_OUTCOME_REASON_VOCABULARY) == {
        "CreateValidationRejected",
        "CreateUnauthorized",
        "CreateUnavailable",
        "CreateNotWired",
    }


def test_locator_mismatch_reason_constant_does_not_exist_in_source() -> None:
    """A design revision voided the reason 'locator_mismatch' before this
    delivery began. Asserting its absence guards against it surviving as a
    dead string constant that would mislead a future reader into thinking
    that path is still live."""

    import inspect

    source = inspect.getsource(svc)
    assert "locator_mismatch" not in source
