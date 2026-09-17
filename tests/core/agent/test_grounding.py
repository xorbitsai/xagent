from __future__ import annotations

import pytest

import xagent.core.agent.grounding as grounding
from xagent.core.agent.grounding import (
    EVIDENCE_REMOVED_FACTS,
    EVIDENCE_UNKNOWN_FACTS,
    VALUE_KINDS,
    evidence_facts,
    grounding_rule,
)

# The sole sentence that may appear inside the answer without a source: it
# names the exception explicitly and is unique to this rule's wording.
TEMPLATE_EXCEPTION_MARKER = (
    "a current user request that explicitly asks you to write a template"
)


def test_grounding_rule_covers_quantitative_data() -> None:
    rule = grounding_rule()

    for term in (
        "entities",
        "incidents",
        "dates",
        "sources",
        "causal explanations",
        "quantitative data",
        "metrics",
        "figures",
        "statistics",
        "percentages",
        "table rows",
        "time series",
    ):
        assert term in rule
    assert "use an appropriate tool to verify" in rule
    assert TEMPLATE_EXCEPTION_MARKER in rule
    # Pin the concatenation: a dropped trailing space would still satisfy
    # every membership assertion above.
    assert "verify. Never fill a gap" in rule


def test_grounding_rule_without_tools_omits_tool_verification() -> None:
    rule = grounding_rule(can_call_tools=False)

    assert "use an appropriate tool" not in rule
    assert "invented values" in rule
    assert "quantitative data" in rule
    assert TEMPLATE_EXCEPTION_MARKER in rule
    assert "invented values. Never fill a gap" in rule


def test_grounding_rule_covers_fact_carrying_tool_arguments() -> None:
    """A fabricated value is worse as a tool argument than as answer text.

    An invented figure in an answer is visible to the user; the same value
    passed to a connector's write tool is invisible, persistent, and lands in
    an external system.
    """
    rule = grounding_rule()

    for phrase in (
        "tool-call arguments that assert facts",
        "identifiers",
        "reference numbers",
        "a value an earlier tool result actually returned",
        "never guess one",
        "never carry one over from a different record",
        "missing information rather than inventing it",
        "omit it when the tool allows",
    ):
        assert phrase in rule
    # A page size the model picks is not a claim about the world, so pausing
    # for it would be the over-asking failure the exemption exists to prevent.
    assert (
        "does not reach a default or inferred parameter value such as a page "
        "size or result limit" in rule
    )
    # Pin the concatenation onto the answer rules that precede it.
    assert "report the gap instead. The same standard applies" in rule


def test_grounding_rule_exempts_the_wording_the_model_composes() -> None:
    """The prohibition must not reach arguments the model is meant to author.

    ReAct runs with ``tool_choice="required"``, and the answer it writes is
    itself a tool argument. Without this exemption the rule would be violated
    on every turn, which would drain the authority of the answer rules sharing
    the same prompt. The exemption covers only wording, not the facts
    asserted inside it, and it is unconditional so the three forced-answer
    prompts -- which have no tool-argument clause of their own -- still tell
    the model that composing its answer is not itself a violation.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert "This does not restrict the wording you compose" in rule
        for example in (
            "how you phrase your reply",
            "a search query",
            "code or a command you write to do the work",
            "document text you were asked to produce",
        ):
            assert example in rule
        assert "it restricts every fact asserted inside that wording" in rule
        # The answer itself is not named as an example of exempt wording:
        # naming it here would read as a self-exemption for the very text
        # this rule constrains.
        assert "a message or answer you write to the user" not in rule


def test_grounding_rule_scopes_the_answer_as_argument_exemption() -> None:
    """The answer reaches the model as a tool argument too, on the tools path.

    Without this clause, the tool-argument standard (identifiers, dates,
    quantities...) would appear to also govern the answer's own wording,
    which is a different, already-covered concern. This clause exempts only
    the wording standard, not the sourcing rule above it.
    """
    rule = grounding_rule()

    assert "The answer you write to the user reaches you as an argument too" in rule
    assert "while the sourcing rule above still governs every fact inside it" in rule
    # This scoping sentence lives in the tool-argument clause, so it has
    # nothing to say on the no-tools path.
    assert "The answer you write to the user reaches you" not in grounding_rule(
        can_call_tools=False
    )


def test_grounding_rule_keeps_literal_facts_inside_composed_text_sourced() -> None:
    """The compose exemption is per value, not per argument.

    ``python_executor`` takes one free-form ``code`` argument and
    ``write_file`` one ``content`` argument. Read at whole-argument
    granularity, the exemption would clear an invented identifier for
    delivery into an external system as long as it rode inside composed code
    or document text -- the same outcome the rule exists to prevent. The
    same value-kind list used for the top-level prohibition is reused here
    so the two statements cannot silently drift apart.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert (
            "still subject to the sourcing rule above: the text you compose "
            "is yours" in rule
        )
        # The list is set off before "is not" so the qualifier reads against
        # the whole list, not against its last member alone.
        assert f"a value you place inside it -- {VALUE_KINDS} -- is not" in rule
        assert rule.count(VALUE_KINDS) == 2


def test_grounding_rule_without_tools_omits_tool_argument_clause() -> None:
    """The three forced-answer sites emit no work-tool call to constrain."""
    rule = grounding_rule(can_call_tools=False)

    assert "tool-call arguments that assert facts" not in rule
    assert "never guess one" not in rule
    assert "a default or inferred parameter value" not in rule
    # The compose exemption and the sourcing rule over composed text are
    # unconditional, so the no-tools variant still carries them.
    assert "This does not restrict the wording you compose" in rule
    assert "still subject to the sourcing rule above" in rule


def test_grounding_rule_offers_no_reusable_disclaimer_phrasing() -> None:
    """The rule must not hand the model a disclaimer phrase it can paste in.

    The #1235 incident's fabricated answer copied its own disclaimer
    near-verbatim from the rule that was supposed to prevent fabrication.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        for phrase in (
            "illustrative placeholder",
            "illustrative placeholders",
            "illustrative example",
            "mockup",
            "mock data",
            "not drawn from any data source",
            "for demonstration purposes",
        ):
            assert phrase not in rule.lower()
    # "plausible-looking placeholder" names a fabricated tool-argument value,
    # not an answer-text disclaimer, and is kept deliberately.
    assert "plausible-looking placeholder" in grounding_rule()


def test_grounding_rule_prohibits_every_unsourced_value_kind() -> None:
    """The prohibition must not be scoped to numbers alone.

    The #1235 incident fabricated person names and reference codes, neither
    of which the old wording ("figures", "numbers") named.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        for kind in (
            "a number",
            "a person or organization name",
            "an identifier or reference code",
            "a date",
            "a status",
            "a row of a table",
        ):
            assert kind in rule


def test_grounding_rule_does_not_license_labelled_fabrication() -> None:
    """Disclosure is no longer an unconditional duty independent of request.

    The old wording ("Labeling is required either way ... whether or not the
    user asked for one") read as permission to fabricate as long as the
    fabrication was labelled, which is exactly what the #1235 incident did.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        for phrase in (
            "Labeling is required either way",
            "whether or not the user asked",
            "either way",
        ):
            assert phrase not in rule


def test_grounding_rule_exception_requires_an_explicit_current_request() -> None:
    """The sole exception is scoped to the current turn's own request.

    A request from an earlier turn must not license fabrication now: the
    #1235 session's request was for a real report, so nothing said in an
    earlier turn should have been able to relax that.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert (
            "The only case in which content that no source supports may "
            "appear -- in the answer, or inside document text or other "
            "content the request asks you to write and hand to a tool -- is "
            "a current user request that explicitly asks" in rule
        )
        assert (
            "Outside that case a caveat does not make an invented value "
            "acceptable" in rule
        )


def test_grounding_rule_states_sample_nature_before_presenting_it() -> None:
    """On the exception path, the disclosure must precede the content."""
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert "before any of that content appears, state in your reply" in rule


def test_grounding_rule_exception_reaches_content_bound_for_a_tool_argument() -> None:
    """A sample the user asked for is often written into a file, not the answer.

    "Write a sample invoice and save it to sample-invoice.md" delivers the
    requested content through a tool argument. An exception scoped to the
    answer alone would leave the request unanswerable: the compose exemption
    hands every fact inside that content back to the sourcing rule, so the
    exception has to reach the same destination the content goes to.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert (
            "may appear -- in the answer, or inside document text or other "
            "content the request asks you to write and hand to a tool --" in rule
        )


def test_grounding_rule_exception_trigger_qualifies_what_a_sample_means() -> None:
    """The trigger must say what a sample is, not offer a third alternative.

    As a third alternative it could never constrain the second: a request
    classified as "a sample" satisfied the trigger before that phrase was
    read, so "make me a sample table of last quarter's real refunds" opened
    the exception.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert (
            "write a template or a sample, meaning content that is not meant "
            "to be real" in rule
        )
        assert "a sample, or content that is not meant to be real" not in rule


def test_grounding_rule_rejects_caveat_as_a_substitute_for_omission() -> None:
    """A caveat must not be used to launder an otherwise-forbidden value.

    The #1235 incident's fabricated answer paired unsourced rows with a note
    explaining they were illustrative -- the exact pattern this closes.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert "a caveat does not make an invented value acceptable" in rule
        assert "remove those values and report the gap instead" in rule


def test_grounding_rule_keeps_the_prohibition_free_of_routing_terms() -> None:
    """The rule states the prohibition without prescribing a remedy.

    Remedies differ by call site (ReAct can retry with a tool, Auto can
    route to react, DAG has neither); the shared rule leaves that choice to
    the calling pattern, per this module's own docstring.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert "leave that value out and say plainly that it is missing" in rule
        for phrase in (
            "choose react",
            "existing_context_sufficient",
            "re-query",
        ):
            assert phrase not in rule


def test_grounding_module_docstring_states_the_default_as_a_prohibition() -> None:
    """The module docstring must not claim disclosure is the instructed default.

    That claim stopped being true once the rule started forbidding unsourced
    values by default and confining disclosure to the explicit-template
    exception.
    """
    doc = grounding.__doc__ or ""
    assert "instructed default" not in doc
    normalized_doc = " ".join(doc.split())
    # A denial of one phrasing is evaded by any synonym, so pin the claim
    # the docstring must positively make.
    assert "makes reporting the gap the instructed response" in normalized_doc
    # Proposal B is no longer open in full: the forced answer turn now keeps
    # its evidence. The docstring states that behaviour rather than claiming
    # the proposal landed, and still says what remains open.
    assert "ReAct's forced answer turn no longer compacts" in normalized_doc
    assert "no other turn's compaction behavior is changed" in normalized_doc
    assert (
        "Proposal C (provenance tracking and a data-source gate) remains open."
        in normalized_doc
    )
    assert "Proposals B (evidence-preserving compaction)" not in normalized_doc
    assert "illustrative" not in (grounding_rule.__doc__ or "")


def test_evidence_removed_facts_states_the_loss_and_forbids_reconstruction() -> None:
    """The sentence every tool-less answer prompt carries, pinned once.

    The suites that check a prompt carries this text derive their expectation
    from the constant, which by construction cannot notice the constant itself
    being emptied or weakened. This cell is where that is noticed: the wording
    lives here, so rewording it is one deliberate edit rather than a sweep
    across every suite that quotes it.
    """
    assert EVIDENCE_REMOVED_FACTS == (
        "Compaction removed tool observations from this run's context and "
        "their values can no longer be read. If a compaction summary stands "
        "above, treat any value not literally present in that summary -- "
        f"{VALUE_KINDS} -- as unavailable rather than recalled. Do not "
        "reconstruct, estimate, or illustrate a removed value, and do not "
        "present one as an example. "
    )


def test_evidence_unknown_facts_states_the_uncertainty_and_forbids_reconstruction() -> (
    None
):
    """The sentence a payload with no marker key carries, pinned once.

    Mirrors the pin on ``EVIDENCE_REMOVED_FACTS`` above: the wording lives
    here, so rewording it is one deliberate edit rather than a sweep across
    every suite that quotes it. It states uncertainty rather than an
    engine-version self-reference: nothing here tells the model which build
    wrote the payload, only that this context does not record the answer.
    """
    assert EVIDENCE_UNKNOWN_FACTS == (
        "This context carries no record of whether compaction removed tool "
        "observations from it, so that cannot be determined. Treat any "
        "value not literally present in the context -- "
        f"{VALUE_KINDS} -- as unavailable rather than recalled. Do not "
        "reconstruct, estimate, or illustrate such a value, and do not "
        "present one as an example. "
    )


@pytest.mark.parametrize(
    "state, expected",
    [
        ("intact", ""),
        ("removed", EVIDENCE_REMOVED_FACTS),
        ("unknown", EVIDENCE_UNKNOWN_FACTS),
        ("corrupted-or-unrecognized", EVIDENCE_REMOVED_FACTS),
    ],
    ids=["intact", "removed", "unknown", "unrecognized_falls_to_removed"],
)
def test_evidence_facts_selects_by_state(state: str, expected: str) -> None:
    """An unrecognized state falls to the removed wording, not to silence.

    Silence is the branch that puts the fabricated answer back, so a state
    string this function does not recognize must not be treated as intact.
    """
    assert evidence_facts(state) == expected


def test_grounding_rule_forbids_reporting_a_sourced_fact_under_the_wrong_entity() -> (
    None
):
    """A genuinely sourced fact must still be attributed to the right entity.

    Production incident: asked for one company's open deals, the model found
    a same-named contact belonging to a different company (via an unscoped
    name search) and reported that company's real deal as if it belonged to
    the company actually asked about. The deal's amount and stage were real;
    only the reported owner was fabricated -- a failure mode the rest of this
    rule, which is about inventing values outright, does not name.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert (
            "reported under the wrong entity: attributing it to a company, "
            "team, project, or account other than the one the tool call was "
            "actually scoped to is fabrication" in rule
        )
        assert "scoped to that entity's own id or name" in rule
        assert "a same-named or related record found while looking for" in rule
        assert "report the fact under the record it actually came from" in rule


def test_grounding_rule_entity_attribution_is_unconditional() -> None:
    """Unlike the tool-argument clause, this does not depend on can_call_tools.

    The misattribution happens in how the answer is composed, which every
    call site -- including the three forced-answer, no-tool sites -- must
    still guard against.
    """
    with_tools = grounding_rule()
    without_tools = grounding_rule(can_call_tools=False)
    marker = "reported under the wrong entity"
    assert marker in with_tools
    assert marker in without_tools
    # Identical wording in both variants, not just present in both.
    start = with_tools.index("A fact can be genuinely present")
    end = with_tools.index("something else; when it did not,") + len(
        "something else; when it did not,"
    )
    assert with_tools[start:end] == without_tools[start:end]


def test_grounding_rule_entity_attribution_precedes_the_pinned_gap_sentence() -> None:
    """The addition must not disturb the existing insufficient-context pins."""
    assert "verify. Never fill a gap" in grounding_rule()
    assert "invented values. Never fill a gap" in grounding_rule(can_call_tools=False)
