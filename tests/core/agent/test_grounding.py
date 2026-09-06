from __future__ import annotations

import xagent.core.agent.grounding as grounding
from xagent.core.agent.grounding import VALUE_KINDS, grounding_rule

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
            "appear in the answer is a current user request that explicitly "
            "asks" in rule
        )
        assert (
            "Outside that case a caveat does not make an invented value "
            "acceptable" in rule
        )


def test_grounding_rule_states_sample_nature_before_presenting_it() -> None:
    """On the exception path, the disclosure must precede the content."""
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert "before any of that content appears in the answer, state" in rule
        assert rule.index("before any of that content appears") < rule.index(
            "keep such content to what the request asked for"
        )


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
    assert (
        "Proposals B (evidence-preserving compaction) and C (provenance "
        "tracking and a data-source gate) remain open." in normalized_doc
    )
    assert "illustrative" not in (grounding_rule.__doc__ or "")
