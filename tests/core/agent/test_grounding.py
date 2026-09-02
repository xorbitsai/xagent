from __future__ import annotations

from xagent.core.agent.grounding import grounding_rule


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
    assert "illustrative placeholders" in rule
    # Pin the concatenation: a dropped trailing space would still satisfy
    # every membership assertion above.
    assert "verify. Never invent figures" in rule


def test_grounding_rule_without_tools_omits_tool_verification() -> None:
    rule = grounding_rule(can_call_tools=False)

    assert "use an appropriate tool" not in rule
    assert "invented values" in rule
    assert "quantitative data" in rule
    assert "illustrative placeholders" in rule
    assert "invented values. Never invent figures" in rule


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
    assert "illustrative placeholders not drawn from any data source. The same" in rule


def test_grounding_rule_exempts_values_the_model_must_compose() -> None:
    """The prohibition must not reach arguments the model is meant to author.

    ReAct runs with ``tool_choice="required"``, and the answer it writes is
    itself a tool argument. Without this exemption the rule would be violated
    on every turn, which would drain the authority of the answer rules sharing
    the same prompt.
    """
    rule = grounding_rule()

    assert "This does not restrict values you are expected to compose yourself" in rule
    for example in (
        "a search query",
        "code or a command you write to do the work",
        "a message or answer you write to the user",
        "document text you were asked to produce",
    ):
        assert example in rule
    # A page size the model picks is not a claim about the world, so pausing
    # for it would be the over-asking failure the exemption exists to prevent.
    assert (
        "does not reach a default or inferred parameter value such as a page "
        "size or result limit" in rule
    )


def test_grounding_rule_keeps_literal_facts_inside_composed_text_sourced() -> None:
    """The compose exemption is per value, not per argument.

    ``python_executor`` takes one free-form ``code`` argument and
    ``write_file`` one ``content`` argument. Read at whole-argument
    granularity, the exemption would clear an invented identifier for
    delivery into an external system as long as it rode inside composed code
    or document text -- the same outcome the rule exists to prevent.
    """
    rule = grounding_rule()

    assert (
        "A fact value written literally inside such composed code or text is "
        "still subject to the sourcing rule above." in rule
    )


def test_grounding_rule_without_tools_omits_tool_argument_clause() -> None:
    """The three forced-answer sites emit no work-tool call to constrain."""
    rule = grounding_rule(can_call_tools=False)

    assert "tool-call arguments that assert facts" not in rule
    assert "never guess one" not in rule
    assert "compose yourself" not in rule
    assert "still subject to the sourcing rule above" not in rule
    assert "a default or inferred parameter value" not in rule


def test_grounding_rule_requires_labeling_regardless_of_request() -> None:
    """Disclosure must not be conditional on the user asking for a template.

    The #1235 session asked for a real KPI report, so a disclosure duty gated
    on "user asked for an example" would not have applied to it.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert "Labeling is required either way" in rule
        assert "whether or not the user asked for one" in rule
        assert "produce unsupported figures only when the user explicitly" in rule
