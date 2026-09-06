"""Shared anti-fabrication rules for prompts that emit answers or tool calls.

Every prompt that produces a final user-facing answer should carry this rule.
That is a design goal, not an enforced invariant: nothing walks the prompt
builders to verify it, and ReAct's no-tool branch still lacks it.

A prompt whose LLM call may invoke work tools additionally receives the
tool-argument clause, which holds fact-carrying argument values to the same
sourcing standard as answer text: a fabricated value is more harmful as a tool
argument than as answer text, because it is invisible to the user and lands in
an external system. That clause is selected by ``can_call_tools`` and is absent
from the forced-answer sites, which emit no work-tool call to constrain. Its
remedy -- ask the user, or finish reporting the gap -- belongs to the calling
pattern, which owns the user-interaction policy this module cannot see.

This is the proposal-A mitigation from issue #1235. It forbids unsourced values
by default and makes reporting the gap the instructed response, but it cannot
repair a session whose evidence compaction already discarded.
Proposals B (evidence-preserving compaction) and C (provenance tracking and a
data-source gate) remain open.
"""

from __future__ import annotations

VALUE_KINDS = (
    "a number, a person or organization name, an identifier or reference "
    "code, a date, a status, or a row of a table"
)


def grounding_rule(*, can_call_tools: bool = True) -> str:
    """Return the grounding rule for answer text and, optionally, tool arguments.

    Args:
        can_call_tools: Whether the receiving LLM call may invoke work tools.
            ``False`` at the three forced-answer sites -- ReAct's forced final
            answer, the DAG completion assessment, and the Auto routing
            decision -- where the rule must tell the model to state the gap
            instead of suggesting a tool call it cannot make, and where the
            tool-argument clause is omitted because no work-tool call is
            possible.

    Returns:
        A prompt fragment forbidding unsupported claims and unsourced values of
        every kind it enumerates, requiring the gap be reported instead, and
        confining unsourced content to a current request that explicitly asks
        for a template or sample. When ``can_call_tools`` is true it also
        forbids supplying a fact-carrying tool-call argument that no source
        provides, while leaving arguments the model is expected to compose
        untouched -- except for a fact value written literally inside composed
        code or document text, which the sourcing requirement still covers.
    """
    insufficient_context_rule = (
        "If available context is insufficient, say so or use an appropriate "
        "tool to verify. "
        if can_call_tools
        else (
            "If available context is insufficient, say so instead of filling "
            "the gap with invented values. "
        )
    )
    tool_argument_rule = (
        " The same standard applies to tool-call arguments that assert facts "
        "rather than wording you compose: record field values, identifiers, "
        "reference numbers, dates, quantities, amounts, statuses, and "
        "account or target references. Take such a value from the user's "
        "messages, the retrieved context, or a value an earlier tool result "
        "actually returned; never guess one, never substitute a "
        "plausible-looking placeholder for one the user has not given, and "
        "never carry one over from a different record. The answer you write "
        "to the user reaches you as an argument too; it is wording you "
        "compose, so this argument standard does not reach it, while the "
        "sourcing rule above still governs every fact inside it. This clause "
        "does not reach a default or inferred parameter value such as a page "
        "size or result limit, which you are expected to decide yourself. "
        "Treat a fact-carrying value you cannot source as missing information "
        "rather than inventing it, and omit it when the tool allows it to be "
        "omitted."
        if can_call_tools
        else ""
    )
    return (
        "Do not introduce specific entities, incidents, dates, sources, "
        "causal explanations, or quantitative data (metrics, figures, "
        "statistics, percentages, table rows, or time series) that are not "
        "supported by the conversation, retrieved context, or tool results. "
        f"{insufficient_context_rule}"
        f"Never fill a gap with an invented value, whether it is {VALUE_KINDS}: "
        "when nothing in this conversation, the provided context, or a tool "
        "result supports a value the answer needs, leave that value out and "
        "say plainly that it is missing, rather than supplying one that "
        "looks right. This does not restrict the wording you compose -- how "
        "you phrase your reply, a search query, code or a command you write "
        "to do the work, or document text you were asked to produce -- it "
        "restricts every fact asserted inside that wording. A fact value "
        "written literally inside such composed code or text is still "
        "subject to the sourcing rule above: the text you compose is yours; "
        f"{VALUE_KINDS} that you place inside it is not. The only case in "
        "which content that no source supports may appear in the answer is a "
        "current user request that explicitly asks you to write a template, "
        "a sample, or content that is not meant to be real; in that case, "
        "before any of that content appears in the answer, state that the "
        "request asked for content that is not real and that none of it "
        "comes from a data source, and keep such content to what the request "
        "asked for. Outside that case a caveat does not make an invented "
        "value acceptable: if you find yourself about to add a note "
        "explaining that some values are not real, remove those values and "
        "report the gap instead."
        f"{tool_argument_rule}"
    )
