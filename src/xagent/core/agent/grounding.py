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

This is the proposal-A mitigation from issue #1235. It reduces how often
unsourced figures are emitted and makes disclosure the instructed default, but
it cannot repair a session whose evidence compaction already discarded.
Proposals B (evidence-preserving compaction) and C (provenance tracking and a
data-source gate) remain open.
"""

from __future__ import annotations


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
        A prompt fragment forbidding unsupported claims and unsourced
        quantitative data, and requiring up-front disclosure of any
        illustrative figures. When ``can_call_tools`` is true it also forbids
        supplying a fact-carrying tool-call argument that no source provides,
        while leaving arguments the model is expected to compose untouched.
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
        "never carry one over from a different record. This does not restrict "
        "values you are expected to compose yourself, such as a search query, "
        "code or a command you write to do the work, a message or answer you "
        "write to the user, or document text you were asked to produce. Treat "
        "a fact-carrying value you cannot source as "
        "missing information rather than inventing it, and omit it when the "
        "tool allows it to be omitted."
        if can_call_tools
        else ""
    )
    return (
        "Do not introduce specific entities, incidents, dates, sources, "
        "causal explanations, or quantitative data (metrics, figures, "
        "statistics, percentages, table rows, or time series) that are not "
        "supported by the conversation, retrieved context, or tool results. "
        f"{insufficient_context_rule}"
        "Never invent figures to fill a gap, and never present invented numbers "
        "as real data; produce unsupported figures only when the user explicitly "
        "asked for a template, mockup, or illustrative example. Labeling is "
        "required either way: if the answer ends up containing any figure that no "
        "tool result or provided context supports, whether or not the user asked "
        "for one, say so up front, before presenting it, and state that those "
        "figures are illustrative placeholders not drawn from any data source."
        f"{tool_argument_rule}"
    )
