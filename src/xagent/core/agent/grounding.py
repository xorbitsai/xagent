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
repair a session whose evidence compaction already discarded. ReAct's forced
answer turn no longer compacts, so that turn's tool observations survive to be
answered from; no other turn's compaction behavior is changed, and
``EVIDENCE_REMOVED_FACTS`` below is what a prompt states once a compaction on
this context has removed observations, and ``EVIDENCE_UNKNOWN_FACTS`` is what
it states for a payload old enough that the engine cannot tell either way;
``evidence_facts`` selects between them. Proposal C (provenance tracking and a
data-source gate) remains open.

The entity-attribution sentence covers a distinct failure the rest of this
rule does not: a value that a tool result genuinely returned, reported under
an entity other than the one that tool call was scoped to. A same-named
contact belonging to a different company, pulled in by a name search that was
never scoped to the company being asked about, is a real production instance
of this -- the fact (a deal's amount and stage) was real, only its reported
owner was not. This is unconditional (unlike the tool-argument clause, it does
not depend on ``can_call_tools``): the misattribution happens when the answer
is composed, which occurs at every call site this module serves.
"""

from __future__ import annotations

VALUE_KINDS = (
    "a number, a person or organization name, an identifier or reference "
    "code, a date, a status, or a row of a table"
)

# What a prompt states once a compaction on this context has removed tool
# observations: what happened, and what the model may not do about it. Held as
# one shared literal because every prompt that carries it must state the same
# facts -- two hand-written copies would drift, and the call that got the
# weaker copy is exactly the one that invents a value.
EVIDENCE_REMOVED_FACTS = (
    "Compaction removed tool observations from this run's context and their "
    "values can no longer be read. If a compaction summary stands above, "
    "treat any value not literally present in that summary -- "
    f"{VALUE_KINDS} -- as unavailable rather than recalled. Do not "
    "reconstruct, estimate, or illustrate a removed value, and do not "
    "present one as an example. "
)

# The same statement for a run whose payload never carried the marker: the
# engine cannot tell a lossless old run from a lossy one, so this states the
# uncertainty instead of either answer, and then imposes the same prohibition.
# Held beside EVIDENCE_REMOVED_FACTS for the same reason that one is shared:
# the call that got the weaker copy is the one that invents a value.
EVIDENCE_UNKNOWN_FACTS = (
    "This context carries no record of whether compaction removed tool "
    "observations from it, so that cannot be determined. Treat any value "
    "not literally present in the context -- "
    f"{VALUE_KINDS} -- as unavailable rather than recalled. Do not "
    "reconstruct, estimate, or illustrate such a value, and do not present "
    "one as an example. "
)


def evidence_facts(state: str) -> str:
    """The statement a tool-less answer prompt carries for one evidence state.

    Empty only for ``"intact"``. An unrecognized state falls to the removed
    wording rather than to silence: silence is the branch that puts the
    fabricated answer back.
    """
    if state == "intact":
        return ""
    if state == "unknown":
        return EVIDENCE_UNKNOWN_FACTS
    return EVIDENCE_REMOVED_FACTS


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
        for a template or sample. It also forbids reporting a genuinely
        sourced fact under an entity other than the one the tool call was
        scoped to, regardless of ``can_call_tools``. When ``can_call_tools``
        is true it also forbids supplying a fact-carrying tool-call argument
        that no source provides, while leaving arguments the model is
        expected to compose untouched -- except for a fact value written
        literally inside composed code or document text, which the sourcing
        requirement still covers unless the request explicitly asked for a
        template or a sample. Both variants distinguish inspected source text
        from citations or search snippets. The tool-capable variant asks for
        requested source content to be retrieved before claiming inspection;
        the forced-answer variant only asks for uninspected content to be
        disclosed, without suggesting another tool call.
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
    entity_attribution_rule = (
        "A fact can be genuinely present in a tool result and still be "
        "reported under the wrong entity: attributing it to a company, "
        "team, project, or account other than the one the tool call was "
        "actually scoped to is fabrication, even though the fact itself is "
        "real. Before naming the entity a sourced fact belongs to, confirm "
        "it came from a tool call scoped to that entity's own id or name, "
        "not from a same-named or related record found while looking for "
        "something else; when it did not, report the fact under the record "
        "it actually came from instead. "
    )
    source_evidence_rule = (
        " A citation or search snippet is not evidence that you inspected the "
        "source body: attribute only claims supported by text actually "
        "available in the conversation, retrieved context, or tool results. "
    )
    source_inspection_rule = (
        "When the user asks you to read or inspect a source, retrieve its "
        "relevant content before claiming to have done so; if retrieval fails, "
        "disclose what you could and could not inspect."
        if can_call_tools
        else "If requested source content was not inspected, disclose that "
        "limitation rather than implying that a source was read."
    )
    return (
        "Do not introduce specific entities, incidents, dates, sources, "
        "causal explanations, or quantitative data (metrics, figures, "
        "statistics, percentages, table rows, or time series) that are not "
        "supported by the conversation, retrieved context, or tool results. "
        f"{entity_attribution_rule}"
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
        f"a value you place inside it -- {VALUE_KINDS} -- is not. The only "
        "case in which content that no source supports may appear -- in the "
        "answer, or inside document text or other content the request asks "
        "you to write and hand to a tool -- is a current user request that "
        "explicitly asks you to write a template or a sample, meaning "
        "content that is not meant to be real; in that case, before any of "
        "that content appears, state in your reply that the request asked "
        "for content that is not real and that none of it comes from a data "
        "source, and keep such content to what the request asked for. "
        "Outside that case a caveat does not make an invented "
        "value acceptable: if you find yourself about to add a note "
        "explaining that some values are not real, remove those values and "
        "report the gap instead."
        f"{tool_argument_rule}"
        f"{source_evidence_rule}{source_inspection_rule}"
    )


def step_intent_not_fact_rule(*, compact: bool = False) -> str:
    """Return the rule that a DAG step's declared intent is not a fact source.

    Args:
        compact: ``True`` for the form rendered at the end of the step's
            system-context scope block, ``False`` for the standalone section of
            the step instruction message. Both forms must carry the same rule
            over the same four fields; only the surrounding prompt shape differs.
            The compact form names the instruction message explicitly because
            that block renders only the title and description itself.
    """
    if compact:
        return (
            "The step title and description above, and the termination condition "
            "and completion evidence in the DAG step instruction message, declare "
            "the work to perform, not "
            "facts about the result. If they presuppose a fact, conclusion, or "
            "solution that this step's tool results and dependency results do not "
            "support, those results decide and the presupposed content must not "
            "reach your answer; report the gap the way your own agent instructions "
            "direct and treat that report as this step done. Facts the user gave in "
            "their own messages stay usable as given."
        )
    return (
        "STEP INTENT IS NOT A SOURCE OF FACTS\n"
        "The step title, description, termination condition, and completion "
        "evidence declare the work to perform and the shape of the result to "
        "report. They are not a source of facts about that result's content. "
        "Where they read as if some fact, "
        "finding, conclusion, recommendation, or workaround were already known, "
        "that is an expectation of what this step may establish, not something "
        "it has established.\n"
        "If they presuppose a fact, conclusion, or solution that this step's "
        "tool results and dependency results do not support, or that those "
        "results contradict, the tool results and dependency results decide and "
        "the presupposed content must not reach your answer. This applies only "
        "to facts that were supposed to come from tool results or dependency "
        "results. Facts the user gave in their own messages, including ones the "
        "plan copied out of a user message, remain usable exactly as given, and "
        "this rule does not restrict how you word them. It restricts the facts "
        "asserted inside content this step asks you to compose, not your choice "
        "of wording for that content.\n"
        "When the information this step needs turns out to be unavailable, or a "
        "dependency result does not support this step's premise, report that gap "
        "the way your own agent instructions tell you to report it, and treat "
        "that report as satisfying this termination condition: it is a complete "
        "and correct result for this step. Do not restate the presupposed "
        "content to fill the gap, and do not retry or stall trying to make the "
        "presupposition true. Do not answer emptily or evasively either: a "
        "description that lays out conditional branches is still valid "
        "instruction, so follow the branch the actual results support and report "
        "every part of this step those results do support."
    )
