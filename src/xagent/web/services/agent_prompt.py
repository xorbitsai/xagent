"""Prompt policies shared by task execution and agent API adapters."""

from typing import List, Optional, cast

from ...core.agent.voice_policy import apply_output_voice, voice_from_preferences


def enhance_system_prompt_with_kb(
    system_prompt: Optional[str], knowledge_bases: Optional[List[str]]
) -> Optional[str]:
    """Append knowledge-base priority instructions when KBs are configured."""
    if not knowledge_bases:
        return system_prompt

    kb_list = ", ".join(knowledge_bases)
    kb_prompt = (
        f"\n\nAvailable knowledge bases: {kb_list}. "
        "These knowledge bases are already selected. "
        "Do not call list_knowledge_bases to discover them; "
        "use knowledge_search directly for answers. "
        "For specific how-to or factual questions, start with one targeted "
        "knowledge_search, inspect all returned results as one evidence set, "
        "and answer from that evidence when it is relevant. Search again only "
        "when the returned results as a group are missing the information "
        "needed to answer the current question."
    )

    if system_prompt:
        return system_prompt + kb_prompt
    return kb_prompt.lstrip("\n")


def apply_user_voice(
    system_prompt: Optional[str], voice: Optional[str]
) -> Optional[str]:
    """Append the given output voice (the current user's onboarding Launch
    step choice, set via PATCH /api/auth/me/preferences) as a `##
    OUTPUT VOICE` section - see core.agent.voice_policy.apply_output_voice
    for the actual policy (shared with delegated AgentTool children).

    Takes the already-resolved voice string rather than a db/user_id:
    callers already have a runtime user object in hand by the time a
    system prompt is assembled (either the full `User` ORM row, or the
    detached `RuntimeUserFields` from an off-loop snapshot -
    task_setup_snapshot.py resolves `voice` onto it from the very same
    query that fetches `id`/`is_admin`), and issuing a fresh query here
    would either duplicate that lookup or - worse - run one against a
    request session that may already have been released back to the
    pool by this point in agent construction."""
    return apply_output_voice(system_prompt, voice)


def voice_from_runtime_user(
    runtime_user: object | None,
) -> Optional[str]:
    """Extract the voice preference from whichever runtime-user shape a
    caller has in hand, without issuing a new query - see
    apply_user_voice's docstring for why. Handles the full `User` ORM row
    (has `.preferences`, a raw JSON dict) and the detached
    `RuntimeUserFields`/`WebSocketPrincipal` snapshots (both already have
    a plain `.voice` string)."""
    if runtime_user is None:
        return None
    voice = getattr(runtime_user, "voice", None)
    if voice is not None:
        return cast(Optional[str], voice)
    return voice_from_preferences(getattr(runtime_user, "preferences", None))
