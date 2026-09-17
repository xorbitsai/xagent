"""Prompt snippet for the onboarding-selected output voice.

Deliberately separate from the output LANGUAGE policy (language.py in this
same package), which controls what language a response is in, not what
tone it uses.

Lives in core (not web/api/agents.py, where this concept originated and
where callers with an authenticated request in hand should still reach it
through ``apply_user_voice``/``voice_from_runtime_user``) so every agent
execution path can apply the same policy to its own system prompt -
including a delegated ``AgentTool`` child, which
core/tools/adapters/vibe/agent_tool.py constructs without ever importing a
web route module.
"""

from typing import Dict, Optional

# The 5 voice options the onboarding "Launch" step offers - must stay
# exactly the set api/auth.py's ``VALID_USER_VOICES`` accepts, checked by
# the module-level guard below. Each example line mirrors the reference
# UI's own "<voice> sounds like" preview, so the model's actual output
# matches what the user was shown when they picked it.
VALID_VOICES = frozenset({"professional", "friendly", "concise", "warm", "playful"})

_VOICE_INSTRUCTIONS: Dict[str, str] = {
    "professional": (
        "Formal and polished: precise word choice, complete sentences, no "
        'slang. Example: "Thanks for reaching out. I have reviewed your '
        'request and will come back to you with an answer by Thursday."'
    ),
    "friendly": (
        "Warm and conversational, like a helpful colleague. Example: "
        "\"Thanks so much for getting in touch! I've had a look and I'll "
        'get you an answer by Thursday."'
    ),
    "concise": (
        "As short as possible: drop pleasantries and filler words, state "
        "only what's needed. Example: \"Reviewed. You will have an answer "
        'by Thursday."'
    ),
    "warm": (
        "Empathetic and reassuring: acknowledge the person's situation "
        'before getting to the point. Example: "Thanks for flagging this '
        "- I know the timing matters, so I've prioritised it and will "
        'have an answer to you by Thursday."'
    ),
    "playful": (
        "Light and upbeat, with personality, not stiff or overly formal. "
        "Example: \"Got it - thanks for the nudge! I'm on it, and you'll "
        'have an answer by Thursday."'
    ),
}
if set(_VOICE_INSTRUCTIONS) != VALID_VOICES:
    # A plain assert would be stripped under `python -O`, silently losing
    # this consistency guarantee in production rather than failing loudly.
    raise ValueError(
        "_VOICE_INSTRUCTIONS must define exactly VALID_VOICES - otherwise a "
        "valid, storable voice preference could silently have no prompt "
        "effect."
    )


def voice_from_preferences(preferences: object) -> Optional[str]:
    """Extract the raw ``voice`` entry from a user's ``preferences`` JSON
    blob, tolerating ``None`` or a malformed non-dict value the same way
    every other reader of this column already does - the column has no
    nested-type constraint, so a corrupted/hand-edited row could hold
    anything here. Returns the raw stored value unvalidated (may be an
    unrecognized string, or - if the column itself was corrupted - even a
    non-string); callers that apply it to a prompt go through
    ``apply_output_voice``, whose own ``isinstance``/lookup guard is what
    makes an invalid value degrade to a no-op rather than raise."""
    if isinstance(preferences, dict):
        return preferences.get("voice")
    return None


_SCOPE_CAVEAT = (
    "This governs only your final natural-language reply to the user. "
    "Arguments passed to any tool call - including but not limited to "
    "agent/workforce names, descriptions, instructions, aliases, or any "
    "other value that gets persisted as configuration rather than shown "
    "to the user as conversation - are not conversation and must stay in "
    "plain, neutral language regardless of this tone."
)


def apply_output_voice(
    system_prompt: Optional[str], voice: Optional[str]
) -> Optional[str]:
    """Append the given output voice as a `## OUTPUT VOICE` section, so
    every agent this user talks to - directly, or as a delegated
    ``AgentTool`` child - writes in the tone they picked.

    Always paired with a caveat scoping that tone to the final reply,
    not to tool-call arguments: some callers (the Workforce Prompt
    Builder, Builder Chat, a delegated child with agent-management tools
    enabled) run one continuous LLM conversation that also emits
    create_agent/update_agent/create_workforce tool calls whose
    arguments get persisted verbatim as configuration - without this,
    a tone like "concise" would start trimming those fields too.
    Baked in here, once, rather than left to each caller to remember
    (a caveat added independently at two call sites, and missing from a
    third that could reach the same tools, is what caused this).

    A no-op when voice is None/empty, doesn't match a known option (e.g.
    an older/unrecognized value), or isn't even a string - the persisted
    JSON has no nested-type constraint, so a corrupted/hand-edited value
    could hold a list or dict here, which would otherwise reach
    ``dict.get`` as an unhashable key and raise ``TypeError`` instead of
    degrading to plain output."""
    instruction = _VOICE_INSTRUCTIONS.get(voice) if isinstance(voice, str) else None
    if not instruction:
        return system_prompt

    voice_prompt = f"\n\n## OUTPUT VOICE\n{instruction}\n\n{_SCOPE_CAVEAT}"
    if system_prompt:
        return system_prompt + voice_prompt
    return voice_prompt.lstrip("\n")
