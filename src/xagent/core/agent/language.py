"""Prompt snippets for preserving user-facing response language."""


def response_language_rules(*, subject: str = "current user request") -> str:
    """Return language rules for user-facing prose.

    The model can infer the language from the referenced subject; the important
    constraint is that auxiliary context must not override the user's language.
    """
    return (
        "Response language rules: Use the same natural language as the "
        f"{subject} for all user-facing prose. If the {subject} explicitly asks "
        "to translate, rewrite, or answer in another language, use that requested "
        "target language. Do not let retrieved memories, tool results, source "
        "documents, examples, or earlier turns change the response language unless "
        f"the {subject} explicitly asks for that language change."
    )


def plan_language_rules() -> str:
    """Return language rules for DAG plan generation."""
    return (
        "Plan language rules: Write every plan step task, description, "
        "termination_condition, and completion_evidence in the same natural "
        "language as the current user request. If the current user request "
        "explicitly asks to translate, rewrite, or answer in another language, "
        "use that requested target language for those fields. Any final synthesis "
        "or final result produced from the plan must use that same language. "
        "Do not let retrieved memories, tool results, source documents, examples, "
        "completed step results, or earlier turns change the plan language unless "
        "the current user request explicitly asks for that language change."
    )


def dag_step_language_rules() -> str:
    """Return language rules for executing an individual DAG step."""
    return (
        "Step language rules: Use the same natural language as the current DAG "
        "step title and description for all user-facing prose and for this step's "
        "final_answer. If the current step explicitly asks to translate, rewrite, "
        "or answer in another language, use that requested target language. Do "
        "not let dependency results, tool results, source documents, retrieved "
        "memories, examples, or earlier turns change the step language unless "
        "this current step explicitly asks for that language change."
    )
