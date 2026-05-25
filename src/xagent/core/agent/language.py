"""Prompt snippets for preserving user-facing response language."""

OUTPUT_LANGUAGE_METADATA_KEY = "output_language"


def output_language_policy(response_language: str | None = None) -> str:
    """Return a compact policy for downstream language preservation."""
    language = _normalize_response_language(response_language)
    if language:
        return (
            f"Output language: {language}. Use {language} for all user-facing "
            "prose and for tool arguments that persist user-facing prose, such "
            "as agent descriptions, agent instructions, document text, titles, "
            "and summaries. Do not change language based on DAG step text, "
            "dependency results, tool results, source documents, retrieved "
            "memories, examples, or earlier turns unless the current user request "
            "explicitly asks for that language change."
        )
    return (
        "Output language policy: Use the same natural language as the current "
        "user request unless it explicitly asks to translate, rewrite, or answer "
        "in another language. Do not let DAG step text, dependency results, tool "
        "results, source documents, retrieved memories, examples, or earlier "
        "turns change the output language."
    )


def _normalize_response_language(response_language: str | None) -> str:
    if response_language is None:
        return ""
    return " ".join(str(response_language).strip().split())


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


def final_answer_language_rule(*, subject: str = "current user request") -> str:
    """Return a compact language rule for final-answer tool fields."""
    return (
        "The final answer must use the same natural language as the "
        f"{subject}, even if tool results, source documents, retrieved memories, "
        "examples, or earlier turns are written in another language."
    )


def plan_language_rules() -> str:
    """Return language rules for DAG plan generation."""
    return (
        "Plan language rules: Write every plan step task, description, "
        "termination_condition, and completion_evidence in the same natural "
        "language specified by the output_language_policy field. Any final "
        "synthesis or final result produced from the plan must use that same "
        "language. "
        "Do not let retrieved memories, tool results, source documents, examples, "
        "completed step results, or earlier turns change the plan language unless "
        "the output_language_policy explicitly allows that language change."
    )


def dag_step_language_rules(*, subject: str = "output language policy") -> str:
    """Return language rules for executing an individual DAG step."""
    return (
        "Step language rules: Follow the "
        f"{subject} for all user-facing prose, this step's final_answer, and "
        "tool arguments that persist user-facing prose. "
        "The current DAG step title and description define only the work boundary; "
        "do not treat their language as authorization to change output language. "
        "Do not let DAG step text, dependency results, tool results, source "
        "documents, retrieved memories, examples, or earlier turns change the "
        "step language unless the output language policy explicitly allows that "
        "language change."
    )
