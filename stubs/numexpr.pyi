"""Type stubs for numexpr package."""

from typing import Any

def evaluate(
    expr: str,
    local_dict: dict[str, Any] | None = None,
    global_dict: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Evaluate a string expression using numexpr."""
    ...
