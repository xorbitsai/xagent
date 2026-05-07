import ast
import json
from typing import Any, TypeGuard, TypeVar

T = TypeVar("T")


def is_list_of_type(
    element_type: type[T],
    obj: list[Any],
) -> TypeGuard[list[T]]:
    return len(obj) > 0 and all(isinstance(elem, element_type) for elem in obj)


def ensure_list(val: Any) -> list[str] | None:
    """Ensure a value is parsed into a list of strings.
    If the input is a stringified JSON array, it will be parsed.
    """
    if val is None:
        return None

    if isinstance(val, list):
        # Handle the case where the LLM wraps a stringified array in a list: ["['a', 'b']"]
        if len(val) == 1 and isinstance(val[0], str):
            val_str = val[0].strip()
            if (val_str.startswith("[") and val_str.endswith("]")) or (
                val_str.startswith("['") and val_str.endswith("']")
            ):
                try:
                    parsed = ast.literal_eval(val_str)
                    if isinstance(parsed, list):
                        return [str(v) for v in parsed]
                except (ValueError, SyntaxError):
                    try:
                        parsed = json.loads(val_str)
                        if isinstance(parsed, list):
                            return [str(v) for v in parsed]
                    except json.JSONDecodeError:
                        pass
        return [str(v) for v in val]

    if isinstance(val, str):
        val = val.strip()
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass

        # Try ast.literal_eval for single-quoted arrays like "['a', 'b']"
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass

        return [val]
    return None
