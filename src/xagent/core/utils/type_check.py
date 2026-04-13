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
        return [str(v) for v in val]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        return [val]
    return None
