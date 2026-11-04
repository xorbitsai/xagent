"""SQL result normalization shared by execution limits and the worker pipe."""

from base64 import b64encode
from typing import Any

from pydantic import ConfigDict, TypeAdapter

_RESULT_JSON = TypeAdapter(dict[str, Any], config=ConfigDict(ser_json_inf_nan="null"))


def normalize_sql_result(value: Any) -> Any:
    """Normalize only buffers; preserve SQL columns and picklable scalar types.

    General-purpose ORM encoders can drop real SQL column names (``_sa*``)
    and coerce Decimal/timedelta values. Let the endpoint-compatible Pydantic
    serializer handle those scalars exactly once at the JSON boundary.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "base64:" + b64encode(bytes(value)).decode("ascii")
    if isinstance(value, dict):
        return {str(key): normalize_sql_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_sql_result(item) for item in value]
    return value


def serialize_sql_result(payload: dict[str, Any]) -> bytes:
    """The single JSON codec used by both the byte ceiling and HTTP response.

    Input buffers must already be normalized. Decimal stays precise as a JSON
    string, intervals use ISO durations, and non-finite floats become null,
    matching the original typed endpoint's Pydantic scalar policy.
    """
    return _RESULT_JSON.dump_json(payload)


def bound_sql_result(
    payload: dict[str, Any], max_result_bytes: int | None
) -> dict[str, Any]:
    """Measure the exact JSON response body, retaining legacy local results."""
    if max_result_bytes is None:
        return payload
    normalized: dict[str, Any] = normalize_sql_result(payload)
    if len(serialize_sql_result(normalized)) > max_result_bytes:
        raise ValueError("SQL query result exceeded the configured byte limit")
    return normalized
