"""SQL result normalization shared by execution limits and the worker pipe."""

from base64 import b64encode
from typing import Any

from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse


def normalize_sql_result(value: Any) -> Any:
    """Normalize driver buffers and JSON values before pickling or measuring."""

    def binary(item: bytes | bytearray | memoryview) -> str:
        return "base64:" + b64encode(bytes(item)).decode("ascii")

    return jsonable_encoder(
        value, custom_encoder={bytes: binary, bytearray: binary, memoryview: binary}
    )


def bound_sql_result(
    payload: dict[str, Any], max_result_bytes: int | None
) -> dict[str, Any]:
    """Measure the exact JSON response body, retaining legacy local results."""
    if max_result_bytes is None:
        return payload
    normalized: dict[str, Any] = normalize_sql_result(payload)
    if len(JSONResponse(normalized).body) > max_result_bytes:
        raise ValueError("SQL query result exceeded the configured byte limit")
    return normalized
