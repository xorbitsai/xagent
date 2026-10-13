"""Connector-scoped integrity checks for successful remote MCP results.

These checks do not try to infer semantics for every MCP server.  They are an
explicit compatibility boundary for hosted connectors whose result contract is
known and whose upstream response can otherwise look successful while ignoring
filters or pagination cursors.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

_ROCKETLANE_SERVER = "rocketlane"
_ROCKETLANE_PROJECT_TOOL = "get_projects"
_PAGE_TOKEN_KEYS = frozenset({"cursor", "nextpagetoken", "pagetoken"})
_PROJECT_ID_KEYS = frozenset({"id", "projectid"})
_PROJECT_NAME_KEYS = frozenset({"name", "projectname"})
_CUSTOMER_NAME_KEYS = frozenset({"companyname", "customercompany", "customername"})
_MAX_WALK_DEPTH = 10
_MAX_WALK_NODES = 20_000


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _search_text(value: Any) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _matches(query: str, candidate: str) -> bool:
    wanted = _search_text(query)
    actual = _search_text(candidate)
    return bool(wanted and actual and (wanted in actual or actual in wanted))


def _walk(value: Any) -> Iterator[Any]:
    pending: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while pending and visited < _MAX_WALK_NODES:
        current, depth = pending.pop()
        visited += 1
        yield current
        if depth >= _MAX_WALK_DEPTH:
            continue
        if isinstance(current, Mapping):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)


def _json_value(value: Any) -> Any | None:
    if isinstance(value, (Mapping, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, (Mapping, list)) else None


def _result_payloads(result: Mapping[str, Any]) -> list[Any]:
    payloads: list[Any] = []
    structured = _json_value(result.get("structured_content"))
    if structured is not None:
        payloads.append(structured)
    if isinstance(structured, Mapping):
        nested_result = _json_value(structured.get("result"))
        if nested_result is not None:
            payloads.insert(0, nested_result)
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, Mapping):
                continue
            parsed = _json_value(item.get("text"))
            if parsed is not None:
                payloads.append(parsed)
    return payloads


def _direct_values(
    record: Mapping[str, Any], accepted_keys: frozenset[str]
) -> list[str]:
    values: list[str] = []
    for name, value in record.items():
        if _key(name) in accepted_keys and isinstance(value, (str, int)):
            text = str(value).strip()
            if text:
                values.append(text)
    return values


def _project_values(
    record: Mapping[str, Any], accepted_keys: frozenset[str]
) -> list[str]:
    values = _direct_values(record, accepted_keys)
    for name, value in record.items():
        if _key(name) == "project" and isinstance(value, Mapping):
            values.extend(_direct_values(value, accepted_keys))
    return values


def _customer_values(record: Mapping[str, Any]) -> list[str]:
    values = _direct_values(record, _CUSTOMER_NAME_KEYS)
    for name, value in record.items():
        if _key(name) not in {"company", "customer", "customercompany"}:
            continue
        if isinstance(value, (str, int)) and str(value).strip():
            values.append(str(value).strip())
        elif isinstance(value, Mapping):
            values.extend(
                _direct_values(
                    value,
                    frozenset({"name", "companyname", "customername"}),
                )
            )
    return values


def _project_identity(record: Mapping[str, Any]) -> str | None:
    identifiers = _project_values(record, _PROJECT_ID_KEYS)
    if identifiers:
        return f"id:{identifiers[0]}"
    names = _project_values(record, _PROJECT_NAME_KEYS)
    return f"name:{_search_text(names[0])}" if names else None


def _project_records(payloads: list[Any]) -> list[Mapping[str, Any]]:
    candidates: list[list[Mapping[str, Any]]] = []
    for payload in payloads:
        for current in _walk(payload):
            if not isinstance(current, list) or not current:
                continue
            records = [item for item in current if isinstance(item, Mapping)]
            if not records:
                continue
            identities = sum(
                _project_identity(record) is not None for record in records
            )
            if identities:
                candidates.append(records)
    if not candidates:
        return []
    return max(
        candidates,
        key=lambda records: (
            sum(_project_identity(record) is not None for record in records),
            len(records),
        ),
    )


def _first_named_value(value: Any, names: frozenset[str]) -> Any | None:
    for current in _walk(value):
        if not isinstance(current, Mapping):
            continue
        for name, candidate in current.items():
            if _key(name) in names and candidate not in (None, ""):
                return candidate
    return None


def _argument_value(args: Mapping[str, Any], *names: str) -> str | None:
    value = _first_named_value(args, frozenset(_key(name) for name in names))
    if isinstance(value, (str, int)) and str(value).strip():
        return str(value).strip()
    return None


def _without_page_tokens(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(name): _without_page_tokens(item)
            for name, item in value.items()
            if _key(name) not in _PAGE_TOKEN_KEYS
        }
    if isinstance(value, list):
        return [_without_page_tokens(item) for item in value]
    return value


def _query_signature(args: Mapping[str, Any]) -> str:
    return json.dumps(
        _without_page_tokens(args),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _project_fingerprint(records: list[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        identity
        for record in records
        if (identity := _project_identity(record)) is not None
    )


def _integrity_failure(reason: str) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    "Rocketlane returned an internally inconsistent project "
                    f"result ({reason}). Do not use these records as evidence; "
                    "report that the Rocketlane data could not be verified."
                ),
            }
        ],
        "structured_content": {
            "status": "error",
            "reason": reason,
            "verified": False,
        },
        "is_error": True,
    }


def _locally_filtered_result(
    records: list[Mapping[str, Any]],
    *,
    project_name: str | None,
    customer_name: str | None,
) -> dict[str, Any]:
    payload = {
        "status": "success",
        "projects": [dict(record) for record in records],
        "matching_records_verified": True,
        "complete": False,
        "local_filter": {
            "projectName": project_name,
            "customerName": customer_name,
        },
        "warning": (
            "The hosted Rocketlane MCP returned additional unrelated records. "
            "XAgent removed them locally. Do not infer that pagination or the "
            "upstream filter was complete."
        ),
    }
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False, default=str),
            }
        ],
        "structured_content": payload,
        "is_error": False,
    }


@dataclass
class MCPResultIntegrityGuard:
    """Execution-local integrity state for one MCP tool adapter."""

    source_server: str | None
    tool_name: str
    _seen_pages: dict[str, set[tuple[str, ...]]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def enabled(self) -> bool:
        return _key(self.source_server or "") == _ROCKETLANE_SERVER and _key(
            self.tool_name
        ) == _key(_ROCKETLANE_PROJECT_TOOL)

    async def validate(
        self,
        args: Mapping[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled or result.get("is_error") is True:
            return result

        project_name = _argument_value(args, "projectName", "project_name")
        customer_name = _argument_value(
            args,
            "customerName",
            "customer_name",
            "companyName",
            "company_name",
        )
        requested_page_token = _argument_value(
            args, "pageToken", "page_token", "nextPageToken", "cursor"
        )
        payloads = _result_payloads(result)
        records = _project_records(payloads)
        if not records:
            if project_name or customer_name or requested_page_token:
                return _integrity_failure("project records were not readable")
            return result

        fingerprint = _project_fingerprint(records)
        signature = _query_signature(args)
        async with self._lock:
            seen = self._seen_pages.setdefault(signature, set())
            if requested_page_token and fingerprint and fingerprint in seen:
                return _integrity_failure(
                    "a later page repeated the same project records"
                )
            if fingerprint:
                seen.add(fingerprint)

        response_token = _first_named_value(payloads, _PAGE_TOKEN_KEYS)
        has_more = _first_named_value(payloads, frozenset({"hasmore"}))
        if (
            requested_page_token
            and response_token is not None
            and str(response_token) == requested_page_token
            and has_more is not False
        ):
            return _integrity_failure("the pagination token did not advance")
        if has_more is True and response_token in (None, ""):
            return _integrity_failure("hasMore was true without a next page token")

        if not project_name and not customer_name:
            return result

        def matches_filters(record: Mapping[str, Any]) -> bool:
            if project_name and not any(
                _matches(project_name, value)
                for value in _project_values(record, _PROJECT_NAME_KEYS)
            ):
                return False
            if customer_name and not any(
                _matches(customer_name, value) for value in _customer_values(record)
            ):
                return False
            return True

        matched = [record for record in records if matches_filters(record)]
        if not matched:
            return _integrity_failure(
                "the requested project/customer filter was ignored"
            )
        if len(matched) != len(records):
            return _locally_filtered_result(
                matched,
                project_name=project_name,
                customer_name=customer_name,
            )
        return result
