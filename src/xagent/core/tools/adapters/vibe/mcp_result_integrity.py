"""Connector-scoped integrity checks for successful remote MCP results.

These checks do not try to infer semantics for every MCP server.  They are an
explicit compatibility boundary for hosted connectors whose result contract is
known and whose upstream response can otherwise look successful while ignoring
filters or pagination cursors.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .....builtin_identity import canonicalize_builtin_identity

_ROCKETLANE_SERVER = "rocketlane"
_ROCKETLANE_PROJECT_TOOL = "get_projects"
_REQUEST_PAGE_TOKEN_KEYS = frozenset({"cursor", "nextpagetoken", "pagetoken"})
_RESPONSE_NEXT_PAGE_TOKEN_KEYS = frozenset({"nextpagetoken"})
_RESPONSE_FALLBACK_PAGE_TOKEN_KEYS = frozenset({"cursor", "pagetoken"})
_PROJECT_ID_KEYS = frozenset({"id", "projectid"})
_PROJECT_NAME_KEYS = frozenset({"name", "projectname"})
_CUSTOMER_NAME_KEYS = frozenset({"companyname", "customercompany", "customername"})
_NESTED_CUSTOMER_NAME_KEYS = frozenset({"name", "companyname", "customername"})
_PROJECT_LIST_KEY_PRIORITY = {"projects": 0, "data": 1, "items": 2, "results": 3}
_PAGINATION_KEYS = frozenset({"pagination"})
_HAS_MORE_KEYS = frozenset({"hasmore"})
_MIN_NORMALIZED_FILTER_CHARS = 2
_MAX_WALK_DEPTH = 10
_MAX_WALK_NODES = 20_000

logger = logging.getLogger(__name__)


def _search_text(value: Any) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _matches(query: str, candidate: str) -> bool:
    wanted = _search_text(query)
    actual = _search_text(candidate)
    return bool(wanted and actual and wanted in actual)


def _walk(value: Any) -> Iterator[Any]:
    for current, _depth, _parent_key in _walk_locations(value):
        yield current


def _walk_locations(value: Any) -> Iterator[tuple[Any, int, str | None]]:
    """Walk breadth-first while retaining the normalized parent key."""
    pending: deque[tuple[Any, int, str | None]] = deque([(value, 0, None)])
    visited = 0
    while pending and visited < _MAX_WALK_NODES:
        current, depth, parent_key = pending.popleft()
        visited += 1
        yield current, depth, parent_key
        if depth >= _MAX_WALK_DEPTH:
            continue
        if isinstance(current, Mapping):
            pending.extend(
                (item, depth + 1, _search_text(name)) for name, item in current.items()
            )
        elif isinstance(current, list):
            pending.extend((item, depth + 1, None) for item in current)


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
    except (ValueError, RecursionError):
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
        if _search_text(name) in accepted_keys and isinstance(value, (str, int)):
            text = str(value).strip()
            if text:
                values.append(text)
    return values


def _project_values(
    record: Mapping[str, Any], accepted_keys: frozenset[str]
) -> list[str]:
    values = _direct_values(record, accepted_keys)
    for name, value in record.items():
        if _search_text(name) == "project" and isinstance(value, Mapping):
            values.extend(_direct_values(value, accepted_keys))
    return values


def _customer_values(record: Mapping[str, Any]) -> list[str]:
    values = _direct_values(record, _CUSTOMER_NAME_KEYS)
    for name, value in record.items():
        if _search_text(name) not in {"company", "customer", "customercompany"}:
            continue
        if isinstance(value, (str, int)) and str(value).strip():
            values.append(str(value).strip())
        elif isinstance(value, Mapping):
            values.extend(_direct_values(value, _NESTED_CUSTOMER_NAME_KEYS))
    return values


def _project_identity(record: Mapping[str, Any]) -> str | None:
    identifiers = _project_values(record, _PROJECT_ID_KEYS)
    if identifiers:
        return f"id:{identifiers[0]}"
    names = _project_values(record, _PROJECT_NAME_KEYS)
    return f"name:{_search_text(names[0])}" if names else None


def _project_records(payloads: list[Any]) -> list[Mapping[str, Any]] | None:
    """Return the most likely project list, preserving an explicit empty list."""
    named_candidates: list[tuple[int, int, int, list[Mapping[str, Any]]]] = []
    fallback_candidates: list[tuple[int, int, int, int, list[Mapping[str, Any]]]] = []
    for payload_index, payload in enumerate(payloads):
        for current, depth, parent_key in _walk_locations(payload):
            if not isinstance(current, list):
                continue
            records = [item for item in current if isinstance(item, Mapping)]
            identities = sum(
                _project_identity(record) is not None for record in records
            )
            is_explicit_empty = not current and (
                depth == 0 or parent_key in _PROJECT_LIST_KEY_PRIORITY
            )
            if (parent_key in _PROJECT_LIST_KEY_PRIORITY and identities) or (
                is_explicit_empty
            ):
                named_candidates.append(
                    (
                        _PROJECT_LIST_KEY_PRIORITY.get(parent_key or "", 0),
                        depth,
                        payload_index,
                        records,
                    )
                )
            if identities:
                # Prefer the shallowest qualifying list before using record
                # count, so a project's nested phases/members cannot displace
                # the outer project collection.
                fallback_candidates.append(
                    (depth, -identities, -len(records), payload_index, records)
                )
    if named_candidates:
        return min(
            named_candidates,
            key=lambda candidate: (candidate[1], candidate[0], candidate[2]),
        )[3]
    if fallback_candidates:
        return min(fallback_candidates, key=lambda candidate: candidate[:4])[4]
    return None


def _first_named_value(value: Any, names: frozenset[str]) -> Any | None:
    for current in _walk(value):
        if not isinstance(current, Mapping):
            continue
        for name, candidate in current.items():
            if _search_text(name) in names and candidate not in (None, ""):
                return candidate
    return None


def _argument_value(args: Mapping[str, Any], *names: str) -> str | None:
    value = _first_named_value(args, frozenset(_search_text(name) for name in names))
    if isinstance(value, (str, int)) and str(value).strip():
        return str(value).strip()
    return None


def _without_page_tokens(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(name): _without_page_tokens(item)
            for name, item in value.items()
            if _search_text(name) not in _REQUEST_PAGE_TOKEN_KEYS
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


def _pagination(
    payloads: list[Any],
    *,
    has_more: Any,
    response_token: Any,
) -> dict[str, Any] | None:
    value = _first_named_value(payloads, _PAGINATION_KEYS)
    pagination = dict(value) if isinstance(value, Mapping) else {}
    if has_more is not None:
        pagination["hasMore"] = has_more
    if response_token is not None:
        pagination["nextPageToken"] = response_token
    return pagination or None


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
    pagination: Mapping[str, Any] | None,
    matching_records_verified: bool = True,
) -> dict[str, Any]:
    payload = {
        "status": "success",
        "projects": [dict(record) for record in records],
        "matching_records_verified": matching_records_verified,
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
    if pagination is not None:
        payload["pagination"] = pagination
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
    _seen_pages: dict[str, dict[str | None, tuple[str, ...]]] = field(
        default_factory=dict
    )

    @property
    def enabled(self) -> bool:
        # Use the same collision-safe built-in identity normalization as the
        # connector registries. In particular, a custom "Rocket-Lane" server
        # remains ``rocket_lane`` after the adapter's source normalization and
        # cannot accidentally enroll in this connector-specific guard.
        return canonicalize_builtin_identity(self.source_server) == (
            _ROCKETLANE_SERVER
        ) and _search_text(self.tool_name) == _search_text(_ROCKETLANE_PROJECT_TOOL)

    def failure_result(self, reason: str) -> dict[str, Any]:
        """Build a fail-closed result for an integrity-check failure."""
        return _integrity_failure(reason)

    def _failure(self, reason: str) -> dict[str, Any]:
        logger.warning(
            "MCP result integrity guard rejected result for server %s tool %s: %s",
            self.source_server,
            self.tool_name,
            reason,
        )
        return self.failure_result(reason)

    async def validate(
        self,
        args: Mapping[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled or result.get("is_error") is True:
            return result

        project_name = _argument_value(args, "projectName")
        customer_name = _argument_value(args, "customerName", "companyName")
        if any(
            value is not None
            and len(_search_text(value)) < _MIN_NORMALIZED_FILTER_CHARS
            for value in (project_name, customer_name)
        ):
            return self._failure("a requested filter was too short to verify safely")
        requested_page_token = _argument_value(
            args, "pageToken", "nextPageToken", "cursor"
        )
        payloads = _result_payloads(result)
        records = _project_records(payloads)
        if records is None:
            if project_name or customer_name or requested_page_token:
                return self._failure("project records were not readable")
            return result

        fingerprint = _project_fingerprint(records)
        signature = _query_signature(args)
        seen = self._seen_pages.setdefault(signature, {})
        if fingerprint and any(
            token != requested_page_token and seen_fingerprint == fingerprint
            for token, seen_fingerprint in seen.items()
        ):
            return self._failure("a later page repeated the same project records")

        response_token = _first_named_value(payloads, _RESPONSE_NEXT_PAGE_TOKEN_KEYS)
        if response_token is None:
            response_token = _first_named_value(
                payloads, _RESPONSE_FALLBACK_PAGE_TOKEN_KEYS
            )
        has_more = _first_named_value(payloads, _HAS_MORE_KEYS)
        if (
            requested_page_token
            and response_token is not None
            and str(response_token) == requested_page_token
            and has_more is not False
        ):
            return self._failure("the pagination token did not advance")
        if has_more is True and response_token in (None, ""):
            return self._failure("hasMore was true without a next page token")

        checked_result = result

        if project_name or customer_name:

            def matches_filters(record: Mapping[str, Any]) -> bool:
                if project_name and not any(
                    _matches(project_name, value)
                    for value in _project_values(record, _PROJECT_NAME_KEYS)
                ):
                    return False
                if customer_name:
                    customer_values = _customer_values(record)
                    if customer_values and not any(
                        _matches(customer_name, value) for value in customer_values
                    ):
                        return False
                return True

            matched = [record for record in records if matches_filters(record)]
            retained_customer_values_missing = bool(
                customer_name
                and any(not _customer_values(record) for record in matched)
            )
            if records and not matched and has_more is not True:
                return self._failure(
                    "the requested project/customer filter was ignored"
                )
            if len(matched) != len(records):
                logger.warning(
                    "MCP result integrity guard locally filtered result for "
                    "server %s tool %s: removed %d of %d project records",
                    self.source_server,
                    self.tool_name,
                    len(records) - len(matched),
                    len(records),
                )
                checked_result = _locally_filtered_result(
                    matched,
                    project_name=project_name,
                    customer_name=customer_name,
                    pagination=_pagination(
                        payloads,
                        has_more=has_more,
                        response_token=response_token,
                    ),
                    matching_records_verified=not retained_customer_values_missing,
                )

        # Record only verified pages. Re-fetching the same token is allowed;
        # only a different token returning an already-seen page is suspect.
        if fingerprint:
            seen[requested_page_token] = fingerprint
        return checked_result
