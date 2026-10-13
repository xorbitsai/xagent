import json

import pytest

from xagent.core.tools.adapters.vibe.mcp_result_integrity import (
    MCPResultIntegrityGuard,
)


def _result(projects, *, has_more=False, next_page_token=None):
    pagination = {"hasMore": has_more}
    if next_page_token is not None:
        pagination["nextPageToken"] = next_page_token
    payload = {"projects": projects, "pagination": pagination}
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structured_content": {"result": payload},
        "is_error": False,
    }


@pytest.mark.asyncio
async def test_rocketlane_project_filter_removes_unrelated_records():
    guard = MCPResultIntegrityGuard("Rocketlane", "get_projects")

    checked = await guard.validate(
        {"params": {"projectName": "Versatile"}},
        _result(
            [
                {"projectId": "p1", "projectName": "Versatile Resource Centre"},
                {"projectId": "p2", "projectName": "Orbital Medical"},
            ]
        ),
    )

    assert checked["is_error"] is False
    assert checked["structured_content"]["matching_records_verified"] is True
    assert checked["structured_content"]["complete"] is False
    assert checked["structured_content"]["projects"] == [
        {"projectId": "p1", "projectName": "Versatile Resource Centre"}
    ]
    assert "Orbital Medical" not in checked["content"][0]["text"]


@pytest.mark.asyncio
async def test_rocketlane_customer_filter_reads_nested_customer_name():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {"params": {"companyName": "B Free"}},
        _result(
            [
                {
                    "projectId": "p1",
                    "projectName": "Warehouse launch",
                    "customer": {"name": "Bfree"},
                },
                {
                    "projectId": "p2",
                    "projectName": "Unrelated",
                    "customer": {"name": "Another Co"},
                },
            ]
        ),
    )

    assert checked["is_error"] is False
    assert [
        project["projectId"] for project in checked["structured_content"]["projects"]
    ] == ["p1"]


@pytest.mark.asyncio
async def test_rocketlane_ignored_filter_is_a_failure():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {"params": {"projectName": "Versatile"}},
        _result([{"projectId": "p2", "projectName": "Orbital Medical"}]),
    )

    assert checked["is_error"] is True
    assert checked["structured_content"] == {
        "status": "error",
        "reason": "the requested project/customer filter was ignored",
        "verified": False,
    }
    assert "Do not use these records as evidence" in checked["content"][0]["text"]


@pytest.mark.asyncio
async def test_rocketlane_repeated_page_is_a_failure():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    projects = [
        {"projectId": "p1", "projectName": "One"},
        {"projectId": "p2", "projectName": "Two"},
    ]

    first = await guard.validate(
        {"params": {"includeFields": ["customer"]}},
        _result(projects, has_more=True, next_page_token="page-2"),
    )
    repeated = await guard.validate(
        {
            "params": {
                "includeFields": ["customer"],
                "pageToken": "page-2",
            }
        },
        _result(projects, has_more=True, next_page_token="page-3"),
    )

    assert first["is_error"] is False
    assert repeated["is_error"] is True
    assert repeated["structured_content"]["reason"] == (
        "a later page repeated the same project records"
    )


@pytest.mark.asyncio
async def test_rocketlane_non_advancing_page_token_is_a_failure():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {"params": {"pageToken": "page-2"}},
        _result(
            [{"projectId": "p3", "projectName": "Three"}],
            has_more=True,
            next_page_token="page-2",
        ),
    )

    assert checked["is_error"] is True
    assert checked["structured_content"]["reason"] == (
        "the pagination token did not advance"
    )


@pytest.mark.asyncio
async def test_other_connectors_are_unchanged():
    guard = MCPResultIntegrityGuard("notion", "get_projects")
    original = _result([{"projectId": "p2", "projectName": "Orbital Medical"}])

    checked = await guard.validate({"params": {"projectName": "Versatile"}}, original)

    assert checked is original
