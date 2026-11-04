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


def _flat_result(
    projects,
    *,
    has_more=False,
    page_token=None,
    next_page_token=None,
):
    payload = {"projects": projects, "hasMore": has_more}
    if page_token is not None:
        payload["pageToken"] = page_token
    if next_page_token is not None:
        payload["nextPageToken"] = next_page_token
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structured_content": {"result": payload},
        "is_error": False,
    }


@pytest.mark.asyncio
async def test_rocketlane_project_filter_removes_unrelated_records(caplog):
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

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
    assert checked["structured_content"]["pagination"] == {"hasMore": False}
    assert "Orbital Medical" not in checked["content"][0]["text"]
    assert (
        "locally filtered result for server rocketlane tool get_projects" in caplog.text
    )


@pytest.mark.asyncio
async def test_rocketlane_empty_project_result_is_a_valid_no_match():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    original = _result([])

    checked = await guard.validate(
        {"params": {"projectName": "Does not exist"}}, original
    )

    assert checked is original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [[], {"results": []}],
    ids=["bare-list", "results-key"],
)
async def test_rocketlane_other_explicit_empty_shapes_are_valid(payload):
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    original = {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "is_error": False,
    }

    checked = await guard.validate(
        {"params": {"projectName": "Does not exist"}}, original
    )

    assert checked is original


@pytest.mark.asyncio
async def test_rocketlane_prefers_named_project_list_over_nested_identity_lists():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    original = _result(
        [
            {
                "projectId": "p1",
                "projectName": "Versatile migration",
                "phases": [
                    {"id": f"phase-{index}", "name": f"Phase {index}"}
                    for index in range(5)
                ],
            }
        ]
    )

    checked = await guard.validate({"params": {"projectName": "Versatile"}}, original)

    assert checked is original


@pytest.mark.asyncio
async def test_rocketlane_no_match_on_incomplete_page_keeps_pagination():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {"params": {"projectName": "Versatile"}},
        _result(
            [{"projectId": "p2", "projectName": "Orbital Medical"}],
            has_more=True,
            next_page_token="page-2",
        ),
    )

    assert checked["is_error"] is False
    assert checked["structured_content"]["projects"] == []
    assert checked["structured_content"]["pagination"] == {
        "hasMore": True,
        "nextPageToken": "page-2",
    }


@pytest.mark.asyncio
async def test_rocketlane_flat_pagination_is_preserved_and_next_token_wins():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {"params": {"projectName": "Versatile", "pageToken": "page-2"}},
        _flat_result(
            [
                {"projectId": "p1", "projectName": "Versatile migration"},
                {"projectId": "p2", "projectName": "Orbital Medical"},
            ],
            has_more=True,
            page_token="page-2",
            next_page_token="page-3",
        ),
    )

    assert checked["is_error"] is False
    assert checked["structured_content"]["pagination"] == {
        "hasMore": True,
        "nextPageToken": "page-3",
    }


@pytest.mark.asyncio
async def test_rocketlane_short_record_name_does_not_reverse_match_query():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {"params": {"projectName": "Phone migration"}},
        _result([{"projectId": "p1", "projectName": "One"}]),
    )

    assert checked["is_error"] is True
    assert checked["structured_content"]["reason"] == (
        "the requested project/customer filter was ignored"
    )


@pytest.mark.asyncio
async def test_rocketlane_filter_that_collapses_too_short_fails_closed():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {"params": {"projectName": "C++"}},
        _result([{"projectId": "p1", "projectName": "Clinical migration"}]),
    )

    assert checked["is_error"] is True
    assert checked["structured_content"]["reason"] == (
        "a requested filter was too short to verify safely"
    )


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
async def test_rocketlane_customer_name_only_filter_is_supported():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    original = _result(
        [
            {
                "projectId": "p1",
                "projectName": "Warehouse launch",
                "customerName": "B Free",
            }
        ]
    )

    checked = await guard.validate({"params": {"customerName": "B Free"}}, original)

    assert checked is original


@pytest.mark.asyncio
async def test_rocketlane_missing_customer_field_is_kept_as_unknown():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    original = _result([{"projectId": "p1", "projectName": "Warehouse launch"}])

    checked = await guard.validate({"params": {"customerName": "B Free"}}, original)

    assert checked is original


@pytest.mark.asyncio
async def test_rocketlane_mixed_unknown_customer_records_are_not_claimed_verified():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {"params": {"customerName": "B Free"}},
        _result(
            [
                {"projectId": "p1", "projectName": "Customer omitted"},
                {
                    "projectId": "p2",
                    "projectName": "Known mismatch",
                    "customerName": "Another Co",
                },
            ]
        ),
    )

    assert checked["structured_content"]["projects"] == [
        {"projectId": "p1", "projectName": "Customer omitted"}
    ]
    assert checked["structured_content"]["matching_records_verified"] is False


@pytest.mark.asyncio
async def test_rocketlane_combines_project_and_customer_filters():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {
            "params": {
                "projectName": "Warehouse",
                "customerName": "B Free",
            }
        },
        _result(
            [
                {
                    "projectId": "p1",
                    "projectName": "Warehouse launch",
                    "customer": {"name": "Bfree"},
                },
                {
                    "projectId": "p2",
                    "projectName": "Warehouse expansion",
                    "customer": {"name": "Another Co"},
                },
            ]
        ),
    )

    assert [
        project["projectId"] for project in checked["structured_content"]["projects"]
    ] == ["p1"]


@pytest.mark.asyncio
async def test_rocketlane_ignored_filter_is_a_failure(caplog):
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
    assert "rejected result for server rocketlane tool get_projects" in caplog.text


@pytest.mark.asyncio
async def test_rocketlane_unreadable_filtered_result_is_a_failure():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {"params": {"projectName": "Versatile"}},
        {
            "content": [{"type": "text", "text": '{"status":"ok"}'}],
            "is_error": False,
        },
    )

    assert checked["is_error"] is True
    assert checked["structured_content"]["reason"] == (
        "project records were not readable"
    )


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
async def test_rocketlane_same_page_token_can_be_refetched():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    args = {"params": {"pageToken": "page-2"}}
    projects = [{"projectId": "p2", "projectName": "Two"}]

    first = await guard.validate(
        args,
        _result(projects, has_more=True, next_page_token="page-3"),
    )
    retried = await guard.validate(
        args,
        _result(projects, has_more=True, next_page_token="page-3"),
    )

    assert first["is_error"] is False
    assert retried["is_error"] is False


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
async def test_rocketlane_failed_page_is_not_recorded_as_seen():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    projects = [{"projectId": "p3", "projectName": "Three"}]

    failed = await guard.validate(
        {"params": {"pageToken": "page-2"}},
        _result(projects, has_more=True, next_page_token="page-2"),
    )
    valid = await guard.validate(
        {"params": {"pageToken": "page-3"}},
        _result(projects, has_more=True, next_page_token="page-4"),
    )

    assert failed["is_error"] is True
    assert valid["is_error"] is False


@pytest.mark.asyncio
async def test_rocketlane_different_queries_do_not_share_seen_pages():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    projects = [{"projectId": "p1", "projectName": "One"}]

    first = await guard.validate(
        {"params": {"status": "active"}},
        _result(projects, has_more=True, next_page_token="active-2"),
    )
    other_query = await guard.validate(
        {"params": {"status": "archived", "pageToken": "archived-2"}},
        _result(projects, has_more=True, next_page_token="archived-3"),
    )

    assert first["is_error"] is False
    assert other_query["is_error"] is False


@pytest.mark.asyncio
async def test_rocketlane_has_more_without_next_token_is_a_failure():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")

    checked = await guard.validate(
        {},
        _result(
            [{"projectId": "p1", "projectName": "One"}],
            has_more=True,
        ),
    )

    assert checked["is_error"] is True
    assert checked["structured_content"]["reason"] == (
        "hasMore was true without a next page token"
    )


@pytest.mark.asyncio
async def test_rocketlane_all_matching_records_pass_through_unchanged():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    original = _result(
        [{"projectId": "p1", "projectName": "Versatile Resource Centre"}]
    )

    checked = await guard.validate({"params": {"projectName": "Versatile"}}, original)

    assert checked is original


@pytest.mark.asyncio
async def test_rocketlane_error_result_passes_through_unchanged():
    guard = MCPResultIntegrityGuard("rocketlane", "get_projects")
    original = {"content": [{"type": "text", "text": "failed"}], "is_error": True}

    checked = await guard.validate({"params": {"projectName": "Versatile"}}, original)

    assert checked is original


@pytest.mark.asyncio
async def test_other_rocketlane_tools_are_unchanged():
    guard = MCPResultIntegrityGuard("rocketlane", "get_tasks")
    original = _result([{"projectId": "p2", "projectName": "Orbital Medical"}])

    checked = await guard.validate({"params": {"projectName": "Versatile"}}, original)

    assert checked is original


@pytest.mark.asyncio
async def test_similarly_named_custom_server_is_unchanged():
    guard = MCPResultIntegrityGuard("rocket_lane", "get_projects")
    original = _result([{"projectId": "p2", "projectName": "Orbital Medical"}])

    checked = await guard.validate({"params": {"projectName": "Versatile"}}, original)

    assert checked is original


@pytest.mark.asyncio
async def test_other_connectors_are_unchanged():
    guard = MCPResultIntegrityGuard("notion", "get_projects")
    original = _result([{"projectId": "p2", "projectName": "Orbital Medical"}])

    checked = await guard.validate({"params": {"projectName": "Versatile"}}, original)

    assert checked is original
