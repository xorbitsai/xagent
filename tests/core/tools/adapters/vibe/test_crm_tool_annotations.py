"""Read-only annotations on the CRM MCP servers."""

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from xagent.web.tools.mcp import hubspot, intercom, salesforce


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server", "unannotated_writes"),
    [
        (
            hubspot.mcp,
            {
                "hubspot_create_company",
                "hubspot_create_contact",
                "hubspot_create_deal",
                "hubspot_create_note",
                "hubspot_update_company",
                "hubspot_update_contact",
                "hubspot_update_deal",
            },
        ),
        (
            intercom.mcp,
            {
                "intercom_add_internal_note",
                "intercom_close_conversation",
                "intercom_reply_to_conversation",
            },
        ),
        (salesforce.mcp, {"salesforce_create_record", "salesforce_update_record"}),
    ],
    ids=["hubspot", "intercom", "salesforce"],
)
async def test_crm_tool_annotations(server, unannotated_writes):
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as session:
        listing = await session.list_tools()

    assert {tool.name for tool in listing.tools if tool.annotations is None} == (
        unannotated_writes
    )
    for tool in listing.tools:
        if tool.name not in unannotated_writes:
            assert tool.annotations is not None, tool.name
            assert tool.annotations.readOnlyHint is True, tool.name
            assert tool.annotations.destructiveHint is not True, tool.name
