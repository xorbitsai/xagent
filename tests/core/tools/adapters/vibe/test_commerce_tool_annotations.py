"""Read-only annotations on the commerce MCP servers."""

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from xagent.web.tools.mcp import magento, shopify, stripe


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server", "unannotated_writes"),
    [
        (
            magento.mcp,
            {
                "magento_add_order_comment",
                "magento_create_product",
                "magento_update_product",
            },
        ),
        (
            shopify.mcp,
            {
                "shopify_create_product",
                "shopify_update_order",
                "shopify_update_product",
            },
        ),
        (stripe.mcp, {"stripe_create_customer", "stripe_create_refund"}),
    ],
    ids=["magento", "shopify", "stripe"],
)
async def test_commerce_tool_annotations(server, unannotated_writes):
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
