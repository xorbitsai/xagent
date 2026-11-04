"""MCP resources.
This module provides MCP resources.
"""

from mcp import ClientSession
from mcp.types import BlobResourceContents, TextResourceContents
from pydantic import AnyUrl


async def load_mcp_resources(
    session: ClientSession,
    *,
    uris: str | list[str] | None = None,
) -> list[tuple[str, list[TextResourceContents | BlobResourceContents]]]:
    """Load MCP resources.

    Args:
        session: MCP client session.
        uris: List of URIs to load. If None, all resources will be loaded.
            Note: Dynamic resources will NOT be loaded when None is specified,
            as they require parameters and are ignored by the MCP SDK's
            session.list_resources() method.

    Returns:
        A list of MCP resources.

    Raises:
        RuntimeError: If an error occurs while fetching a resource.
    """
    results = []
    if uris is None:
        resources_list = await session.list_resources()
        uri_list = [r.uri for r in resources_list.resources]
    elif isinstance(uris, str):
        uri_list = [AnyUrl(uris)]
    elif isinstance(uris, list):
        uri_list = [AnyUrl(uri) for uri in uris]

    current_uri = None
    try:
        for uri in uri_list:
            current_uri = uri
            resource = await session.read_resource(uri)
            res_tuple = (str(uri), resource.contents if resource.contents else [])
            results.append(res_tuple)
    except Exception as e:
        msg = f"Error fetching resource {current_uri}"
        raise RuntimeError(msg) from e

    return results
