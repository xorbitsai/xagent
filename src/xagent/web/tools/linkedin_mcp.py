import asyncio
import os
import urllib.parse

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

app = Server("linkedin-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_profile",
            description="Get your LinkedIn profile",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_posts",
            description="List your recent posts",
            inputSchema={
                "type": "object",
                "properties": {"count": {"type": "number", "default": 10}},
            },
        ),
        Tool(
            name="create_post",
            description="Publish a text post to LinkedIn",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content of the post",
                    }
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="create_article_post",
            description="Publish an article post to LinkedIn",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "articleUrl": {"type": "string"},
                    "articleTitle": {"type": "string"},
                    "articleDescription": {"type": "string"},
                },
                "required": ["text", "articleUrl", "articleTitle"],
            },
        ),
        Tool(
            name="delete_post",
            description="Delete a LinkedIn post by URN",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_urn": {
                        "type": "string",
                        "description": "The URN of the post to delete (e.g., urn:li:share:12345)",
                    }
                },
                "required": ["post_urn"],
            },
        ),
        Tool(
            name="create_comment",
            description="Add a comment to a LinkedIn post",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_urn": {
                        "type": "string",
                        "description": "The URN of the post to comment on",
                    },
                    "text": {"type": "string", "description": "The comment text"},
                },
                "required": ["post_urn", "text"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
    if not token:
        return [
            TextContent(
                type="text",
                text="Error: LINKEDIN_ACCESS_TOKEN environment variable is missing",
            )
        ]

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "LinkedIn-Version": "202603",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        if name == "get_profile":
            r = requests.get(
                "https://api.linkedin.com/v2/userinfo", headers=headers, proxies=proxies
            )
            r.raise_for_status()
            return [TextContent(type="text", text=r.text)]

        elif name == "list_posts":
            count = int(arguments.get("count", 10))
            r = requests.get(
                "https://api.linkedin.com/v2/userinfo", headers=headers, proxies=proxies
            )
            r.raise_for_status()
            sub = r.json().get("sub", "")
            author_urn = urllib.parse.quote(f"urn:li:person:{sub}")
            url = f"https://api.linkedin.com/rest/posts?author={author_urn}&q=author&count={count}&sortBy=LAST_MODIFIED"
            r2 = requests.get(url, headers=headers, proxies=proxies)
            r2.raise_for_status()
            return [TextContent(type="text", text=r2.text)]

        elif name == "create_post":
            text = arguments.get("text")
            r = requests.get(
                "https://api.linkedin.com/v2/userinfo", headers=headers, proxies=proxies
            )
            r.raise_for_status()
            sub = r.json().get("sub", "")
            author_urn = f"urn:li:person:{sub}"
            body = {
                "author": author_urn,
                "commentary": text,
                "visibility": "PUBLIC",
                "distribution": {
                    "feedDistribution": "MAIN_FEED",
                    "targetEntities": [],
                    "thirdPartyDistributionChannels": [],
                },
                "lifecycleState": "PUBLISHED",
                "isReshareDisabledByAuthor": False,
            }
            r2 = requests.post(
                "https://api.linkedin.com/rest/posts",
                headers=headers,
                json=body,
                proxies=proxies,
            )
            r2.raise_for_status()
            post_id = r2.headers.get("x-restli-id", "")
            return [
                TextContent(
                    type="text",
                    text=f"Post created successfully! URN: urn:li:share:{post_id}",
                )
            ]

        elif name == "create_article_post":
            text = arguments.get("text")
            r = requests.get(
                "https://api.linkedin.com/v2/userinfo", headers=headers, proxies=proxies
            )
            r.raise_for_status()
            sub = r.json().get("sub", "")
            author_urn = f"urn:li:person:{sub}"
            body = {
                "author": author_urn,
                "commentary": text,
                "visibility": "PUBLIC",
                "distribution": {
                    "feedDistribution": "MAIN_FEED",
                    "targetEntities": [],
                    "thirdPartyDistributionChannels": [],
                },
                "content": {
                    "article": {
                        "source": arguments.get("articleUrl"),
                        "title": arguments.get("articleTitle"),
                        "description": arguments.get("articleDescription", ""),
                    }
                },
                "lifecycleState": "PUBLISHED",
                "isReshareDisabledByAuthor": False,
            }
            r2 = requests.post(
                "https://api.linkedin.com/rest/posts",
                headers=headers,
                json=body,
                proxies=proxies,
            )
            r2.raise_for_status()
            post_id = r2.headers.get("x-restli-id", "")
            return [
                TextContent(
                    type="text",
                    text=f"Article post created successfully! URN: urn:li:share:{post_id}",
                )
            ]

        elif name == "delete_post":
            post_urn = arguments.get("post_urn")
            encoded_urn = urllib.parse.quote(str(post_urn))
            r = requests.delete(
                f"https://api.linkedin.com/rest/posts/{encoded_urn}",
                headers=headers,
                proxies=proxies,
            )
            r.raise_for_status()
            return [
                TextContent(type="text", text=f"Post {post_urn} deleted successfully!")
            ]

        elif name == "create_comment":
            post_urn = arguments.get("post_urn")
            text = arguments.get("text")
            r = requests.get(
                "https://api.linkedin.com/v2/userinfo", headers=headers, proxies=proxies
            )
            r.raise_for_status()
            sub = r.json().get("sub", "")
            actor_urn = f"urn:li:person:{sub}"

            encoded_urn = urllib.parse.quote(str(post_urn))
            body = {"actor": actor_urn, "object": post_urn, "message": {"text": text}}
            r2 = requests.post(
                f"https://api.linkedin.com/rest/socialActions/{encoded_urn}/comments",
                headers=headers,
                json=body,
                proxies=proxies,
            )
            r2.raise_for_status()
            comment_id = r2.headers.get("x-restli-id", "")
            return [
                TextContent(
                    type="text", text=f"Comment created successfully! ID: {comment_id}"
                )
            ]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        error_msg = str(e)
        if isinstance(e, requests.HTTPError) and e.response is not None:
            error_msg = f"{e} - {e.response.text}"
        return [TextContent(type="text", text=f"Error: {error_msg}")]


async def main() -> None:
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
