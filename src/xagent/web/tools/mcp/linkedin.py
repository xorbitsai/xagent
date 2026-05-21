import asyncio
import mimetypes
import os
import urllib.parse
from pathlib import Path

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

app = Server("linkedin-mcp")


POSTS_URL = "https://api.linkedin.com/rest/posts"
USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
IMAGES_URL = "https://api.linkedin.com/rest/images"


def _post_urn_from_header(post_id: str) -> str:
    if post_id.startswith("urn:li:"):
        return post_id
    return f"urn:li:share:{post_id}"


def _build_post_body(author_urn: str, text: str) -> dict:
    return {
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


def _sanitize_post_text(text: str) -> str:
    # Escape ASCII parentheses before publishing to avoid LinkedIn truncating
    # automated posts at the first bracket in some publishing flows.
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _get_author_urn(headers: dict, proxies: dict | None) -> str:
    r = requests.get(USERINFO_URL, headers=headers, proxies=proxies)
    r.raise_for_status()
    sub = r.json().get("sub", "")
    return f"urn:li:person:{sub}"


def _allowed_image_dirs() -> list[Path]:
    raw_dirs = os.environ.get("XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS", "")
    if not raw_dirs.strip():
        return [Path.cwd().resolve()]
    return [
        Path(raw_dir).expanduser().resolve()
        for raw_dir in raw_dirs.split(",")
        if raw_dir.strip()
    ]


def _resolve_allowed_image_path(image_path: str) -> Path:
    local_path = Path(image_path).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    local_path = local_path.resolve()

    if not local_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    allowed_dirs = _allowed_image_dirs()
    for allowed_dir in allowed_dirs:
        if local_path == allowed_dir or local_path.is_relative_to(allowed_dir):
            return local_path

    allowed_dirs_str = ", ".join(str(path) for path in allowed_dirs)
    raise PermissionError(
        f"Image path {image_path} is outside allowed directories: {allowed_dirs_str}"
    )


def _upload_image(
    image_path: str, author_urn: str, headers: dict, proxies: dict | None
) -> str:
    local_path = _resolve_allowed_image_path(image_path)

    initialize_body = {
        "initializeUploadRequest": {
            "owner": author_urn,
        }
    }
    init_response = requests.post(
        IMAGES_URL,
        params={"action": "initializeUpload"},
        headers=headers,
        json=initialize_body,
        proxies=proxies,
    )
    init_response.raise_for_status()

    upload_value = init_response.json().get("value", {})
    upload_url = upload_value.get("uploadUrl")
    image_urn = upload_value.get("image")
    if not upload_url or not isinstance(image_urn, str):
        raise ValueError("LinkedIn image upload initialization returned no upload URL")

    mime_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    upload_headers = {
        "Authorization": headers["Authorization"],
        "Content-Type": mime_type,
    }
    with local_path.open("rb") as image_file:
        upload_response = requests.put(
            upload_url,
            headers=upload_headers,
            data=image_file,
            proxies=proxies,
        )
    upload_response.raise_for_status()
    return image_urn


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_profile",
            description="Get your LinkedIn profile",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="create_post",
            description=(
                "Publish a LinkedIn post. Supports text-only posts and posts with "
                "one image. When the user asks to publish a generated image, visual, "
                "graphic, poster, or image attachment, include image_path."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content of the post",
                    },
                    "image_path": {
                        "type": "string",
                        "description": (
                            "Optional absolute local path of an image to upload and "
                            "attach to the post"
                        ),
                    },
                    "altText": {
                        "type": "string",
                        "description": "Optional alt text for the image",
                    },
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
            r = requests.get(USERINFO_URL, headers=headers, proxies=proxies)
            r.raise_for_status()
            return [TextContent(type="text", text=r.text)]

        elif name == "create_post":
            text = _sanitize_post_text(str(arguments.get("text") or ""))
            image_path = (arguments.get("image_path") or "").strip()
            alt_text = str(arguments.get("altText") or "")
            author_urn = _get_author_urn(headers, proxies)
            body = _build_post_body(author_urn, text)
            if image_path:
                image_urn = _upload_image(image_path, author_urn, headers, proxies)
                body["content"] = {
                    "media": {
                        "id": image_urn,
                        "altText": alt_text,
                    }
                }
            r2 = requests.post(
                POSTS_URL,
                headers=headers,
                json=body,
                proxies=proxies,
            )
            r2.raise_for_status()
            post_id = r2.headers.get("x-restli-id", "")
            return [
                TextContent(
                    type="text",
                    text=f"Post created successfully! URN: {_post_urn_from_header(post_id)}",
                )
            ]

        elif name == "create_article_post":
            text = _sanitize_post_text(str(arguments.get("text") or ""))
            author_urn = _get_author_urn(headers, proxies)
            body = _build_post_body(author_urn, text)
            body["content"] = {
                "article": {
                    "source": arguments.get("articleUrl"),
                    "title": _sanitize_post_text(
                        str(arguments.get("articleTitle") or "")
                    ),
                    "description": _sanitize_post_text(
                        str(arguments.get("articleDescription") or "")
                    ),
                }
            }
            r2 = requests.post(
                POSTS_URL,
                headers=headers,
                json=body,
                proxies=proxies,
            )
            r2.raise_for_status()
            post_id = r2.headers.get("x-restli-id", "")
            return [
                TextContent(
                    type="text",
                    text=f"Article post created successfully! URN: {_post_urn_from_header(post_id)}",
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
            text = _sanitize_post_text(str(arguments.get("text") or ""))
            r = requests.get(USERINFO_URL, headers=headers, proxies=proxies)
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
