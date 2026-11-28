import asyncio
import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import BinaryIO

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from PIL import Image

from ....core.tools.core.web_content import get_trusted_proxy_url
from ....core.utils import public_requests as requests
from .utils import allowed_dirs_from_env

app = Server("linkedin-mcp")
logger = logging.getLogger("linkedin-mcp")


POSTS_URL = "https://api.linkedin.com/rest/posts"
USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
IMAGES_URL = "https://api.linkedin.com/rest/images"
_UPLOAD_ALLOWED_DIRS_ENV_VAR = "XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS"
_IMAGE_MIME_TYPES = {"JPEG": "image/jpeg", "GIF": "image/gif", "PNG": "image/png"}
_IMAGE_PIXEL_LIMIT = 36_152_320
_GIF_FRAME_LIMIT = 250
# A definitive rejection can be reported as an MCP error. A timeout, 5xx, or
# ambiguous status after dispatch may still have committed the write.
_WRITE_REJECTION_STATUSES = frozenset({400, 401, 403, 404, 413, 415, 422})


class _LinkedInToolError(ValueError):
    """A connector error whose message is safe to return to the caller."""


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
    try:
        return allowed_dirs_from_env(_UPLOAD_ALLOWED_DIRS_ENV_VAR)
    except ValueError as exc:
        logger.warning("Invalid LinkedIn upload directory configuration: %s", exc)
        raise _LinkedInToolError("Upload directory configuration is invalid") from None


def _resolve_allowed_image_path(image_path: str) -> Path:
    local_path = Path(image_path).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    local_path = local_path.resolve()

    if not local_path.is_file():
        raise _LinkedInToolError(f"Image file not found: {image_path}")

    allowed_dirs = _allowed_image_dirs()
    for allowed_dir in allowed_dirs:
        if local_path == allowed_dir or local_path.is_relative_to(allowed_dir):
            return local_path

    raise _LinkedInToolError(f"Image path {image_path} is outside allowed directories")


def _check_image_dimensions(image: Image.Image) -> None:
    if image.width * image.height >= _IMAGE_PIXEL_LIMIT:
        raise _LinkedInToolError(
            "LinkedIn images must have fewer than 36,152,320 pixels"
        )


def _validate_image(image_file: BinaryIO) -> str:
    try:
        with Image.open(image_file, formats=list(_IMAGE_MIME_TYPES)) as image:
            image_format = image.format
            if image_format is None or image_format not in _IMAGE_MIME_TYPES:
                raise _LinkedInToolError("LinkedIn requires a JPEG, PNG, or GIF image")
            _check_image_dimensions(image)
            image.load()
            if image_format == "GIF":
                for frame in range(1, _GIF_FRAME_LIMIT + 1):
                    try:
                        image.seek(frame)
                    except EOFError:
                        break
                    if frame == _GIF_FRAME_LIMIT:
                        raise _LinkedInToolError(
                            "LinkedIn GIFs can have at most 250 frames"
                        )
                    _check_image_dimensions(image)
                    image.load()
    except (OSError, SyntaxError, EOFError) as exc:
        raise _LinkedInToolError(
            "LinkedIn requires a valid JPEG, PNG, or GIF image"
        ) from exc
    except Image.DecompressionBombError as exc:
        raise _LinkedInToolError(
            "LinkedIn images must have fewer than 36,152,320 pixels"
        ) from exc
    image_file.seek(0)
    return _IMAGE_MIME_TYPES[image_format]


def _upload_image(
    image_path: str, author_urn: str, headers: dict, proxies: dict | None
) -> str:
    local_path = _resolve_allowed_image_path(image_path)
    with local_path.open("rb") as image_file:
        mime_type = _validate_image(image_file)
        initialize_body = {"initializeUploadRequest": {"owner": author_urn}}
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
        if not upload_url or not isinstance(image_urn, str) or not image_urn:
            raise _LinkedInToolError(
                "LinkedIn image upload initialization returned no upload URL or image URN"
            )

        upload_headers = {
            "Authorization": headers["Authorization"],
            "Content-Type": mime_type,
        }
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
        raise _LinkedInToolError(
            "LINKEDIN_ACCESS_TOKEN environment variable is missing"
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "LinkedIn-Version": "202603",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    write_started = False
    try:
        proxy = get_trusted_proxy_url()
        proxies = {"http": proxy, "https": proxy} if proxy else None
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
            write_started = True
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
            write_started = True
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
            write_started = True
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
            write_started = True
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
            raise _LinkedInToolError(f"Unknown tool: {name}")
    except _LinkedInToolError:
        raise
    except Exception as exc:
        status = (
            exc.response.status_code
            if isinstance(exc, requests.HTTPError) and exc.response is not None
            else None
        )
        logger.warning(
            "LinkedIn tool %s failed (%s, status %s)", name, type(exc).__name__, status
        )
        if status is not None:
            detail = f"LinkedIn returned HTTP {status}."
        elif isinstance(exc, requests.Timeout):
            detail = "LinkedIn request timed out."
        else:
            detail = "LinkedIn request failed."
        if write_started and status not in _WRITE_REJECTION_STATUSES:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "indeterminate",
                            "retryable": False,
                            "mutation_may_have_completed": True,
                            "message": f"{detail} Check LinkedIn before retrying.",
                        }
                    ),
                )
            ]
        raise _LinkedInToolError(detail) from exc


async def main() -> None:
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
