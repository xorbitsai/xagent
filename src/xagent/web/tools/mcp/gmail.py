import base64
import json
import logging
import mimetypes
import os
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gmail-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("gmail-mcp")

# Gmail's own limit on the combined size of a message's attachments (raw
# bytes, before base64 encoding inflates them by ~33%). Enforced here so a
# too-large attachment fails with a clear message instead of an opaque error
# from the send API.
_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024


def _allowed_file_dirs() -> list[Path]:
    raw_dirs = os.environ.get("XAGENT_GMAIL_FILE_ALLOWED_DIRS", "")
    if not raw_dirs.strip():
        return [Path.cwd().resolve()]
    return [
        Path(stripped).expanduser().resolve()
        for raw_dir in raw_dirs.split(",")
        if (stripped := raw_dir.strip())
    ]


def _resolve_allowed_file_path(file_path: str) -> Path:
    """Restrict gmail_send_messages attachments to files under an
    allowlisted directory, the same defense slack_upload_file and the
    LinkedIn connector's image upload use — without it an agent could be
    tricked into exfiltrating arbitrary host files through this tool."""
    local_path = Path(file_path).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    local_path = local_path.resolve()

    if not local_path.is_file():
        raise FileNotFoundError(f"Attachment not found: {file_path}")

    allowed_dirs = _allowed_file_dirs()
    for allowed_dir in allowed_dirs:
        if local_path.is_relative_to(allowed_dir):
            return local_path

    # The absolute host path is deliberately kept out of the raised message:
    # it reaches the caller/LLM unfiltered via the error payload below, and
    # host filesystem layout has no business in a model transcript. Full
    # detail (including the allowed directories) is logged server-side.
    logger.warning(
        "Rejected gmail attachment path %s outside allowed directories: %s",
        local_path,
        ", ".join(str(path) for path in allowed_dirs),
    )
    raise PermissionError(
        "attachment path is outside the allowed directories; ask the user "
        "for a file inside the task workspace or another allowed location"
    )


def _resolve_message_attachments(msg_data: dict) -> list[tuple[str, bytes]]:
    """Resolve+read every attachment for one message, enforcing the
    allowlist, empty-file, and total-size checks. Reading (not just
    stat-ing) here — and only once — is what lets the caller pre-validate
    every message's attachments before sending any of them."""
    attachments: list[tuple[str, bytes]] = []
    total_bytes = 0
    for attachment_path in msg_data.get("attachments") or []:
        local_path = _resolve_allowed_file_path(attachment_path)
        with local_path.open("rb") as fh:
            data = fh.read()
        if not data:
            raise ValueError(f"Attachment is empty: {attachment_path}")
        total_bytes += len(data)
        if total_bytes > _MAX_ATTACHMENT_BYTES:
            raise ValueError(
                f"Attachments for '{msg_data.get('to', '')}' exceed the "
                f"{_MAX_ATTACHMENT_BYTES // (1024 * 1024)}MB limit."
            )
        attachments.append((local_path.name, data))
    return attachments


def get_gmail_service() -> Any:
    token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

    if not token:
        raise ValueError("GOOGLE_ACCESS_TOKEN environment variable is missing")

    creds_kwargs = {"token": token}
    if refresh_token and client_id and client_secret:
        creds_kwargs.update(
            {
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )

    credentials = Credentials(**creds_kwargs)
    return build("gmail", "v1", credentials=credentials)


@mcp.tool()
def gmail_search_messages(
    query: str = "", label_ids: list[str] | None = None, max_results: int = 10
) -> str:
    """
    Search and list Gmail messages with optional query and label filters.
    Use query parameter for Gmail search syntax (e.g. 'is:unread', 'from:example@test.com').
    """
    try:
        service = get_gmail_service()
        kwargs = {"userId": "me", "maxResults": max_results}
        if query:
            kwargs["q"] = query
        if label_ids:
            kwargs["labelIds"] = label_ids

        results = service.users().messages().list(**kwargs).execute()
        messages = results.get("messages", [])

        if not messages:
            return json.dumps({"status": "success", "messages": []})

        message_details = []
        errors = []

        def callback(request_id: Any, response: Any, exception: Any) -> None:
            if exception is not None:
                errors.append(str(exception))
            else:
                headers = response.get("payload", {}).get("headers", [])
                subject = next(
                    (h["value"] for h in headers if h["name"].lower() == "subject"),
                    "No Subject",
                )
                sender = next(
                    (h["value"] for h in headers if h["name"].lower() == "from"),
                    "Unknown",
                )
                message_details.append(
                    {
                        "id": response["id"],
                        "threadId": response.get("threadId", ""),
                        "snippet": response.get("snippet", ""),
                        "subject": subject,
                        "from": sender,
                    }
                )

        # Utilize Google API batch requests to prevent LLM tool execution timeouts
        batch = service.new_batch_http_request(callback=callback)
        for msg in messages:
            batch.add(
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=msg["id"],
                    format="metadata",
                    metadataHeaders=["Subject", "From"],
                )
            )

        batch.execute()

        if errors:
            logger.error(f"Batch errors: {errors}")

        return json.dumps(
            {
                "status": "success",
                "messages": message_details,
                "errors": errors if errors else None,
            }
        )

    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def gmail_read_threads(thread_ids: list[str]) -> str:
    """
    Read one or more Gmail threads by ID.
    """
    try:
        service = get_gmail_service()
        threads_data = []
        errors = []

        def callback(request_id: Any, response: Any, exception: Any) -> None:
            if exception is not None:
                errors.append(str(exception))
            else:
                threads_data.append(response)

        batch = service.new_batch_http_request(callback=callback)
        for t_id in thread_ids:
            batch.add(service.users().threads().get(userId="me", id=t_id))

        batch.execute()

        return json.dumps(
            {
                "status": "success",
                "threads": threads_data,
                "errors": errors if errors else None,
            }
        )
    except Exception as e:
        logger.error(f"Error reading threads: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def gmail_send_messages(messages: list[dict], action: str = "draft") -> str:
    """
    Send multiple Gmail messages or save them as drafts. Calling this tool triggers an interactive user confirmation in the UI to choose "Save to drafts" or "Send" before any message is sent.
    messages should be a list of dicts with 'to', 'subject', 'body', and optionally 'cc', 'bcc', 'attachments'.
    action can be "send" or "draft".

    'attachments' is a list of local file paths (e.g. a file exported to the
    task workspace) to attach to that message — each path must be inside an
    allowed directory (automatically scoped to the current task workspace),
    the same restriction slack_upload_file uses. If you tell the recipient a
    file is attached, you MUST list it here: writing "see attached" in the
    body does not attach anything by itself, and this tool has no other way
    to send a file. Each successful result includes an 'attachments' list of
    what was actually attached (filename + byte size) — check that before
    telling the user a file was sent, don't assume it from the request alone.
    """
    try:
        # Resolve every message's attachments before sending any message, so
        # a bad attachment path on message 2 can't leave message 1 already
        # sent while message 2 silently fails partway through the batch.
        resolved_attachments = [_resolve_message_attachments(msg) for msg in messages]

        service = get_gmail_service()
        results = []

        for msg_data, attachments in zip(messages, resolved_attachments):
            message = EmailMessage()
            message.set_content(msg_data.get("body", ""))
            message["To"] = msg_data.get("to", "")
            message["From"] = "me"
            message["Subject"] = msg_data.get("subject", "")

            if "cc" in msg_data:
                message["Cc"] = msg_data["cc"]
            if "bcc" in msg_data:
                message["Bcc"] = msg_data["bcc"]

            attached_summary: list[dict[str, Any]] = []
            for filename, data in attachments:
                guessed_type, _ = mimetypes.guess_type(filename)
                maintype, _, subtype = (
                    guessed_type or "application/octet-stream"
                ).partition("/")
                message.add_attachment(
                    data,
                    maintype=maintype,
                    subtype=subtype or "octet-stream",
                    filename=filename,
                )
                attached_summary.append({"filename": filename, "size": len(data)})

            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            create_message = {"raw": encoded_message}

            if action == "send":
                sent_message = (
                    service.users()
                    .messages()
                    .send(userId="me", body=create_message)
                    .execute()
                )
                results.append(
                    {
                        "status": "sent",
                        "id": sent_message["id"],
                        "attachments": attached_summary,
                    }
                )
            else:
                draft = (
                    service.users()
                    .drafts()
                    .create(userId="me", body={"message": create_message})
                    .execute()
                )
                results.append(
                    {
                        "status": "drafted",
                        "id": draft["id"],
                        "attachments": attached_summary,
                    }
                )

        return json.dumps({"status": "success", "results": results})
    except Exception as e:
        logger.error(f"Error sending/drafting messages: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def gmail_manage_labels(
    action: str,
    label_id: str | None = None,
    name: str | None = None,
    message_id: str | None = None,
    add_label_ids: list[str] | None = None,
    remove_label_ids: list[str] | None = None,
) -> str:
    """
    Manage Gmail labels: list, get, create, update, delete labels, or apply/remove labels on messages.
    Use this to organize Gmail by managing labels and applying them to emails.
    action must be one of: 'list', 'get', 'create', 'update', 'delete', 'modify_message'.
    """
    try:
        service = get_gmail_service()

        if action == "list":
            results = service.users().labels().list(userId="me").execute()
            return json.dumps(
                {"status": "success", "labels": results.get("labels", [])}
            )

        elif action == "get":
            if not label_id:
                raise ValueError("label_id is required for 'get' action")
            result = service.users().labels().get(userId="me", id=label_id).execute()
            return json.dumps({"status": "success", "label": result})

        elif action == "create":
            if not name:
                raise ValueError("name is required for 'create' action")
            label_object = {
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            }
            result = (
                service.users()
                .labels()
                .create(userId="me", body=label_object)
                .execute()
            )
            return json.dumps({"status": "success", "label": result})

        elif action == "update":
            if not label_id or not name:
                raise ValueError("label_id and name are required for 'update' action")
            label_object = {"name": name}
            result = (
                service.users()
                .labels()
                .patch(userId="me", id=label_id, body=label_object)
                .execute()
            )
            return json.dumps({"status": "success", "label": result})

        elif action == "delete":
            if not label_id:
                raise ValueError("label_id is required for 'delete' action")
            service.users().labels().delete(userId="me", id=label_id).execute()
            return json.dumps(
                {"status": "success", "message": f"Label {label_id} deleted"}
            )

        elif action == "modify_message":
            if not message_id:
                raise ValueError("message_id is required for 'modify_message' action")
            body = {}
            if add_label_ids:
                body["addLabelIds"] = add_label_ids
            if remove_label_ids:
                body["removeLabelIds"] = remove_label_ids

            result = (
                service.users()
                .messages()
                .modify(userId="me", id=message_id, body=body)
                .execute()
            )
            return json.dumps({"status": "success", "message": result})

        else:
            return json.dumps(
                {"status": "error", "message": f"Unknown action: {action}"}
            )

    except Exception as e:
        logger.error(f"Error managing labels: {e}")
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    mcp.run()
