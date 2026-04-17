import base64
import json
import logging
import os
from email.message import EmailMessage
from typing import Any

from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gmail-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("gmail-mcp")


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
    messages should be a list of dicts with 'to', 'subject', 'body', and optionally 'cc', 'bcc'.
    action can be "send" or "draft".
    """
    try:
        service = get_gmail_service()
        results = []

        for msg_data in messages:
            message = EmailMessage()
            message.set_content(msg_data.get("body", ""))
            message["To"] = msg_data.get("to", "")
            message["From"] = "me"
            message["Subject"] = msg_data.get("subject", "")

            if "cc" in msg_data:
                message["Cc"] = msg_data["cc"]
            if "bcc" in msg_data:
                message["Bcc"] = msg_data["bcc"]

            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            create_message = {"raw": encoded_message}

            if action == "send":
                sent_message = (
                    service.users()
                    .messages()
                    .send(userId="me", body=create_message)
                    .execute()
                )
                results.append({"status": "sent", "id": sent_message["id"]})
            else:
                draft = (
                    service.users()
                    .drafts()
                    .create(userId="me", body={"message": create_message})
                    .execute()
                )
                results.append({"status": "drafted", "id": draft["id"]})

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
