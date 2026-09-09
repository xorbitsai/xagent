"""WhatsApp Business Platform (Cloud API) MCP connector.

Runs against the Meta Graph API with the user's Meta OAuth token, the same
way the Facebook Pages and Instagram connectors do; the token is injected as
``META_ACCESS_TOKEN`` and must carry the ``whatsapp_business_management`` and
``whatsapp_business_messaging`` scopes (plus ``business_management`` so the
user's businesses -- and through them their WhatsApp Business Accounts -- can
be discovered from ``/me/businesses``).

Discovery goes business -> WhatsApp Business Account (WABA) -> phone number.
Messages are sent from a *phone number id* (not the display number), so the
usual flow is ``whatsapp_list_business_accounts`` ->
``whatsapp_list_phone_numbers`` -> one of the ``whatsapp_send_*`` tools.
"""

import logging
import re
from typing import Any
from urllib.parse import quote

from mcp.server.fastmcp import FastMCP

from . import meta_graph
from .meta_graph import (
    GraphAPIError,
)
from .meta_graph import bounded_limit as _bounded_limit
from .meta_graph import error_response as _error
from .meta_graph import graph_error_response as _graph_error
from .meta_graph import graph_request as _graph_request
from .meta_graph import is_public_image_url as _is_public_http_url
from .meta_graph import redact_secrets as _redact_secrets
from .meta_graph import success_response as _success
from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whatsapp-mcp")

setup_proxy_env()

mcp = FastMCP("whatsapp-mcp")
requests = meta_graph.requests  # exposed for test monkeypatching

MESSAGING_PRODUCT = "whatsapp"

WABA_FIELDS = (
    "id,name,currency,timezone_id,message_template_namespace,account_review_status"
)
# One request: each business is expanded with the WABAs it owns and the ones
# shared with it as a client, instead of two extra round trips per business.
# Each nested edge gets its own .limit() modifier -- Graph defaults every
# edge (including nested ones) to a 25-item page, and a business or agency
# account can easily own more WABAs than that. 100 matches this file's
# highest allowed page size elsewhere (_bounded_limit's ceiling) without
# adding a second round trip.
BUSINESS_FIELDS = (
    "id,name,"
    f"owned_whatsapp_business_accounts.limit(100){{{WABA_FIELDS}}},"
    f"client_whatsapp_business_accounts.limit(100){{{WABA_FIELDS}}}"
)
PHONE_NUMBER_FIELDS = (
    "id,display_phone_number,verified_name,quality_rating,"
    "code_verification_status,name_status,platform_type"
)
BUSINESS_PROFILE_FIELDS = (
    "about,address,description,email,profile_picture_url,websites,vertical"
)
TEMPLATE_FIELDS = "id,name,status,category,language,components,quality_score"

# Cloud API limits (per Meta's messages reference).
MAX_TEXT_BODY_CHARS = 4096
MAX_MEDIA_CAPTION_CHARS = 1024
MIN_RECIPIENT_DIGITS = 5
MAX_RECIPIENT_DIGITS = 15

TEMPLATE_STATUSES = frozenset(
    {
        "APPROVED",
        "PENDING",
        "REJECTED",
        "PAUSED",
        "DISABLED",
        "IN_APPEAL",
        "PENDING_DELETION",
        "DELETED",
        "LIMIT_EXCEEDED",
    }
)

# Media types the Cloud API accepts by public link. "sticker" is deliberately
# left out: it needs a WebP of an exact size and no caption, and is rarely
# what an agent means by "send a file".
MEDIA_TYPES = frozenset({"image", "video", "audio", "document"})
CAPTIONLESS_MEDIA_TYPES = frozenset({"audio"})

# Actionable hints for the Cloud API error codes an agent is most likely to
# hit when sending. The raw error is still returned in full; this only adds
# a "hint" so the model doesn't have to know Meta's error catalogue. Codes
# from Meta's "Cloud API Error Codes" reference.
_SEND_ERROR_HINTS: dict[int, str] = {
    131047: (
        "More than 24 hours have passed since the recipient last messaged this "
        "number, so free-form messages can't be delivered. Send an approved "
        "message template with whatsapp_send_template_message instead."
    ),
    131026: (
        "The recipient could not receive the message: the number may not be on "
        "WhatsApp, may have blocked this business, or hasn't accepted the "
        "latest WhatsApp terms."
    ),
    131030: (
        "This phone number is still in test mode and can only message "
        "recipients on its allowed list. Add the recipient in Meta's WhatsApp "
        "developer dashboard or register a production phone number."
    ),
    131031: (
        "This business account has been locked by Meta (policy or payment "
        "issue). Check the WhatsApp Manager for details before retrying."
    ),
    131051: (
        "Unsupported message type for this endpoint. Check that media_type and "
        "the payload shape match the Cloud API messages reference."
    ),
    132000: (
        "The template's components do not match its definition: the number of "
        "parameters passed differs from what the approved template expects."
    ),
    132001: (
        "No approved template with this name exists for the given language. "
        "Check whatsapp_list_message_templates for the exact name and "
        "language code (e.g. en_US, not en)."
    ),
    132012: (
        "A template parameter has the wrong format (e.g. a newline, tab, or "
        "more than four consecutive spaces in a body parameter)."
    ),
    133010: (
        "This phone number is not registered with the Cloud API. Register it in "
        "WhatsApp Manager (or via /{phone_number_id}/register) before sending."
    ),
}


def _graph_path(object_id: str, suffix: str | None = None) -> str:
    if not object_id or not str(object_id).strip():
        raise ValueError("object id is required")
    path = f"/{quote(str(object_id).strip(), safe='')}"
    if suffix:
        path = f"{path}/{suffix}"
    return path


def _normalize_recipient(to: str) -> str:
    """Normalize a recipient into the digits-only form the Cloud API takes.

    Accepts an E.164 number with or without a leading ``+`` -- or the ``00``
    international dialing prefix used in its place in many countries (e.g.
    the UK's ``"0044 20 7946 0958"``) -- and tolerates the spaces, dashes,
    dots and parentheses people paste from address books
    (``"+1 (555) 123-4567"`` -> ``"15551234567"``). A WhatsApp ``wa_id`` is
    already digits-only and passes through unchanged. Anything else -- a
    non-digit character, or a length outside E.164's bounds -- is rejected
    up front so a malformed value fails here rather than as a confusing
    Graph API error.
    """
    if not to or not str(to).strip():
        raise ValueError("to (recipient phone number) is required")
    cleaned = re.sub(r"[\s().-]", "", str(to).strip())
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    elif cleaned.startswith("00"):
        cleaned = cleaned[2:]
    if not cleaned.isdigit():
        raise ValueError(
            "to must be a phone number in international format (digits with an "
            "optional leading +), e.g. +15551234567"
        )
    if not MIN_RECIPIENT_DIGITS <= len(cleaned) <= MAX_RECIPIENT_DIGITS:
        raise ValueError(
            f"to must contain between {MIN_RECIPIENT_DIGITS} and "
            f"{MAX_RECIPIENT_DIGITS} digits including the country code"
        )
    return cleaned


def _data_list(result: Any) -> list[dict[str, Any]]:
    items = result.get("data") if isinstance(result, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _first_dict(items: Any) -> dict[str, Any]:
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                return item
    return {}


def _next_link(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    paging = result.get("paging")
    if not isinstance(paging, dict):
        return None
    return paging.get("next")


def _error_code(error: GraphAPIError) -> int | None:
    details = error.details
    error_body = details.get("error") if isinstance(details, dict) else None
    if not isinstance(error_body, dict):
        return None
    code = error_body.get("code")
    return code if isinstance(code, int) else None


def _send_error(error: GraphAPIError) -> str:
    """graph_error_response plus an actionable hint for well-known send codes.

    _error_code only ever returns a (non-None) key present in
    _SEND_ERROR_HINTS when error.details is itself a dict (it has to unwrap
    a nested dict "error" object to read the code at all), so error.details
    is guaranteed to be a dict by the time `hint` is non-None -- no
    fallback branch needed for a non-dict details here.
    """
    code = _error_code(error)
    hint = _SEND_ERROR_HINTS.get(code) if code is not None else None
    if hint is None:
        return _graph_error(error)
    return _error(
        str(error),
        details={**error.details, "hint": hint},
        sensitive_values=error.sensitive_values,
    )


def _normalize_waba(
    waba: dict[str, Any], business: dict[str, Any], relationship: str
) -> dict[str, Any]:
    return {
        "id": waba.get("id"),
        "name": waba.get("name"),
        "currency": waba.get("currency"),
        "timezone_id": waba.get("timezone_id"),
        "message_template_namespace": waba.get("message_template_namespace"),
        "account_review_status": waba.get("account_review_status"),
        "relationship": relationship,
        "business": {"id": business.get("id"), "name": business.get("name")},
    }


def _list_business_accounts() -> tuple[list[dict[str, Any]], str | None]:
    """Returns (accounts, next_link).

    next_link is the outer /me/businesses page's own "next" cursor (a user
    who is a member of more than 100 businesses); it does not cover a single
    business owning more than 100 WABAs, which BUSINESS_FIELDS's nested
    .limit(100) already treats as the practical ceiling for one connector
    call rather than paginating recursively per-business.
    """
    result = _graph_request(
        "GET", "/me/businesses", params={"fields": BUSINESS_FIELDS, "limit": 100}
    )
    accounts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for business in _data_list(result):
        for edge, relationship in (
            ("owned_whatsapp_business_accounts", "owned"),
            ("client_whatsapp_business_accounts", "client"),
        ):
            # The edge is absent (not an empty list) when a business has none.
            for waba in _data_list(business.get(edge)):
                waba_id = str(waba.get("id") or "")
                # The same WABA can be owned by one business and shared with
                # another the user also belongs to; report it once.
                if not waba_id or waba_id in seen:
                    continue
                seen.add(waba_id)
                accounts.append(_normalize_waba(waba, business, relationship))
    return accounts, _next_link(result)


def _send_message(phone_number_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = _graph_request(
        "POST",
        _graph_path(phone_number_id, "messages"),
        json_body={"messaging_product": MESSAGING_PRODUCT, **payload},
    )
    message = _first_dict(result.get("messages") if isinstance(result, dict) else None)
    contact = _first_dict(result.get("contacts") if isinstance(result, dict) else None)
    message_id = message.get("id")
    if not message_id:
        raise ValueError(
            f"WhatsApp did not return a message id; response: {_redact_secrets(result)}"
        )
    summary: dict[str, Any] = {
        "message_id": message_id,
        "recipient": contact.get("input"),
        "wa_id": contact.get("wa_id"),
    }
    # Only present when Meta holds the message (e.g. held_for_quality_assessment);
    # surfaced so the agent can tell "accepted" from "queued for review".
    if message.get("message_status"):
        summary["message_status"] = message.get("message_status")
    return summary


@mcp.tool()
def whatsapp_auth_status() -> str:
    """Check whether the injected Meta access token is usable."""
    try:
        me = _graph_request("GET", "/me", params={"fields": "id,name,email"})
        return _success(
            authenticated=True,
            user={
                "id": me.get("id"),
                "name": me.get("name"),
                "email": me.get("email"),
            },
        )
    except GraphAPIError as e:
        logger.error("Error checking WhatsApp auth status: %s", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error checking WhatsApp auth status: %s", e)
        return _error(str(e))


@mcp.tool()
def whatsapp_list_business_accounts() -> str:
    """List the WhatsApp Business Accounts (WABAs) reachable from the connected
    Meta user, across every business they belong to.

    Each entry carries the owning business and whether the WABA is "owned" by
    that business or shared with it as a "client" account. Use the WABA id
    with whatsapp_list_phone_numbers and whatsapp_list_message_templates.
    next_link is non-null only if the user belongs to more than 100
    businesses -- WABAs of a single business beyond the first 100 aren't
    signaled here (see BUSINESS_FIELDS).
    """
    try:
        accounts, next_link = _list_business_accounts()
        return _success(accounts=accounts, next_link=next_link)
    except GraphAPIError as e:
        logger.error("Error listing WhatsApp Business Accounts: %s", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing WhatsApp Business Accounts: %s", e)
        return _error(str(e))


@mcp.tool()
def whatsapp_list_phone_numbers(waba_id: str) -> str:
    """List the phone numbers registered under a WhatsApp Business Account.

    Messages are sent from a phone number's *id* (the "id" field here), not
    from its display number. quality_rating (GREEN/YELLOW/RED) and
    code_verification_status tell you whether a number is healthy and
    verified before you send from it.
    """
    try:
        result = _graph_request(
            "GET",
            _graph_path(waba_id, "phone_numbers"),
            params={"fields": PHONE_NUMBER_FIELDS},
        )
        return _success(phone_numbers=_data_list(result), next_link=_next_link(result))
    except GraphAPIError as e:
        logger.error("Error listing WhatsApp phone numbers for %s: %s", waba_id, e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing WhatsApp phone numbers for %s: %s", waba_id, e)
        return _error(str(e))


@mcp.tool()
def whatsapp_get_business_profile(phone_number_id: str) -> str:
    """Get the public WhatsApp Business profile (about, address, description,
    email, websites, vertical, profile picture) shown for a phone number."""
    try:
        result = _graph_request(
            "GET",
            _graph_path(phone_number_id, "whatsapp_business_profile"),
            params={"fields": BUSINESS_PROFILE_FIELDS},
        )
        profiles = _data_list(result)
        return _success(profile=profiles[0] if profiles else {})
    except GraphAPIError as e:
        logger.error(
            "Error getting WhatsApp business profile for %s: %s", phone_number_id, e
        )
        return _graph_error(e)
    except Exception as e:
        logger.error(
            "Error getting WhatsApp business profile for %s: %s", phone_number_id, e
        )
        return _error(str(e))


@mcp.tool()
def whatsapp_list_message_templates(
    waba_id: str,
    status: str | None = None,
    limit: int = 25,
) -> str:
    """List message templates defined on a WhatsApp Business Account.

    Templates are the only way to start a conversation or message someone
    outside the 24-hour customer service window. Filter with status
    (e.g. "APPROVED" -- only approved templates can be sent). Each template's
    "components" shows which header/body/button parameters
    whatsapp_send_template_message must fill in, and "language" is the exact
    code to pass as language_code.
    """
    try:
        params: dict[str, Any] = {
            "fields": TEMPLATE_FIELDS,
            "limit": _bounded_limit(limit),
        }
        if status is not None and status.strip():
            normalized_status = status.strip().upper()
            if normalized_status not in TEMPLATE_STATUSES:
                raise ValueError(
                    "status must be one of: " + ", ".join(sorted(TEMPLATE_STATUSES))
                )
            params["status"] = normalized_status
        result = _graph_request(
            "GET", _graph_path(waba_id, "message_templates"), params=params
        )
        return _success(templates=_data_list(result), next_link=_next_link(result))
    except GraphAPIError as e:
        logger.error("Error listing WhatsApp templates for %s: %s", waba_id, e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing WhatsApp templates for %s: %s", waba_id, e)
        return _error(str(e))


@mcp.tool()
def whatsapp_send_text_message(
    phone_number_id: str,
    to: str,
    body: str,
    preview_url: bool = False,
    reply_to_message_id: str | None = None,
) -> str:
    """Send a free-form text message from a business phone number.

    Free-form messages are only delivered inside the 24-hour customer service
    window that opens when the recipient last messaged this number; outside it
    use whatsapp_send_template_message. `to` is the recipient's phone number
    in international format (e.g. +15551234567). preview_url renders a link
    preview for the first URL in the body. reply_to_message_id quotes an
    earlier message (a "wamid..." id) so the reply threads under it.
    """
    try:
        if not body or not body.strip():
            raise ValueError("body is required")
        if len(body) > MAX_TEXT_BODY_CHARS:
            raise ValueError(f"body must be at most {MAX_TEXT_BODY_CHARS} characters")
        payload: dict[str, Any] = {
            "recipient_type": "individual",
            "to": _normalize_recipient(to),
            "type": "text",
            "text": {"preview_url": bool(preview_url), "body": body},
        }
        if reply_to_message_id and reply_to_message_id.strip():
            payload["context"] = {"message_id": reply_to_message_id.strip()}
        return _success(**_send_message(phone_number_id, payload))
    except GraphAPIError as e:
        logger.error("Error sending WhatsApp text from %s: %s", phone_number_id, e)
        return _send_error(e)
    except Exception as e:
        logger.error("Error sending WhatsApp text from %s: %s", phone_number_id, e)
        return _error(str(e))


@mcp.tool()
def whatsapp_send_template_message(
    phone_number_id: str,
    to: str,
    template_name: str,
    language_code: str,
    components: list[dict[str, Any]] | None = None,
) -> str:
    """Send an approved message template from a business phone number.

    Templates work at any time, including to start a conversation. Find the
    template with whatsapp_list_message_templates; language_code must match
    the template's "language" exactly (e.g. "en_US"). components fills the
    template's placeholders using the Cloud API shape, e.g.
    [{"type": "body", "parameters": [{"type": "text", "text": "Alice"}]}];
    omit it for templates with no variables.
    """
    try:
        if not template_name or not template_name.strip():
            raise ValueError("template_name is required")
        if not language_code or not language_code.strip():
            raise ValueError("language_code is required")
        template: dict[str, Any] = {
            "name": template_name.strip(),
            "language": {"code": language_code.strip()},
        }
        # No isinstance validation here: `components`' type annotation is
        # enforced by FastMCP/pydantic at the MCP transport boundary, the
        # only way this tool is ever invoked in production (this module runs
        # as its own MCP server process) -- unlike the Graph API response
        # parsing elsewhere in this file, which validates because that data
        # is untrusted and external.
        if components:
            template["components"] = components
        payload: dict[str, Any] = {
            "recipient_type": "individual",
            "to": _normalize_recipient(to),
            "type": "template",
            "template": template,
        }
        return _success(**_send_message(phone_number_id, payload))
    except GraphAPIError as e:
        logger.error(
            "Error sending WhatsApp template %s from %s: %s",
            template_name,
            phone_number_id,
            e,
        )
        return _send_error(e)
    except Exception as e:
        logger.error(
            "Error sending WhatsApp template %s from %s: %s",
            template_name,
            phone_number_id,
            e,
        )
        return _error(str(e))


@mcp.tool()
def whatsapp_send_media_message(
    phone_number_id: str,
    to: str,
    media_type: str,
    media_url: str,
    caption: str | None = None,
    filename: str | None = None,
) -> str:
    """Send an image, video, audio clip, or document by public URL.

    media_type is one of image, video, audio, document. media_url must be a
    publicly reachable http(s) URL that Meta's servers can download. caption
    applies to image, video and document (not audio); filename is the name a
    document shows with (documents only). Like text messages, media is only
    delivered inside the 24-hour customer service window.
    """
    try:
        normalized_type = (media_type or "").strip().lower()
        if normalized_type not in MEDIA_TYPES:
            raise ValueError(
                "media_type must be one of: " + ", ".join(sorted(MEDIA_TYPES))
            )
        media_url = (media_url or "").strip()
        if not _is_public_http_url(media_url):
            raise ValueError("media_url must be a public http or https URL")
        media: dict[str, Any] = {"link": media_url}
        if caption is not None and caption.strip():
            if normalized_type in CAPTIONLESS_MEDIA_TYPES:
                raise ValueError(f"{normalized_type} messages do not support a caption")
            if len(caption) > MAX_MEDIA_CAPTION_CHARS:
                raise ValueError(
                    f"caption must be at most {MAX_MEDIA_CAPTION_CHARS} characters"
                )
            media["caption"] = caption
        if filename is not None and filename.strip():
            if normalized_type != "document":
                raise ValueError("filename is only supported for document messages")
            media["filename"] = filename.strip()
        payload: dict[str, Any] = {
            "recipient_type": "individual",
            "to": _normalize_recipient(to),
            "type": normalized_type,
            normalized_type: media,
        }
        return _success(**_send_message(phone_number_id, payload))
    except GraphAPIError as e:
        logger.error(
            "Error sending WhatsApp %s from %s: %s", media_type, phone_number_id, e
        )
        return _send_error(e)
    except Exception as e:
        logger.error(
            "Error sending WhatsApp %s from %s: %s", media_type, phone_number_id, e
        )
        return _error(str(e))


@mcp.tool()
def whatsapp_mark_message_read(phone_number_id: str, message_id: str) -> str:
    """Mark an inbound message as read (shows blue ticks to the sender).

    message_id is the "wamid..." id of a message the customer sent to this
    phone number, as delivered by the WhatsApp webhook.
    """
    try:
        if not message_id or not message_id.strip():
            raise ValueError("message_id is required")
        result = _graph_request(
            "POST",
            _graph_path(phone_number_id, "messages"),
            json_body={
                "messaging_product": MESSAGING_PRODUCT,
                "status": "read",
                "message_id": message_id.strip(),
            },
        )
        return _success(
            message_id=message_id.strip(),
            marked_read=bool(result.get("success"))
            if isinstance(result, dict)
            else False,
        )
    except GraphAPIError as e:
        logger.error(
            "Error marking WhatsApp message %s read on %s: %s",
            message_id,
            phone_number_id,
            e,
        )
        return _graph_error(e)
    except Exception as e:
        logger.error(
            "Error marking WhatsApp message %s read on %s: %s",
            message_id,
            phone_number_id,
            e,
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
