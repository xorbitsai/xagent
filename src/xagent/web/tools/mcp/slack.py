import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("slack-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("slack-mcp")

SLACK_BASE_URL = "https://slack.com/api"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_PAGES = 20
MAX_CHANNELS = 1000
# conversations.list is Tier-2 rate-limited (~20 req/min); on a 429 with a
# small Retry-After we wait once and retry rather than failing the page.
MAX_RETRY_AFTER_SECONDS = 30
# Bound on how many distinct threads slack_search_messages will fetch replies
# for in one call — a channel history page can contain many threaded parents,
# and each one is a separate conversations.replies call.
MAX_SEARCH_THREADS = 20

# Slack conversation ids are uppercase alphanumerics prefixed by their
# conversation type: "C" (public channel), "G" (private channel or
# multi-party DM), "D" (1:1 DM). Restricted to those three prefixes — rather
# than any letter — so a user id ("U...") or bot id ("B...") passed by
# mistake (e.g. copied from slack_list_direct_messages' "user" field) is
# rejected up front instead of being misclassified as an already-resolved
# conversation id. Channel *names* are forced lowercase by Slack, so this
# can't misclassify a name as an id either way. Used only by
# _resolve_channel_id, whose conversations.*/reactions.*/files.* endpoints
# genuinely require a conversation id — never by _normalize_channel below.
_SLACK_ID_PATTERN = re.compile(r"^[CGD][A-Z0-9]{5,}$")

# Any Slack object id (channel, user "U...", bot "B..."), for
# _normalize_channel: chat.postMessage accepts a user id directly to open/
# post into a DM, so that id must pass through unchanged rather than being
# misread as a bare channel name and prefixed with "#" (which 404s).
_SLACK_ANY_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{5,}$")


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _headers() -> dict[str, str]:
    access_token = os.environ.get("SLACK_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("SLACK_ACCESS_TOKEN environment variable is missing")
    return {"Authorization": f"Bearer {access_token}"}


def _normalize_channel(channel: str) -> str:
    """Accept a channel id ("C0123456789"), a user id ("U0123456789", for
    posting into a DM), or a name with or without the leading "#"; bare
    names get the "#" prepended so all forms reach the API in a shape it
    accepts."""
    value = channel.strip()
    if value.startswith("#") or _SLACK_ANY_ID_PATTERN.match(value):
        return value
    return f"#{value}"


def _resolve_channel_id(channel: str) -> str:
    """Resolve a channel id, bare name, or "#name" to a real Slack channel id.

    Unlike chat.postMessage (which Slack resolves names for natively), the
    conversations.*/reactions.*/files.* endpoints below require an actual
    channel id — a name has to be looked up via conversations.list first.
    DM and group-DM ids have no name form and always match
    _SLACK_ID_PATTERN, so they pass through unchanged.
    """
    value = channel.strip()
    if _SLACK_ID_PATTERN.match(value):
        return value
    name = value.lstrip("#").lower()
    if not name:
        raise ValueError("channel must not be empty")
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        params: dict[str, Any] = {
            "types": "public_channel,private_channel",
            "exclude_archived": "false",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        result = _request("GET", "conversations.list", params=params)
        for candidate in result.get("channels") or []:
            if str(candidate.get("name") or "").lower() == name:
                return str(candidate.get("id"))
        cursor = (result.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    else:
        # MAX_PAGES exhausted with a cursor still pending: the workspace has
        # more channels than this scan covers, so "not found" is not a firm
        # answer — say so rather than implying an exhaustive search.
        if cursor:
            raise ValueError(
                f"Could not resolve channel '{channel}' to an id after scanning "
                f"the first {MAX_PAGES * 200} channels (more remain) — pass the "
                "channel id directly, or narrow the search via slack_list_channels."
            )
    raise ValueError(
        f"Could not resolve channel '{channel}' to an id — use slack_list_channels "
        "or slack_list_direct_messages to look it up, or pass the channel id "
        "directly."
    )


def _allowed_file_dirs() -> list[Path]:
    raw_dirs = os.environ.get("XAGENT_SLACK_FILE_ALLOWED_DIRS", "")
    if not raw_dirs.strip():
        return [Path.cwd().resolve()]
    return [
        Path(stripped).expanduser().resolve()
        for raw_dir in raw_dirs.split(",")
        if (stripped := raw_dir.strip())
    ]


def _resolve_allowed_file_path(file_path: str) -> Path:
    """Restrict slack_upload_file to files under an allowlisted directory,
    the same defense used by the LinkedIn connector's image upload — without
    it an agent could be tricked into exfiltrating arbitrary host files
    through this tool."""
    local_path = Path(file_path).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    local_path = local_path.resolve()

    if not local_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    allowed_dirs = _allowed_file_dirs()
    for allowed_dir in allowed_dirs:
        # is_relative_to already covers the equality case (a path is
        # relative to itself), so no separate "== allowed_dir" arm is needed.
        if local_path.is_relative_to(allowed_dir):
            return local_path

    # The absolute host path is deliberately kept out of the raised message:
    # it reaches the caller/LLM unfiltered via _error(str(e)), and host
    # filesystem layout has no business in a model transcript. Full detail
    # (including the allowed directories) is logged server-side instead.
    logger.warning(
        "Rejected slack_upload_file path %s outside allowed directories: %s",
        local_path,
        ", ".join(str(path) for path in allowed_dirs),
    )
    raise PermissionError(
        "file_path is outside the allowed upload directories; ask the user "
        "for a file inside the task workspace or another allowed location"
    )


class _SlackAPIError(RuntimeError):
    """Raised by _request for any non-ok Slack response.

    Carries Slack's raw error code as structured state (`.code`) so callers
    can branch on it directly instead of parsing the rendered message
    string — str(e) still equals the code, so every existing `except
    Exception as e: ... str(e)` call site is unaffected.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _SlackNotAMemberError(RuntimeError):
    """Raised by _request_requiring_membership once a not-a-member-style
    failure has been translated into an actionable, user-facing message.

    Distinct from _SlackAPIError so a caller that needs to tell "this is the
    actionable one" apart from any other failure (see slack_search_messages,
    where a later transient error must not mask an earlier actionable one)
    can use isinstance() instead of substring-matching the rendered text.
    """


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call a Slack Web API method.

    Slack always answers with HTTP 200, even on failure — success/failure is
    only signalled by the body's "ok" boolean plus an "error" code, so error
    detection here is on that field, not on raise_for_status().

    Write calls pass json_data rather than params: message text can exceed
    URL length limits, and query strings are commonly logged in plaintext by
    proxies/load balancers, which would leak message content.

    Rate limiting is the one non-200 case worth special handling: on a 429
    with a small Retry-After, wait once and retry instead of failing.
    """
    for attempt in (0, 1):
        response = requests.request(
            method=method,
            url=f"{SLACK_BASE_URL}/{path}",
            headers=_headers(),
            params=params,
            json=json_data,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        if response.status_code == 429 and attempt == 0:
            try:
                retry_after = int(response.headers.get("Retry-After", "0"))
            except ValueError:
                retry_after = 0
            if 0 < retry_after <= MAX_RETRY_AFTER_SECONDS:
                time.sleep(retry_after)
                continue
        break
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if not payload.get("ok"):
        raise _SlackAPIError(payload.get("error") or "Unknown Slack API error")
    return payload


# Slack documents "not_in_channel" for conversations.history/replies,
# chat.postMessage, and files.completeUploadExternal, but conversations.replies
# and reactions.add/remove do NOT document it — they document
# "channel_not_found" instead, which is ambiguous (bad id vs. a real channel
# hidden from a non-member caller). Only conversations.replies and
# reactions.* need this widened set: by the time either is called here,
# _resolve_channel_id has already confirmed the channel exists via
# conversations.list, so a channel_not_found from these specific endpoints is
# far more likely "the bot isn't a member and Slack is hiding it" than "this
# channel doesn't exist".
_HIDDEN_FROM_NON_MEMBER_CODES = frozenset({"not_in_channel", "channel_not_found"})
_DEFAULT_NOT_A_MEMBER_CODES = frozenset({"not_in_channel"})


def _request_requiring_membership(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
    not_a_member_codes: frozenset[str] = _DEFAULT_NOT_A_MEMBER_CODES,
) -> dict[str, Any]:
    """Call a Slack endpoint that requires the bot to already be a channel
    member, and raise an actionable error if it isn't yet.

    This covers conversations.history/replies (regardless of which
    *:history scopes are granted), reactions.add/remove, and
    files.completeUploadExternal — none of them work for a channel the bot
    hasn't joined. chat.postMessage is the one exception for a *public*
    channel (chat:write.public lets a bot post there without joining), but
    can still hit this for a private channel/DM, so slack_post_message uses
    this too.

    not_a_member_codes: which Slack error code(s) this specific endpoint
    actually uses to signal "not a member" — see _HIDDEN_FROM_NON_MEMBER_CODES
    above for the two call sites that need the wider set.

    This deliberately does NOT join the channel on the caller's behalf:
    joining changes the channel's member list, which is visible to everyone
    in it, so the calling agent should check with the user first and only
    then call slack_join_channel — never silently, just because a call
    failed.
    """
    try:
        return _request(method, path, params=params, json_data=json_data)
    except _SlackAPIError as e:
        if e.code not in not_a_member_codes:
            raise
        raise _SlackNotAMemberError(
            f"{e.code}: the bot cannot access this channel — either it "
            "isn't a member, or (for a private channel) the channel itself "
            "is hidden from non-members. Ask the user whether to add the "
            "bot to this channel; if they agree, call slack_join_channel "
            "(only works for a public channel) — for a private channel or "
            "DM, a member needs to run `/invite @<this app's bot name>` "
            "instead."
        ) from None


@mcp.tool()
def slack_join_channel(channel: str) -> str:
    """
    Add the bot to a public channel so it can read, post, react, and upload
    files there.

    Only call this after the user has explicitly confirmed they want the
    bot added to this specific channel — joining changes the channel's
    member list, which is visible to everyone in it, so it should never
    happen silently just because another Slack tool call failed with
    "not_in_channel".
    channel: a channel id, or a channel name (with or without "#").
    This only works for a public channel: Slack does not let a bot join a
    private channel or DM on its own (conversations.join fails with
    method_not_supported_for_channel_type there) — for those, ask a member
    to run `/invite @<this app's bot name>` instead.
    """
    try:
        channel_id = _resolve_channel_id(channel)
        try:
            result = _request(
                "POST", "conversations.join", json_data={"channel": channel_id}
            )
        except _SlackAPIError as e:
            if e.code != "missing_scope":
                raise
            raise RuntimeError(
                "missing_scope: this connector's Slack connection hasn't "
                "been re-authorized with the channels:join permission yet "
                "— ask the user to reconnect the Slack connector, then "
                "retry."
            ) from None
        # conversations.join succeeds (with warning="already_in_channel")
        # even when the bot already had membership, so this distinguishes
        # a fresh join from a no-op re-join rather than reporting both
        # identically.
        already_member = "already_in_channel" in (
            result.get("response_metadata") or {}
        ).get("warnings", [])
        return _success(channel=channel_id, already_member=already_member)
    except Exception as e:
        logger.error(f"Error joining Slack channel {channel}: {e}")
        return _error(str(e))


@mcp.tool()
def slack_list_channels(
    exclude_archived: bool = True,
    name_contains: str = "",
    limit: int = MAX_CHANNELS,
) -> str:
    """
    List public channels in the connected Slack workspace (id, name, is_archived).
    Use this to resolve a channel name to an id before posting.
    name_contains: optional case-insensitive substring filter on the channel
    name — pass it when looking for a specific channel in a large workspace
    instead of listing everything.
    limit: maximum channels to return (default 1000).
    The response includes truncated=true when there were more channels than
    returned (more pages, the limit was hit, or a page failed mid-way — a
    partial list is returned rather than discarded).
    """
    channels: list[dict[str, Any]] = []
    needle = name_contains.strip().lower()
    max_channels = max(1, min(int(limit), MAX_CHANNELS))
    cursor: str | None = None
    truncated = False
    pages_scanned = 0
    try:
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "types": "public_channel",
                # Slack expects a lowercase "true"/"false" string; requests
                # would otherwise serialize the Python bool as "True"/"False",
                # which Slack's parser doesn't recognize as a boolean.
                "exclude_archived": "true" if exclude_archived else "false",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                result = _request("GET", "conversations.list", params=params)
            except Exception as page_exc:
                if not pages_scanned:
                    raise
                # A mid-pagination failure (e.g. a rate limit that outlived
                # the single retry) must not discard the pages already
                # fetched — return the partial list with a marker instead.
                # This can legitimately be an empty list when name_contains
                # filtered out everything scanned so far.
                logger.warning(f"Slack channel pagination stopped early: {page_exc}")
                return _success(channels=channels, truncated=True, error=str(page_exc))
            pages_scanned += 1
            raw_channels = result.get("channels") or []
            for index, channel in enumerate(raw_channels):
                name = str(channel.get("name") or "")
                if needle and needle not in name.lower():
                    continue
                channels.append(
                    {
                        "id": channel.get("id"),
                        "name": channel.get("name"),
                        "is_archived": channel.get("is_archived", False),
                    }
                )
                if len(channels) >= max_channels:
                    # Only truncated if there is actually more data: more raw
                    # results left in this page, or another page pending —
                    # hitting the limit on the very last item of the very
                    # last page is not truncation.
                    more_in_page = index + 1 < len(raw_channels)
                    more_pages = bool(
                        (result.get("response_metadata") or {}).get("next_cursor")
                    )
                    return _success(
                        channels=channels, truncated=more_in_page or more_pages
                    )
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break
        else:
            # MAX_PAGES exhausted with a cursor still pending.
            truncated = bool(cursor)
        return _success(channels=channels, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing Slack channels: {e}")
        return _error(str(e))


@mcp.tool()
def slack_post_message(channel: str, text: str, thread_ts: str = "") -> str:
    """
    Post a message to a Slack channel, DM, or as a threaded reply.
    channel: a channel id (e.g. "C0123456789"), a user id (e.g.
    "U0123456789", to open/post into a 1:1 DM — see
    slack_list_direct_messages' "user" field), or a channel name (e.g.
    "#incidents" or "incidents" — a bare name is normalized to "#incidents").
    text: the message body (plain text or Slack mrkdwn).
    thread_ts: optional parent message "ts" (from slack_get_channel_history,
    slack_search_messages, or a prior slack_post_message result) to reply
    inside that thread instead of posting to the channel's main timeline.
    This connector's chat:write.public scope already lets the bot post into
    any public channel without joining it first — no user confirmation is
    needed for that case. A private channel or DM the bot isn't in still
    fails with not_in_channel; ask a member to `/invite` the bot there.
    """
    try:
        json_data: dict[str, Any] = {
            "channel": _normalize_channel(channel),
            "text": text,
        }
        if thread_ts:
            json_data["thread_ts"] = thread_ts
        result = _request_requiring_membership(
            "POST", "chat.postMessage", json_data=json_data
        )
        return _success(channel=result.get("channel"), ts=result.get("ts"))
    except Exception as e:
        logger.error(f"Error posting Slack message to {channel}: {e}")
        return _error(str(e))


@mcp.tool()
def slack_get_channel_history(
    channel: str,
    limit: int = 50,
    oldest: str = "",
    latest: str = "",
    cursor: str = "",
) -> str:
    """
    Fetch recent messages from a channel, private channel, or DM.
    channel: a channel/DM id, or a channel name (with or without "#") —
    names, including private channels, are resolved to an id internally.
    slack_list_channels only lists public channels; for a private channel
    or DM either pass its id directly, or look it up via
    slack_list_direct_messages (DMs).
    limit: max messages to return in this page (Slack caps at 1000).
    oldest/latest: optional Slack message timestamps (e.g.
    "1710000000.000100") to bound the time range.
    cursor: pass the next_cursor from a previous call's response to fetch
    the next page.
    Each message includes its "ts" (needed for slack_get_thread_replies,
    slack_add_reaction, and replying via slack_post_message's thread_ts) and,
    when it starts a thread, "thread_ts" and "reply_count".
    If the bot isn't a member of the channel yet, you'll get an actionable
    error instead of an empty/opaque failure: check with the user before
    calling slack_join_channel to add the bot (only works for a public
    channel), or ask a member to `/invite` the bot for a private channel or
    DM.
    """
    try:
        channel_id = _resolve_channel_id(channel)
        params: dict[str, Any] = {
            "channel": channel_id,
            "limit": max(1, min(int(limit), 1000)),
        }
        if oldest:
            params["oldest"] = oldest
        if latest:
            params["latest"] = latest
        if cursor:
            params["cursor"] = cursor
        result = _request_requiring_membership(
            "GET", "conversations.history", params=params
        )
        messages = [
            {
                "ts": m.get("ts"),
                "user": m.get("user"),
                "text": m.get("text"),
                "thread_ts": m.get("thread_ts"),
                "reply_count": m.get("reply_count"),
            }
            for m in result.get("messages") or []
        ]
        return _success(
            messages=messages,
            has_more=bool(result.get("has_more")),
            next_cursor=(result.get("response_metadata") or {}).get("next_cursor")
            or "",
        )
    except Exception as e:
        logger.error(f"Error fetching Slack channel history for {channel}: {e}")
        return _error(str(e))


@mcp.tool()
def slack_get_thread_replies(
    channel: str, thread_ts: str, limit: int = 100, cursor: str = ""
) -> str:
    """
    Fetch a page of replies in a message thread (the parent message is
    included as the first result). Defaults to the 100 most recent replies
    per call — pass cursor to page through a longer thread.
    channel: a channel/DM id or name the thread lives in.
    thread_ts: the parent message's "ts".
    cursor: pass the next_cursor from a previous call's response to fetch
    the next page of replies.
    If the bot isn't a member of the channel yet, you'll get an actionable
    error instead of an empty/opaque failure: check with the user before
    calling slack_join_channel to add the bot (only works for a public
    channel), or ask a member to `/invite` the bot for a private channel or
    DM.
    """
    try:
        channel_id = _resolve_channel_id(channel)
        params: dict[str, Any] = {
            "channel": channel_id,
            "ts": thread_ts,
            "limit": max(1, min(int(limit), 1000)),
        }
        if cursor:
            params["cursor"] = cursor
        result = _request_requiring_membership(
            "GET",
            "conversations.replies",
            params=params,
            not_a_member_codes=_HIDDEN_FROM_NON_MEMBER_CODES,
        )
        messages = [
            {
                "ts": m.get("ts"),
                "user": m.get("user"),
                "text": m.get("text"),
            }
            for m in result.get("messages") or []
        ]
        return _success(
            messages=messages,
            has_more=bool(result.get("has_more")),
            next_cursor=(result.get("response_metadata") or {}).get("next_cursor")
            or "",
        )
    except Exception as e:
        logger.error(
            f"Error fetching Slack thread replies for {channel}:{thread_ts}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def slack_get_channel_info(channel: str) -> str:
    """
    Get metadata for a channel, private channel, or DM — including its
    topic, purpose, member count, and archived/private flags.
    """
    try:
        channel_id = _resolve_channel_id(channel)
        result = _request("GET", "conversations.info", params={"channel": channel_id})
        info = result.get("channel") or {}
        return _success(
            id=info.get("id"),
            name=info.get("name"),
            topic=(info.get("topic") or {}).get("value", ""),
            purpose=(info.get("purpose") or {}).get("value", ""),
            is_archived=info.get("is_archived", False),
            is_private=info.get("is_private", False),
            is_im=info.get("is_im", False),
            is_mpim=info.get("is_mpim", False),
            num_members=info.get("num_members"),
        )
    except Exception as e:
        logger.error(f"Error fetching Slack channel info for {channel}: {e}")
        return _error(str(e))


@mcp.tool()
def slack_list_direct_messages(limit: int = 200) -> str:
    """
    List direct-message and group-direct-message conversations the bot is a
    member of: id, and either the other user's id (1:1 DM) or the
    conversation name (group DM).
    Use the returned id with slack_get_channel_history, slack_search_messages,
    or slack_post_message to work with a specific DM.
    """
    conversations: list[dict[str, Any]] = []
    max_conversations = max(1, min(int(limit), MAX_CHANNELS))
    cursor: str | None = None
    truncated = False
    pages_scanned = 0
    try:
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "types": "im,mpim",
                "exclude_archived": "true",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                result = _request("GET", "conversations.list", params=params)
            except Exception as page_exc:
                if not pages_scanned:
                    raise
                # Mirrors slack_list_channels: a mid-pagination failure (e.g.
                # a rate limit that outlived the single retry) must not
                # discard the pages already fetched.
                logger.warning(f"Slack DM pagination stopped early: {page_exc}")
                return _success(
                    conversations=conversations, truncated=True, error=str(page_exc)
                )
            pages_scanned += 1
            raw_conversations = result.get("channels") or []
            for index, conv in enumerate(raw_conversations):
                entry: dict[str, Any] = {
                    "id": conv.get("id"),
                    "is_group_dm": bool(conv.get("is_mpim")),
                }
                if conv.get("is_mpim"):
                    entry["name"] = conv.get("name")
                else:
                    entry["user"] = conv.get("user")
                conversations.append(entry)
                if len(conversations) >= max_conversations:
                    more_in_page = index + 1 < len(raw_conversations)
                    more_pages = bool(
                        (result.get("response_metadata") or {}).get("next_cursor")
                    )
                    return _success(
                        conversations=conversations,
                        truncated=more_in_page or more_pages,
                    )
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break
        else:
            truncated = bool(cursor)
        return _success(conversations=conversations, truncated=truncated)
    except Exception as e:
        logger.error(f"Error listing Slack direct messages: {e}")
        return _error(str(e))


@mcp.tool()
def slack_search_messages(
    channel: str,
    query: str,
    limit: int = 50,
    scan_limit: int = 200,
    include_thread_replies: bool = True,
) -> str:
    """
    Search a channel, private channel, or DM's recent history (and, by
    default, its threads' replies) for messages containing `query`
    (case-insensitive substring match). If the bot isn't a member of the
    channel yet, you'll get an actionable error instead of an empty/opaque
    failure: check with the user before calling slack_join_channel to add
    the bot (only works for a public channel), or ask a member to `/invite`
    the bot for a private channel or DM.
    limit: max matching messages to return (default 50, capped at 1000).
    scan_limit: max messages to scan from the conversation's history to look
    for matches in (default 200, capped at 1000) — distinct from `limit`,
    which bounds the result count, not the scan depth. Each thread's replies
    are scanned up to 200 per thread regardless of scan_limit; a thread with
    more replies than that may under-report matches in its tail.
    This only scans the given conversation's own history, up to scan_limit
    most recent messages — it is not a Slack workspace-wide search, which
    needs a user-token search:read scope this bot-token connector does not
    request. To search multiple conversations, call this once per channel/DM
    (see slack_list_channels / slack_list_direct_messages).
    """
    needle = query.strip().lower()
    if not needle:
        return _error("query must not be empty")
    try:
        channel_id = _resolve_channel_id(channel)
        max_scan = max(1, min(int(scan_limit), 1000))
        max_matches = max(1, min(int(limit), 1000))
        matches: list[dict[str, Any]] = []
        matched_ts: set[str] = set()
        threaded_parents: list[dict[str, Any]] = []
        scanned = 0
        cursor: str | None = None
        history_pagination_error: str | None = None
        history_pagination_error_is_actionable = False

        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "channel": channel_id,
                "limit": min(200, max_scan - scanned),
            }
            if cursor:
                params["cursor"] = cursor
            try:
                result = _request_requiring_membership(
                    "GET", "conversations.history", params=params
                )
            except Exception as page_exc:
                if cursor is None:
                    raise
                # Mirrors slack_list_channels/slack_list_direct_messages: a
                # mid-pagination failure must not discard the matches
                # already found on earlier pages. Unlike those two, this
                # doesn't return immediately either — any threaded parents
                # already collected from earlier pages still get scanned
                # below instead of being silently dropped.
                logger.warning(
                    f"Slack search history pagination stopped early: {page_exc}"
                )
                history_pagination_error = str(page_exc)
                history_pagination_error_is_actionable = isinstance(
                    page_exc, _SlackNotAMemberError
                )
                break
            for m in result.get("messages") or []:
                scanned += 1
                text = str(m.get("text") or "")
                if needle in text.lower():
                    ts = m.get("ts")
                    matches.append({"ts": ts, "user": m.get("user"), "text": text})
                    if ts:
                        matched_ts.add(ts)
                if m.get("reply_count"):
                    threaded_parents.append(m)
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor or scanned >= max_scan or len(matches) >= max_matches:
                break

        threads_scanned = 0
        thread_fetch_error = history_pagination_error
        thread_fetch_error_is_actionable = history_pagination_error_is_actionable
        thread_replies_truncated = False
        if include_thread_replies:
            for parent in threaded_parents:
                if threads_scanned >= MAX_SEARCH_THREADS or len(matches) >= max_matches:
                    break
                thread_ts = parent.get("thread_ts") or parent.get("ts")
                # Counted before the request, not after a successful one: a
                # failing conversations.replies call (rate limit, permission
                # error) must still consume the budget, or a heavily
                # threaded/rate-limited channel could retry far past
                # MAX_SEARCH_THREADS attempts with no caller-side bound on
                # total wall time.
                threads_scanned += 1
                try:
                    replies_result = _request_requiring_membership(
                        "GET",
                        "conversations.replies",
                        params={"channel": channel_id, "ts": thread_ts, "limit": 200},
                        not_a_member_codes=_HIDDEN_FROM_NON_MEMBER_CODES,
                    )
                except Exception as thread_exc:
                    logger.warning(
                        f"Slack thread search failed for {thread_ts}: {thread_exc}"
                    )
                    # An attempted-but-failed thread is a genuine coverage
                    # gap distinct from threads never attempted at all
                    # (below) — it must still flag truncated even though
                    # threads_scanned already counts the attempt for budget
                    # purposes. An actionable "call slack_join_channel"
                    # message (identified structurally via isinstance, not
                    # by searching the rendered text) always wins over a
                    # later transient one (rate limit, etc.) or over the
                    # history-pagination failure above, so a membership
                    # problem is never masked by an unrelated failure
                    # elsewhere in this same call.
                    is_actionable = isinstance(thread_exc, _SlackNotAMemberError)
                    if thread_fetch_error is None or (
                        is_actionable and not thread_fetch_error_is_actionable
                    ):
                        thread_fetch_error = str(thread_exc)
                        thread_fetch_error_is_actionable = is_actionable
                    continue
                # A thread with over 200 replies is only partially scanned —
                # has_more/next_cursor are not followed, so a match in the
                # untouched tail is missed. Flag it rather than silently
                # under-reporting as "complete".
                if replies_result.get("has_more"):
                    thread_replies_truncated = True
                for reply in replies_result.get("messages") or []:
                    reply_ts = reply.get("ts")
                    if reply_ts == thread_ts or reply_ts in matched_ts:
                        # Skip the parent (already scanned above) and any
                        # reply already matched via the history scan — a
                        # thread-broadcast reply is surfaced in both.
                        continue
                    text = str(reply.get("text") or "")
                    if needle in text.lower():
                        matches.append(
                            {
                                "ts": reply_ts,
                                "user": reply.get("user"),
                                "text": text,
                                "thread_ts": thread_ts,
                            }
                        )
                        if reply_ts:
                            matched_ts.add(reply_ts)

        # Unscanned/failed/under-scanned threads only count as truncation
        # when thread search was requested — with include_thread_replies=
        # False the caller opted out, so leftover threaded parents are not
        # missing coverage.
        truncated = (
            len(matches) > max_matches
            or bool(cursor)
            or (
                include_thread_replies
                and (
                    thread_fetch_error is not None
                    or thread_replies_truncated
                    or len(threaded_parents) > threads_scanned
                )
            )
        )
        result_fields: dict[str, Any] = {
            "matches": matches[:max_matches],
            "truncated": truncated,
        }
        if thread_fetch_error is not None:
            result_fields["error"] = thread_fetch_error
        return _success(**result_fields)
    except Exception as e:
        logger.error(f"Error searching Slack messages in {channel}: {e}")
        return _error(str(e))


def _set_reaction(action: str, channel: str, timestamp: str, emoji_name: str) -> str:
    """Shared body for slack_add_reaction/slack_remove_reaction.
    action: the Slack API method suffix — "add" or "remove"."""
    name = emoji_name.strip().strip(":")
    try:
        channel_id = _resolve_channel_id(channel)
        _request_requiring_membership(
            "POST",
            f"reactions.{action}",
            json_data={"channel": channel_id, "timestamp": timestamp, "name": name},
            not_a_member_codes=_HIDDEN_FROM_NON_MEMBER_CODES,
        )
        return _success(channel=channel_id, timestamp=timestamp, emoji_name=name)
    except Exception as e:
        logger.error(
            f"Error setting Slack reaction ({action}) on {channel}:{timestamp}: {e}"
        )
        return _error(str(e))


@mcp.tool()
def slack_add_reaction(channel: str, timestamp: str, emoji_name: str) -> str:
    """
    Add an emoji reaction to a message.
    channel: the channel/DM id or name the message is in.
    timestamp: the message's "ts" (from slack_get_channel_history,
    slack_get_thread_replies, slack_search_messages, or slack_post_message).
    emoji_name: the emoji short name, with or without colons (e.g.
    "thumbsup" or ":thumbsup:").
    """
    return _set_reaction("add", channel, timestamp, emoji_name)


@mcp.tool()
def slack_remove_reaction(channel: str, timestamp: str, emoji_name: str) -> str:
    """
    Remove an emoji reaction the bot previously added from a message.
    Same arguments as slack_add_reaction.
    """
    return _set_reaction("remove", channel, timestamp, emoji_name)


@mcp.tool()
def slack_upload_file(
    channel: str,
    file_path: str,
    filename: str = "",
    title: str = "",
    initial_comment: str = "",
    thread_ts: str = "",
) -> str:
    """
    Upload a local file into a Slack channel, DM, or thread.
    file_path: path to a file already on disk (e.g. something written to the
    task workspace) — must be inside an allowed directory (automatically
    scoped to the current task workspace) so this tool cannot be used to
    exfiltrate arbitrary files from the host. Pass an absolute path — a
    relative path resolves against this process's own working directory,
    not the allowed directory, and will not find a file written to the
    task workspace.
    channel: the target channel/DM id or name.
    thread_ts: optional parent message "ts" to post the file as a thread
    reply instead of into the channel's main timeline.
    """
    try:
        local_path = _resolve_allowed_file_path(file_path)
        channel_id = _resolve_channel_id(channel)
        resolved_filename = filename.strip() or local_path.name

        # Size and upload both read from the same open handle (rather than a
        # separate local_path.stat() call before opening) so the file that
        # was allowlist-checked is the exact file read from — a fresh
        # by-path stat/open after the check would reopen a window for a
        # symlink swapped in between the two.
        with local_path.open("rb") as fh:
            file_size = os.fstat(fh.fileno()).st_size
            if file_size == 0:
                raise ValueError(f"File is empty: {file_path}")

            init_result = _request(
                "GET",
                "files.getUploadURLExternal",
                params={"filename": resolved_filename, "length": file_size},
            )
            upload_url = init_result.get("upload_url")
            file_id = init_result.get("file_id")
            if not upload_url or not file_id:
                raise ValueError("Slack did not return an upload URL")

            # Per Slack's files.completeUploadExternal migration guide, the
            # returned upload_url is a pre-authenticated endpoint that takes
            # the raw file as multipart form data under the "file" field —
            # no bearer token needed (or accepted) on this leg.
            upload_response = requests.post(
                upload_url,
                files={"file": (resolved_filename, fh)},
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
        if not upload_response.ok:
            # Checked directly rather than via raise_for_status(): its
            # HTTPError embeds the pre-authenticated upload_url in str(e),
            # which would otherwise reach the caller/LLM through
            # _error(str(e)) below — the same host-detail leak this file's
            # own PermissionError message is careful to avoid elsewhere.
            logger.warning(
                "Slack file upload leg failed with status %s for upload_url=%s",
                upload_response.status_code,
                upload_url,
            )
            raise ValueError(
                f"Slack rejected the file upload (status {upload_response.status_code})"
            )

        complete_payload: dict[str, Any] = {
            "files": [{"id": file_id, "title": title or resolved_filename}],
            "channel_id": channel_id,
        }
        if initial_comment:
            complete_payload["initial_comment"] = initial_comment
        if thread_ts:
            complete_payload["thread_ts"] = thread_ts

        complete_result = _request_requiring_membership(
            "POST", "files.completeUploadExternal", json_data=complete_payload
        )
        uploaded = (complete_result.get("files") or [{}])[0]
        return _success(file_id=uploaded.get("id", file_id), filename=resolved_filename)
    except Exception as e:
        logger.error(f"Error uploading file to Slack channel {channel}: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
