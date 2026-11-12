"""Each connector's V2 declaration: declared here, or derived from its L3 probe.

A connector that has adopted V2 has an entry in ``CONNECTOR_CHECKS``. Every
other catalog connector keeps ``web/connector_probes.py`` as its single
source: its probe becomes one read check, ``<app_id>.l3_probe``, judged by
``L3Envelope`` exactly as the probe route judges it, and an unprobeable
connector gets an ``UnprobeableCheck`` with its L3 reason. A connector that
adopts V2 and still wants its L3 call declares it again through
``l3_read_check``.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..connector_probes import get_connector_probe, get_unprobeable_reason
from .declarations import (
    Assertion,
    ConnectorChecks,
    Contains,
    CreatedRef,
    DeclarationError,
    EachHasKeys,
    Equals,
    HasKeys,
    IsType,
    ItemRef,
    L3Envelope,
    Length,
    ReadCheck,
    RunScopedName,
    StatusSuccess,
    Sweep,
    TenantPin,
    ToolExemption,
    ToolStep,
    UnprobeableCheck,
    WriteCheck,
)
from .fixtures import FixtureRef

# Why a tool of an in-repo connector has no check yet (#3565). A reason for
# work still to come names the issue of the connector's family that decides
# it; a write whose object no tool can delete is a reason of its own.
_GOOGLE_TIER_1 = "#3567"
_JIRA_TIER_1 = "#3570"
_MICROSOFT_365_TIER_1 = "#3569"
_SLACK_TIER_1 = "#3571"
# Whether Slack gets write checks at all: no tool of its server deletes a
# message or a file, a reaction is named by an existing emoji, no tool reads
# the tenant id, and the bot grant only reads.
_SLACK_WRITE_CHECKS = "#3743"
# Tier 2 runs structural reads only: no seeded objects and no write checks.
_TIER_2 = "#3573"
WRITE_WITHOUT_CLEANUP = (
    "write that creates an object no tool of this server deletes, so a write "
    "check has nothing to clean up with"
)
# Gmail's one label tool reads and writes labels. A label write check's verify
# or cleanup guard would therefore call the tool the check creates and deletes
# with, which the read-step rule of #3653 refuses; #3567 allows leaving the
# tool exempt instead of adding the separate label read such a check needs.
_GMAIL_LABEL_WITHOUT_READ = (
    "read and write in one tool: it is the only Gmail tool that reads a label "
    "and the only one that creates or deletes one, so a label write check's "
    "verify or cleanup guard would call its own create and cleanup tool, which "
    "#3653 refuses; no label-reading tool is added, which #3567 allows"
)
# The deck the seed creates holds Google's default slide, so what a thumbnail
# check lacks is a fixture for that slide's id, not a slide: the tenant
# registry records none, and the tool addresses a slide by its id.
_SLIDE_THUMBNAIL_WITHOUT_SLIDE_ID = (
    "read that takes a slide id, and the tenant registry records none: the "
    "seed deck holds Google's default slide, but no fixture names it, so "
    "whether a check is added is decided in #3567"
)
# The registry pins a Microsoft 365 connection by microsoft-365.tenant_id,
# the Entra tenant id, and no tool of the OneDrive, Excel or PowerPoint
# servers returns one, so their write checks would have no tenant pin to run
# before they write. Teams' writes are blocked by cleanup instead: its
# server has no tool that deletes a message.
_MICROSOFT_365_WRITE_WITHOUT_TENANT_PIN = (
    "write; its write check waits on a tenant pin: the registry pins "
    "microsoft-365.tenant_id, the Entra tenant id, and no tool of this server "
    "returns a tenant id, so whether it gets a write check is decided in "
    f"{_MICROSOFT_365_TIER_1}"
)


def _exemptions(
    *,
    issue: str,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    in_place_edits: tuple[str, ...] = (),
    writes_without_cleanup: tuple[str, ...] = (),
) -> tuple[ToolExemption, ...]:
    return (
        *(
            ToolExemption(
                tool, f"read; whether it gets a read check is decided in {issue}"
            )
            for tool in reads
        ),
        *(
            ToolExemption(
                tool,
                "write; whether it gets a write check under the rules of #3653 "
                f"is decided in {issue}",
            )
            for tool in writes
        ),
        *(
            ToolExemption(
                tool,
                "write that edits content in place: a write check creates, "
                "verifies and deletes its own object and has no step for an "
                f"edit, so whether a check is added is decided in {issue}",
            )
            for tool in in_place_edits
        ),
        *(
            ToolExemption(tool, WRITE_WITHOUT_CLEANUP)
            for tool in writes_without_cleanup
        ),
    )


def _microsoft_365_exemptions(
    name: str,
    *,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    in_place_edits: tuple[str, ...] = (),
    writes_without_cleanup: tuple[str, ...] = (),
) -> tuple[ToolExemption, ...]:
    """A Microsoft 365 connector's exemptions (#3569): it runs read checks
    only, which carry over 80% of ``name``'s calls."""
    possessive = f"{name}'" if name.endswith("s") else f"{name}'s"
    return (
        *(
            ToolExemption(
                tool,
                "read left unchecked: the checked tools carry over 80% of "
                f"{possessive} calls, as decided in {_MICROSOFT_365_TIER_1}",
            )
            for tool in reads
        ),
        *(
            ToolExemption(tool, _MICROSOFT_365_WRITE_WITHOUT_TENANT_PIN)
            for tool in writes
        ),
        *_exemptions(
            issue=_MICROSOFT_365_TIER_1,
            in_place_edits=in_place_edits,
            writes_without_cleanup=writes_without_cleanup,
        ),
    )


def l3_read_check(app_id: str) -> ReadCheck | UnprobeableCheck | None:
    """``app_id``'s L3 probe as a V2 read check, or None without one.

    An unprobeable reason wins over a probe, as in the probe route.
    """
    check_id = f"{app_id}.l3_probe"
    reason = get_unprobeable_reason(app_id)
    if reason is not None:
        return UnprobeableCheck(check_id, reason)
    probe = get_connector_probe(app_id)
    if probe is None:
        return None
    return ReadCheck(
        check_id,
        probe.tool,
        probe.arguments,
        assertions=(L3Envelope(probe.expect_keys),),
    )


def _declared_l3(app_id: str) -> ReadCheck:
    # A probe removed from connector_probes.py must fail at import, not put
    # None or an UnprobeableCheck into the declaration.
    check = l3_read_check(app_id)
    if not isinstance(check, ReadCheck):
        raise DeclarationError(f"{app_id} declares no L3 read check")
    return check


def _xero_list_check(list_field: str, tool: str, *, paged: bool) -> ReadCheck:
    """A structural read check on one of Xero's list tools that take no id.

    Every Xero list answers ``{status, <list_field>: [...], returned}`` through
    ``_unpaged_response`` (with ``total_count``) or ``_paged_response`` (with
    ``page``, ``page_size`` and ``has_more``). The items are ``_compact`` views
    that drop an empty field, so only the list and its paging facts are
    asserted, never an item's keys.
    """
    if paged:
        facts: tuple[Assertion, ...] = (
            HasKeys((), (list_field, "page", "page_size", "returned", "has_more")),
            IsType(("has_more",), "boolean"),
            Length((list_field,), max=1),
        )
    else:
        facts = (
            HasKeys((), (list_field, "total_count", "returned")),
            IsType(("total_count",), "number"),
        )
    return ReadCheck(
        f"xero.{tool}",
        tool,
        {"page_size": 1} if paged else {},
        assertions=(
            StatusSuccess(),
            *facts,
            IsType((list_field,), "array"),
            IsType(("returned",), "number"),
        ),
    )


def _jira_checks() -> ConnectorChecks:
    """Jira's Tier 1 checks (#3570).

    The reads cover the tools that carry most of Jira's calls, against the
    test tenant's seeded read project; the write check creates an issue in its
    write project, reads it back and deletes it. Each call that reads or
    writes issues names the test site by ``cloud_id`` instead of relying on
    the tool's single-site auto-detection; the tenant pin still refuses the
    write when the grant reaches any other site.
    """
    tenant = FixtureRef("jira.tenant_id")
    seeded_issue = FixtureRef("jira.seed.issue_key")
    write_project = FixtureRef("jira.write_project_key")
    name = RunScopedName("issue")
    created = CreatedRef(("issue", "key"))
    pin = TenantPin(
        ToolStep("jira_list_accessible_sites", assertions=(Length(("sites",), 1, 1),)),
        path=("sites", 0, "id"),
        expected=tenant,
    )
    read_back = ToolStep(
        "jira_get_issue",
        {"cloud_id": tenant, "issue_key": created},
        assertions=(StatusSuccess(), Equals(("issue", "summary"), name)),
    )
    return ConnectorChecks(
        "jira",
        reads=(
            _declared_l3("jira"),
            ReadCheck(
                "jira.search_issues",
                "jira_search_issues",
                {
                    "cloud_id": tenant,
                    "jql": FixtureRef("jira.seed.search_jql"),
                    "raw_fields": False,
                },
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("issues", "returned_count", "truncated")),
                    EachHasKeys(
                        ("issues",),
                        ("key", "summary", "status", "issue_type", "project_key"),
                    ),
                    Equals(("issues", 0, "key"), seeded_issue),
                ),
            ),
            ReadCheck(
                "jira.list_comments",
                "jira_list_comments",
                {"cloud_id": tenant, "issue_key": seeded_issue},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("comments", "total", "truncated")),
                    EachHasKeys(("comments",), ("id", "author", "body", "created")),
                    Equals(("comments", 0, "id"), FixtureRef("jira.seed.comment_id")),
                    IsType(("comments", 0, "body"), "string"),
                ),
            ),
            ReadCheck(
                "jira.get_issue",
                "jira_get_issue",
                {"cloud_id": tenant, "issue_key": seeded_issue},
                assertions=(
                    StatusSuccess(),
                    Equals(("issue", "key"), seeded_issue),
                    HasKeys(
                        ("issue",),
                        ("summary", "description", "status", "issue_type"),
                    ),
                    IsType(("issue", "summary"), "string"),
                    IsType(("issue", "description"), "string"),
                ),
            ),
            ReadCheck(
                "jira.current_user",
                "jira_get_current_user",
                assertions=(
                    StatusSuccess(),
                    HasKeys(("user",), ("account_id", "email", "name")),
                    IsType(("user", "account_id"), "string"),
                ),
            ),
        ),
        writes=(
            WriteCheck(
                "jira.issue_round_trip",
                create=ToolStep(
                    "jira_create_issue",
                    {"cloud_id": tenant, "project_key": write_project, "summary": name},
                    assertions=(StatusSuccess(), IsType(("issue", "key"), "string")),
                ),
                verify=read_back,
                cleanup=ToolStep(
                    "jira_delete_issue",
                    {"cloud_id": tenant, "issue_key": created},
                    assertions=(StatusSuccess(), Equals(("deleted",), True)),
                ),
                cleanup_guard=read_back,
                tenant_pin=pin,
                # Toby withholds both from a member who did not install it.
                expected_verdicts={
                    "jira_create_issue": "deny",
                    "jira_delete_issue": "deny",
                },
            ),
        ),
        exemptions=(
            *(
                ToolExemption(
                    tool,
                    "read left unchecked: the checked tools carry over 80% of "
                    f"Jira's calls, as decided in {_JIRA_TIER_1}",
                )
                for tool in (
                    "jira_list_projects",
                    "jira_search_users",
                    "jira_list_transitions",
                )
            ),
            *(
                ToolExemption(
                    tool,
                    "write that edits an existing issue: a write check creates, "
                    "verifies and deletes its own object and has no step for an "
                    f"edit, as decided in {_JIRA_TIER_1}",
                )
                for tool in ("jira_update_issue", "jira_transition_issue")
            ),
            *(
                ToolExemption(
                    tool,
                    "write that adds to an existing issue, which no tool of this "
                    "server removes on its own: a write check can neither create "
                    f"the issue first nor delete what it adds, as decided in "
                    f"{_JIRA_TIER_1}",
                )
                for tool in ("jira_add_comment", "jira_add_attachment")
            ),
        ),
        sweep=Sweep(
            tenant_pin=pin,
            find=ToolStep(
                "jira_search_issues",
                {
                    "cloud_id": tenant,
                    "jql": FixtureRef("jira.write_sweep_jql"),
                    "raw_fields": False,
                },
                assertions=(StatusSuccess(), IsType(("issues",), "array")),
            ),
            items=("issues",),
            name=("summary",),
            delete=ToolStep(
                "jira_delete_issue",
                {"cloud_id": tenant, "issue_key": ItemRef(("key",))},
                assertions=(StatusSuccess(),),
            ),
        ),
    )


def _slack_checks() -> ConnectorChecks:
    """Slack's Tier 1 checks (#3571).

    Reads only, against the QA workspace's seed: a public channel with a set
    topic, and one thread in it, a parent with one reply. The thread is
    posted last, after the bots joined and the topic was set, each of which
    leaves a message in the channel. The checks cover the tools that carry
    most of Slack's calls; no write tool was called in the window the tools
    were picked from.

    A new installation of an app that is not on the Slack Marketplace may call
    ``conversations.history`` and ``conversations.replies`` once a minute and
    gets at most 15 messages back, so a run calls each of them once with a
    small limit. History is read up to the reply, which is exclusive, so a
    message posted to the channel after the seed shifts nothing the check
    asserts. A later reply in the thread comes after the one the thread read
    asserts; the parent's reply count grows with it, so the history read
    asserts only that the count is there.

    A second run on the same Slack app within about a minute gets a 429 whose
    ``Retry-After`` exceeds the 30 s the tool waits, so the throttled reads
    fail as ``assertion_failed`` (``status_success``), as they do when the
    seed's channel or thread is gone. The QA workspace is on Slack's free
    plan, which hides messages older than 90 days, so the thread must be
    re-seeded before then, with its new ``ts`` values in the registry.
    """
    channel = FixtureRef("slack.seed.channel_id")
    thread = FixtureRef("slack.seed.thread_ts")
    reply = FixtureRef("slack.seed.reply_ts")
    channel_name = "xagent-smoke-seed"
    return ConnectorChecks(
        "slack",
        reads=(
            _declared_l3("slack"),
            ReadCheck(
                "slack.find_seed_channel",
                "slack_list_channels",
                {"name_contains": channel_name, "limit": 10},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("channels", "truncated")),
                    Length(("channels",), 1, 1),
                    EachHasKeys(("channels",), ("id", "name", "is_archived")),
                    Equals(("channels", 0, "id"), channel),
                    Equals(("channels", 0, "name"), channel_name),
                ),
            ),
            ReadCheck(
                "slack.read_seed_history",
                "slack_get_channel_history",
                {"channel": channel, "latest": reply, "limit": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("messages", "has_more", "next_cursor")),
                    Length(("messages",), 1, 1),
                    EachHasKeys(
                        ("messages",),
                        ("ts", "user", "text", "thread_ts", "reply_count"),
                    ),
                    Equals(("messages", 0, "ts"), thread),
                    Equals(("messages", 0, "thread_ts"), thread),
                    Equals(("messages", 0, "text"), "xagent smoke seed thread"),
                    IsType(("messages", 0, "reply_count"), "number"),
                ),
            ),
            ReadCheck(
                "slack.read_seed_thread",
                "slack_get_thread_replies",
                {"channel": channel, "thread_ts": thread, "limit": 15},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("messages", "has_more", "next_cursor")),
                    Length(("messages",), 2),
                    EachHasKeys(("messages",), ("ts", "user", "text")),
                    Equals(("messages", 0, "ts"), thread),
                    Equals(("messages", 1, "ts"), reply),
                    Equals(("messages", 1, "text"), "xagent smoke seed reply"),
                ),
            ),
            ReadCheck(
                "slack.read_seed_channel",
                "slack_get_channel_info",
                {"channel": channel},
                assertions=(
                    StatusSuccess(),
                    HasKeys(
                        (),
                        ("id", "name", "topic", "purpose", "is_archived", "is_private"),
                    ),
                    Equals(("id",), channel),
                    Equals(("name",), channel_name),
                    Equals(("topic",), "xagent smoke seed topic"),
                    Equals(("is_archived",), False),
                    Equals(("is_private",), False),
                ),
            ),
        ),
        exemptions=(
            ToolExemption(
                "slack_search_messages",
                "read left unchecked: it calls conversations.history and "
                "conversations.replies again, which a new install of an app not "
                "on the Slack Marketplace may each call once a minute, after "
                "slack.read_seed_history and slack.read_seed_thread have; the checked "
                f"tools carry over 80% of Slack's calls, as decided in {_SLACK_TIER_1}",
            ),
            ToolExemption(
                "slack_list_direct_messages",
                "read left unchecked: a Toby run's bot grant admits public channels "
                "only (xorbitsai/xagent-saas#2553), so the listing it sees is empty "
                "and holds nothing seeded; the checked tools carry over 80% of "
                f"Slack's calls, as decided in {_SLACK_TIER_1}",
            ),
            ToolExemption("slack_post_message", WRITE_WITHOUT_CLEANUP),
            ToolExemption("slack_upload_file", WRITE_WITHOUT_CLEANUP),
            ToolExemption(
                "slack_join_channel",
                "write that adds the bot to a channel, which no tool of this "
                "server leaves again, so a write check could not undo it; whether "
                f"Slack gets write checks is decided in {_SLACK_WRITE_CHECKS}",
            ),
            *(
                ToolExemption(
                    tool,
                    "write on a reaction, which is named by an existing emoji, so a "
                    "write check cannot give it the run-scoped name its create must "
                    "use; "
                    f"whether Slack gets write checks is decided in {_SLACK_WRITE_CHECKS}",
                )
                for tool in ("slack_add_reaction", "slack_remove_reaction")
            ),
        ),
    )


def _onedrive_checks() -> ConnectorChecks:
    """OneDrive's Tier 1 read checks (#3569), on the closed seed folder of
    the test account's drive and the text note in it."""
    note_id = FixtureRef("onedrive.seed_file_id")
    return ConnectorChecks(
        "onedrive",
        reads=(
            ReadCheck(
                "onedrive.list_seed_folder",
                "onedrive_list_items",
                {"folder_path": FixtureRef("onedrive.seed_folder_path")},
                assertions=(
                    StatusSuccess(),
                    # The note and the Word, Excel and PowerPoint seed files.
                    Length(("items",), 4, 4),
                ),
            ),
            ReadCheck(
                "onedrive.search_seed_file",
                "onedrive_search_files",
                # q7k2 is in the note's name and in no other seeded file's
                # name or content, so the search finds the note alone. It
                # spans the whole drive: no other object may carry q7k2.
                {"query": "q7k2"},
                assertions=(
                    StatusSuccess(),
                    Length(("items",), 1, 1),
                    Equals(("items", 0, "id"), note_id),
                ),
            ),
            ReadCheck(
                "onedrive.current_user",
                "onedrive_get_profile",
                assertions=(
                    StatusSuccess(),
                    HasKeys(("user",), ("id", "displayName", "userPrincipalName")),
                    IsType(("user", "id"), "string"),
                ),
            ),
            ReadCheck(
                "onedrive.get_seed_file",
                "onedrive_get_item",
                {"item_id": note_id},
                assertions=(
                    StatusSuccess(),
                    Equals(("item", "name"), "xagent-smoke-seed-note-q7k2.txt"),
                ),
            ),
            ReadCheck(
                "onedrive.read_seed_file",
                "onedrive_get_file_content",
                {"file_path": FixtureRef("onedrive.seed_file_path")},
                assertions=(
                    StatusSuccess(),
                    Equals(
                        ("text_content",),
                        "seeded by the xagent connector smoke check\n",
                    ),
                ),
            ),
        ),
        exemptions=_microsoft_365_exemptions(
            "OneDrive",
            reads=("onedrive_download_file",),
            writes=(
                "onedrive_upload_text_file",
                "onedrive_upload_file",
                "onedrive_create_folder",
                "onedrive_rename_item",
                "onedrive_move_item",
                "onedrive_delete_item",
            ),
        ),
    )


def _teams_checks() -> ConnectorChecks:
    """Teams' Tier 1 read checks (#3569), on the read team's one channel,
    which lists the seeded message first. The account belongs to other
    teams too, so the team list is checked for its shape only, and the seed
    cannot hold a chat, so the chat list is checked for its shape on an
    empty list too. The probe-derived L3 row, which reads get_current_user,
    is kept beside the five reads and takes no slot (#3568, #3569)."""
    team = FixtureRef("teams.seed_team_id")
    channel = FixtureRef("teams.seed_channel_id")
    return ConnectorChecks(
        "teams",
        reads=(
            _declared_l3("teams"),
            ReadCheck(
                "teams.auth_status",
                "auth_status",
                assertions=(
                    StatusSuccess(),
                    IsType(("user", "id"), "string"),
                ),
            ),
            ReadCheck(
                "teams.list_teams",
                "list_teams",
                assertions=(
                    StatusSuccess(),
                    Length(("teams",), 1),
                    EachHasKeys(("teams",), ("id", "displayName")),
                ),
            ),
            ReadCheck(
                "teams.list_chats",
                "list_chats",
                # Any chat must have an id; an empty list, the seed's, passes.
                assertions=(StatusSuccess(), EachHasKeys(("chats",), ("id",))),
            ),
            ReadCheck(
                "teams.list_seed_channels",
                "list_channels",
                {"team_id": team},
                assertions=(
                    StatusSuccess(),
                    Length(("channels",), 1, 1),
                    Equals(("channels", 0, "id"), channel),
                ),
            ),
            ReadCheck(
                "teams.read_seed_message",
                "get_channel_messages",
                {"team_id": team, "channel_id": channel},
                assertions=(
                    StatusSuccess(),
                    # Graph's message id is a string of digits, so the
                    # registry records the fixture as a string.
                    Equals(("messages", 0, "id"), FixtureRef("teams.seed_message_id")),
                    Contains(
                        ("messages", 0, "body", "content"), "xagent smoke seed message"
                    ),
                ),
            ),
        ),
        exemptions=_microsoft_365_exemptions(
            "Teams",
            reads=("get_chat_messages", "list_team_members"),
            writes_without_cleanup=("send_channel_message", "send_chat_message"),
        ),
    )


# Connectors that have adopted V2, by catalog app id. Each tool of an in-repo
# connector has a check or an exemption (#3565, tests/web/connector_checks/
# test_tool_coverage.py); a remote one declares the checks it has.
CONNECTOR_CHECKS: Mapping[str, ConnectorChecks] = {
    "atlassian": ConnectorChecks(
        "atlassian",
        reads=(
            ReadCheck(
                "atlassian.current_user",
                "atlassianUserInfo",
                payload="json_text",
                assertions=(HasKeys(("data",), ("accountId",)),),
                require_read_only_hint=True,
            ),
        ),
    ),
    "excel": ConnectorChecks(
        "excel",
        reads=(
            ReadCheck(
                "excel.read_fixture",
                "excel_list_worksheets",
                {"file_path": FixtureRef("excel.file_path")},
                assertions=(
                    StatusSuccess(),
                    Length(("worksheets",), 1, 1),
                    Equals(("worksheets", 0, "name"), "seed"),
                ),
            ),
            ReadCheck(
                "excel.read_seed_range",
                "excel_get_range",
                {
                    "file_path": FixtureRef("excel.file_path"),
                    "worksheet": "seed",
                    "address": "A1:B3",
                },
                assertions=(
                    StatusSuccess(),
                    Equals(("range", "values", 0, 0), "item"),
                    Equals(("range", "values", 0, 1), "quantity"),
                    Equals(("range", "values", 1, 0), "alpha"),
                    Equals(("range", "values", 1, 1), 1),
                    Equals(("range", "values", 2, 0), "beta"),
                    Equals(("range", "values", 2, 1), 2),
                ),
            ),
            ReadCheck(
                "excel.read_used_range",
                "excel_get_used_range",
                # Formatting outside A1:B3 must not widen the address.
                {
                    "file_path": FixtureRef("excel.file_path"),
                    "worksheet": "seed",
                    "values_only": True,
                },
                assertions=(
                    StatusSuccess(),
                    Equals(("range", "address"), "seed!A1:B3"),
                ),
            ),
        ),
        exemptions=_microsoft_365_exemptions(
            "Excel",
            reads=(
                "excel_count_values",
                "excel_find_rows",
                "excel_list_tables",
                "excel_list_table_rows",
            ),
            writes=("excel_add_worksheet", "excel_delete_worksheet"),
            in_place_edits=(
                "excel_update_range",
                "excel_clear_range",
                "excel_delete_columns",
                "excel_delete_rows",
                "excel_add_table_rows",
                "excel_delete_table_row",
            ),
            writes_without_cleanup=("excel_add_table",),
        ),
    ),
    "gmail": ConnectorChecks(
        "gmail",
        reads=(
            ReadCheck(
                "gmail.search_seed_message",
                "gmail_search_messages",
                {
                    "label_ids": [FixtureRef("gmail.seed_label_id")],
                    "max_results": 10,
                },
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("messages",)),
                    Length(("messages",), 1, 1),
                    EachHasKeys(
                        ("messages",),
                        ("id", "threadId", "snippet", "subject", "from"),
                    ),
                    Equals(("messages", 0, "subject"), "xagent smoke seed message"),
                ),
            ),
        ),
        exemptions=(
            *_exemptions(
                issue=_GOOGLE_TIER_1,
                reads=("gmail_read_threads",),
                writes_without_cleanup=("gmail_send_messages",),
            ),
            ToolExemption("gmail_manage_labels", _GMAIL_LABEL_WITHOUT_READ),
        ),
    ),
    # Tier 2 (#3573): the connector's only read that takes no object id or
    # query is the tool its L3 probe calls, so that tool keeps the L3 row and
    # carries the check.
    "google-ads": ConnectorChecks(
        "google-ads",
        reads=(
            _declared_l3("google-ads"),
            ReadCheck(
                "google-ads.list_accessible_customers",
                "google_ads_list_accessible_customers",
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("customer_ids",)),
                    IsType(("customer_ids",), "array"),
                ),
            ),
        ),
        exemptions=_exemptions(issue=_TIER_2, reads=("google_ads_search",)),
    ),
    "google-analytics": ConnectorChecks(
        "google-analytics",
        reads=(
            _declared_l3("google-analytics"),
            ReadCheck(
                "google-analytics.run_report",
                "google_analytics_run_report",
                {
                    "property_id": FixtureRef("google-analytics.property_id"),
                    "metrics": ["sessions"],
                    "date_ranges": [
                        {"start_date": "7daysAgo", "end_date": "yesterday"}
                    ],
                },
                assertions=(
                    StatusSuccess(),
                    HasKeys(
                        (),
                        (
                            "query",
                            "metric_headers",
                            "totals",
                            "rows",
                            "row_count",
                            "returned_count",
                            "has_more",
                            "truncated",
                        ),
                    ),
                    HasKeys(
                        ("query",),
                        ("property_id", "metrics", "date_ranges", "limit", "offset"),
                    ),
                    IsType(("rows",), "array"),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_GOOGLE_TIER_1,
            reads=("google_analytics_get_metadata",),
        ),
    ),
    "google-calendar": ConnectorChecks(
        "google-calendar",
        reads=(_declared_l3("google-calendar"),),
        exemptions=_exemptions(
            issue=_GOOGLE_TIER_1,
            reads=("google_calendar_get_event",),
            writes=(
                "google_calendar_create_events",
                "google_calendar_delete_events",
                "google_calendar_update_events",
                "google_calendar_respond_to_event",
            ),
        ),
    ),
    "google-docs": ConnectorChecks(
        "google-docs",
        reads=(
            ReadCheck(
                "google-docs.read_fixture",
                "google_docs_get_document",
                {"document_id": FixtureRef("google-docs.document_id")},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("document_id", "title", "content")),
                    Equals(("document_id",), FixtureRef("google-docs.document_id")),
                    Equals(("title",), "xagent-smoke-seed-doc"),
                    Contains(("content",), "xagent smoke seed document"),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_GOOGLE_TIER_1,
            in_place_edits=(
                "google_docs_append_text",
                "google_docs_batch_update",
                "google_docs_replace_text",
            ),
            writes_without_cleanup=("google_docs_create_document",),
        ),
    ),
    "google-drive": ConnectorChecks(
        "google-drive",
        reads=(
            ReadCheck(
                "google-drive.search_seed_folder",
                "google_drive_search",
                {
                    "query": (
                        "name = 'xagent-smoke-seed' and "
                        "mimeType = 'application/vnd.google-apps.folder' and "
                        "trashed = false"
                    ),
                    "max_results": 10,
                },
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("files", "truncated")),
                    Length(("files",), 1, 1),
                    EachHasKeys(("files",), ("id", "name", "mimeType", "modifiedTime")),
                    Equals(
                        ("files", 0, "id"), FixtureRef("google-drive.seed_folder_id")
                    ),
                ),
            ),
            ReadCheck(
                "google-drive.read_seed_file",
                "google_drive_get_file_content",
                {"file_id": FixtureRef("google-docs.document_id")},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("file", "content", "encoding", "truncated")),
                    HasKeys(("file",), ("id", "name", "mimeType")),
                    Equals(("file", "id"), FixtureRef("google-docs.document_id")),
                    Equals(("file", "name"), "xagent-smoke-seed-doc"),
                    Equals(("encoding",), "utf-8"),
                    Contains(("content",), "xagent smoke seed document"),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_GOOGLE_TIER_1,
            reads=("google_drive_download_file", "google_drive_list_permissions"),
            writes=(
                "google_drive_create_file",
                "google_drive_create_folder",
                "google_drive_delete_file",
                "google_drive_move_file",
                "google_drive_remove_permission",
                "google_drive_rename_file",
                "google_drive_share_file",
                "google_drive_update_file_content",
                "google_drive_update_permission",
                "google_drive_upload_file",
            ),
        ),
    ),
    "google-sheets": ConnectorChecks(
        "google-sheets",
        reads=(
            ReadCheck(
                "google-sheets.read_fixture",
                "google_sheets_get_spreadsheet",
                {"spreadsheet_id": FixtureRef("google-sheets.spreadsheet_id")},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("spreadsheet_id", "title", "url", "sheets")),
                    Equals(
                        ("spreadsheet_id",),
                        FixtureRef("google-sheets.spreadsheet_id"),
                    ),
                    Equals(("title",), "xagent-smoke-seed-sheet"),
                    Length(("sheets",), 1, 1),
                    EachHasKeys(
                        ("sheets",), ("sheet_id", "title", "row_count", "column_count")
                    ),
                ),
            ),
            ReadCheck(
                "google-sheets.read_seed_range",
                "google_sheets_read_range",
                {
                    "spreadsheet_id": FixtureRef("google-sheets.spreadsheet_id"),
                    "range_name": "A1:B3",
                },
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("range", "values", "truncated")),
                    Length(("values",), 3, 3),
                    Equals(("values", 0, 0), "item"),
                    Equals(("values", 0, 1), "quantity"),
                    Equals(("values", 1, 0), "alpha"),
                    Equals(("values", 1, 1), "1"),
                    Equals(("values", 2, 0), "beta"),
                    Equals(("values", 2, 1), "2"),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_GOOGLE_TIER_1,
            writes=(
                "google_sheets_update_range",
                "google_sheets_append_rows",
                "google_sheets_clear_range",
                "google_sheets_add_sheet",
                "google_sheets_delete_sheet",
            ),
            writes_without_cleanup=("google_sheets_create_spreadsheet",),
        ),
    ),
    "google-slides": ConnectorChecks(
        "google-slides",
        reads=(
            ReadCheck(
                "google-slides.read_fixture",
                "google_slides_get_presentation",
                {"presentation_id": FixtureRef("google-slides.presentation_id")},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("presentation_id", "title", "slide_count", "slides")),
                    Equals(
                        ("presentation_id",),
                        FixtureRef("google-slides.presentation_id"),
                    ),
                    Equals(("title",), "xagent-smoke-seed-slides"),
                    Equals(("slide_count",), 1),
                ),
            ),
        ),
        exemptions=(
            *_exemptions(
                issue=_GOOGLE_TIER_1,
                writes=(
                    "google_slides_add_slide",
                    "google_slides_update_slide",
                    "google_slides_delete_slide",
                    "google_slides_batch_update",
                ),
                writes_without_cleanup=(
                    "google_slides_create_presentation",
                    "google_slides_import_pptx",
                ),
            ),
            ToolExemption(
                "google_slides_get_slide_thumbnail",
                _SLIDE_THUMBNAIL_WITHOUT_SLIDE_ID,
            ),
        ),
    ),
    "jira": _jira_checks(),
    # Create-only (#3655): its one tool files a ticket no tool of the server
    # can delete, and its L3 slot stays unprobeable for the same reason.
    "jira-ticket": ConnectorChecks(
        "jira-ticket",
        reads=(UnprobeableCheck("jira-ticket.l3_probe", "write_only"),),
        exemptions=(ToolExemption("jira_ticket_create", WRITE_WITHOUT_CLEANUP),),
    ),
    "miro": ConnectorChecks(
        "miro",
        reads=(
            ReadCheck(
                "miro.current_user",
                "user_who_am_i",
                payload="structured_content",
                assertions=(HasKeys((), ("user_id",)),),
                require_read_only_hint=True,
            ),
        ),
    ),
    "notion": ConnectorChecks(
        "notion",
        reads=(
            ReadCheck(
                "notion.current_user",
                "notion-get-self",
                payload="json_text",
                assertions=(
                    HasKeys((), ("workspace", "user")),
                    HasKeys(("user",), ("id",)),
                ),
                require_read_only_hint=True,
            ),
        ),
    ),
    "onedrive": _onedrive_checks(),
    # Tier 2 (#3573). Which $select fields Graph returns for every message,
    # event or contact has not been checked against a live mailbox, so a list
    # item asserts its id alone.
    "outlook": ConnectorChecks(
        "outlook",
        reads=(
            _declared_l3("outlook"),
            ReadCheck(
                "outlook.get_profile",
                "outlook_get_profile",
                assertions=(
                    StatusSuccess(),
                    HasKeys(("user",), ("id", "userPrincipalName")),
                    IsType(("user", "id"), "string"),
                ),
            ),
            ReadCheck(
                "outlook.list_messages",
                "outlook_list_messages",
                {"top": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("messages", "next_link")),
                    EachHasKeys(("messages",), ("id",)),
                    Length(("messages",), max=1),
                ),
            ),
            ReadCheck(
                "outlook.list_events",
                "outlook_list_events",
                {"top": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("events", "next_link")),
                    EachHasKeys(("events",), ("id",)),
                    Length(("events",), max=1),
                ),
            ),
            ReadCheck(
                "outlook.list_contacts",
                "outlook_list_contacts",
                {"top": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("contacts", "next_link")),
                    EachHasKeys(("contacts",), ("id",)),
                    Length(("contacts",), max=1),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_TIER_2,
            reads=("outlook_get_message",),
            writes=("outlook_create_event", "outlook_delete_event"),
            in_place_edits=("outlook_update_event",),
            writes_without_cleanup=("outlook_send_message",),
        ),
    ),
    # Tier 2 (#3573). A list takes no page size, so no check bounds its
    # length, and a page too large for the output limit is projected down to
    # each item's id and label (and a plan's web_url), so an item asserts
    # only the keys that survive that projection.
    "planner": ConnectorChecks(
        "planner",
        reads=(
            _declared_l3("planner"),
            ReadCheck(
                "planner.list_my_plans",
                "planner_list_my_plans",
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("plans", "next_link", "truncated")),
                    EachHasKeys(("plans",), ("id", "web_url")),
                    IsType(("truncated",), "boolean"),
                ),
            ),
            ReadCheck(
                "planner.list_my_tasks",
                "planner_list_my_tasks",
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("tasks", "next_link", "truncated")),
                    EachHasKeys(("tasks",), ("id",)),
                    IsType(("truncated",), "boolean"),
                ),
            ),
            ReadCheck(
                "planner.list_my_groups",
                "planner_list_my_groups",
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("groups", "next_link", "truncated")),
                    EachHasKeys(("groups",), ("id",)),
                    IsType(("truncated",), "boolean"),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_TIER_2,
            reads=(
                "planner_list_plans",
                "planner_get_plan",
                "planner_list_buckets",
                "planner_list_tasks",
                "planner_get_task",
                "planner_get_task_details",
            ),
            writes=(
                "planner_create_plan",
                "planner_delete_plan",
                "planner_create_bucket",
                "planner_delete_bucket",
                "planner_create_task",
                "planner_delete_task",
                "planner_add_checklist_item",
                "planner_delete_checklist_item",
            ),
            in_place_edits=(
                "planner_update_task",
                "planner_update_task_description",
                "planner_assign_task",
                "planner_unassign_task",
                "planner_set_checklist_item_checked",
            ),
        ),
    ),
    "powerpoint": ConnectorChecks(
        "powerpoint",
        reads=(
            ReadCheck(
                "powerpoint.read_fixture",
                "powerpoint_list_slides",
                {"file_path": FixtureRef("powerpoint.file_path")},
                assertions=(StatusSuccess(), Equals(("total_count",), 1)),
            ),
            ReadCheck(
                "powerpoint.read_slide_text",
                "powerpoint_get_slide_text",
                {"file_path": FixtureRef("powerpoint.file_path"), "slide_index": 0},
                assertions=(
                    StatusSuccess(),
                    # The slide's one text box.
                    Equals(("total_count",), 1),
                    Equals(("shapes", 0, "text"), "xagent smoke seed presentation"),
                ),
            ),
            ReadCheck(
                "powerpoint.read_presentation_text",
                "powerpoint_get_presentation_text",
                {"file_path": FixtureRef("powerpoint.file_path")},
                assertions=(
                    StatusSuccess(),
                    Equals(("total_count",), 1),
                    Length(("slides", 0, "shapes"), 1, 1),
                    Equals(
                        ("slides", 0, "shapes", 0), "xagent smoke seed presentation"
                    ),
                    Equals(("slides", 0, "notes"), None),
                ),
            ),
        ),
        exemptions=_microsoft_365_exemptions(
            "PowerPoint",
            reads=("powerpoint_list_slide_layouts",),
            writes=("powerpoint_add_slide", "powerpoint_delete_slide"),
            in_place_edits=("powerpoint_set_shape_text",),
            writes_without_cleanup=("powerpoint_create_presentation",),
        ),
    ),
    # Tier 2 (#3573). get_site and every list take a site id; Graph's "root"
    # alias names the tenant's root site, so no fixture is needed. list_drives takes no
    # page size, so its list has no length bound.
    "sharepoint": ConnectorChecks(
        "sharepoint",
        reads=(
            _declared_l3("sharepoint"),
            ReadCheck(
                "sharepoint.get_root_site",
                "sharepoint_get_root_site",
                assertions=(
                    StatusSuccess(),
                    HasKeys(("site",), ("id", "webUrl")),
                    IsType(("site", "id"), "string"),
                ),
            ),
            ReadCheck(
                "sharepoint.get_site",
                "sharepoint_get_site",
                {"site_id": "root"},
                assertions=(
                    StatusSuccess(),
                    HasKeys(("site",), ("id", "webUrl")),
                    IsType(("site", "id"), "string"),
                ),
            ),
            ReadCheck(
                "sharepoint.list_drives",
                "sharepoint_list_drives",
                {"site_id": "root"},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("drives", "truncated")),
                    EachHasKeys(("drives",), ("id", "name")),
                    IsType(("truncated",), "boolean"),
                ),
            ),
            ReadCheck(
                "sharepoint.list_items",
                "sharepoint_list_items",
                {"site_id": "root", "top": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("items", "truncated")),
                    EachHasKeys(("items",), ("id", "name")),
                    IsType(("truncated",), "boolean"),
                    Length(("items",), max=1),
                ),
            ),
            ReadCheck(
                "sharepoint.list_lists",
                "sharepoint_list_lists",
                {"site_id": "root", "top": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("lists", "truncated")),
                    EachHasKeys(("lists",), ("id", "name")),
                    IsType(("truncated",), "boolean"),
                    Length(("lists",), max=1),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_TIER_2,
            reads=(
                "sharepoint_search_sites",
                "sharepoint_search_files",
                "sharepoint_get_file_content",
                "sharepoint_download_file",
                "sharepoint_list_list_items",
            ),
            writes=(
                "sharepoint_create_list_item",
                "sharepoint_delete_list_item",
                "sharepoint_move_item",
                "sharepoint_upload_text_file",
                "sharepoint_upload_file",
            ),
            in_place_edits=("sharepoint_update_list_item",),
        ),
    ),
    "slack": _slack_checks(),
    "teams": _teams_checks(),
    "word": ConnectorChecks(
        "word",
        reads=(
            ReadCheck(
                "word.read_fixture",
                "word_get_document_text",
                {"file_path": FixtureRef("word.file_path")},
                assertions=(
                    StatusSuccess(),
                    Contains(("text",), "xagent smoke seed document"),
                    Contains(("text",), "second seeded paragraph"),
                ),
            ),
            ReadCheck(
                "word.list_paragraphs",
                "word_list_paragraphs",
                {"file_path": FixtureRef("word.file_path")},
                assertions=(
                    StatusSuccess(),
                    Equals(("total_paragraphs",), 2),
                    Equals(("paragraphs", 0, "text"), "xagent smoke seed document"),
                    Equals(("paragraphs", 1, "text"), "second seeded paragraph"),
                ),
            ),
        ),
        exemptions=_microsoft_365_exemptions(
            "Word",
            in_place_edits=(
                "word_set_paragraph_text",
                "word_append_paragraph",
                "word_add_heading",
                "word_insert_page_break",
                "word_replace_text",
            ),
            writes_without_cleanup=("word_create_document",),
        ),
    ),
    "zendesk": ConnectorChecks(
        "zendesk",
        reads=(
            _declared_l3("zendesk"),
            ReadCheck(
                "zendesk.list_tickets",
                "zendesk_list_tickets",
                {"limit": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("tickets", "has_more", "after_cursor", "truncated")),
                    EachHasKeys(
                        ("tickets",),
                        (
                            "id",
                            "subject",
                            "status",
                            "priority",
                            "requester_id",
                            "assignee_id",
                            "group_id",
                            "tags",
                            "created_at",
                            "updated_at",
                        ),
                    ),
                    Length(("tickets",), max=1),
                ),
            ),
            ReadCheck(
                "zendesk.list_organizations",
                "zendesk_list_organizations",
                {"limit": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys(
                        (), ("organizations", "has_more", "after_cursor", "truncated")
                    ),
                    EachHasKeys(
                        ("organizations",),
                        ("id", "name", "domain_names", "created_at"),
                    ),
                    Length(("organizations",), max=1),
                ),
            ),
            ReadCheck(
                "zendesk.list_macros",
                "zendesk_list_macros",
                {"limit": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("macros", "has_more", "after_cursor", "truncated")),
                    EachHasKeys(
                        ("macros",), ("id", "title", "description", "updated_at")
                    ),
                    Length(("macros",), max=1),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_TIER_2,
            reads=(
                "zendesk_search",
                "zendesk_get_ticket",
                "zendesk_list_ticket_comments",
                "zendesk_get_user",
                "zendesk_search_users",
                "zendesk_get_organization",
                "zendesk_search_articles",
                "zendesk_get_article",
                "zendesk_search_macros",
                "zendesk_get_macro",
                "zendesk_list_side_conversations",
                "zendesk_get_side_conversation",
            ),
            writes=(
                "zendesk_create_ticket",
                "zendesk_update_ticket",
                "zendesk_delete_ticket",
                "zendesk_reply_to_ticket",
                "zendesk_add_internal_note",
                "zendesk_create_side_conversation",
                "zendesk_reply_to_side_conversation",
                "zendesk_update_side_conversation",
                "zendesk_apply_macro",
            ),
        ),
    ),
    "zoom": ConnectorChecks(
        "zoom",
        reads=(
            _declared_l3("zoom"),
            ReadCheck(
                "zoom.list_meetings",
                "zoom_list_meetings",
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("meetings", "next_page_token")),
                    IsType(("meetings",), "array"),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_TIER_2,
            reads=(
                "zoom_get_meeting",
                "zoom_list_recordings",
                "zoom_get_meeting_transcript",
            ),
            writes_without_cleanup=("zoom_create_meeting",),
        ),
    ),
    "linear": ConnectorChecks(
        "linear",
        reads=(
            _declared_l3("linear"),
            ReadCheck(
                "linear.list_teams",
                "linear_list_teams",
                {"limit": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("teams", "truncated")),
                    EachHasKeys(("teams",), ("id", "key", "name")),
                    Length(("teams",), max=1),
                    IsType(("truncated",), "boolean"),
                ),
            ),
            ReadCheck(
                "linear.list_projects",
                "linear_list_projects",
                {"limit": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("projects", "truncated")),
                    EachHasKeys(
                        ("projects",),
                        ("id", "name", "status", "progress", "url", "teams"),
                    ),
                    Length(("projects",), max=1),
                    IsType(("truncated",), "boolean"),
                ),
            ),
            ReadCheck(
                "linear.search_users",
                "linear_search_users",
                {"limit": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("users", "truncated")),
                    EachHasKeys(("users",), ("id", "name", "email", "active")),
                    Length(("users",), max=1),
                    IsType(("truncated",), "boolean"),
                ),
            ),
            ReadCheck(
                "linear.search_issues",
                "linear_search_issues",
                {"limit": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("issues", "truncated")),
                    EachHasKeys(
                        ("issues",),
                        (
                            "id",
                            "identifier",
                            "title",
                            "priority",
                            "url",
                            "createdAt",
                            "updatedAt",
                            "state",
                            "assignee",
                            "team",
                            "project",
                            "labels",
                        ),
                    ),
                    Length(("issues",), max=1),
                    IsType(("truncated",), "boolean"),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_TIER_2,
            reads=(
                "linear_get_issue",
                "linear_list_comments",
                "linear_list_labels",
                "linear_list_workflow_states",
            ),
            in_place_edits=("linear_update_issue",),
            writes_without_cleanup=("linear_create_issue", "linear_add_comment"),
        ),
    ),
    "intercom": ConnectorChecks(
        "intercom",
        reads=(
            _declared_l3("intercom"),
            ReadCheck(
                "intercom.list_conversations",
                "intercom_list_conversations",
                {"limit": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys(
                        (), ("conversations", "total_count", "has_more", "next_cursor")
                    ),
                    EachHasKeys(
                        ("conversations",),
                        (
                            "id",
                            "state",
                            "title",
                            "subject",
                            "preview",
                            "created_at",
                            "updated_at",
                        ),
                    ),
                    Length(("conversations",), max=1),
                    IsType(("has_more",), "boolean"),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_TIER_2,
            reads=(
                "intercom_get_contact",
                "intercom_get_conversation",
                "intercom_search_contacts",
            ),
            in_place_edits=("intercom_close_conversation",),
            writes_without_cleanup=(
                "intercom_add_internal_note",
                "intercom_reply_to_conversation",
            ),
        ),
    ),
    "deputy": ConnectorChecks(
        "deputy",
        reads=(
            _declared_l3("deputy"),
            # The resource is a Deputy object type, not an object id.
            ReadCheck(
                "deputy.list_companies",
                "deputy_list_resource",
                {"resource": "Company"},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("records", "truncated")),
                    IsType(("records",), "array"),
                    IsType(("truncated",), "boolean"),
                ),
            ),
            ReadCheck(
                "deputy.employee_info",
                "deputy_resource_info",
                {"resource": "Employee"},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("info",)),
                    IsType(("info",), "object"),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_TIER_2,
            reads=("deputy_get_resource", "deputy_query_resource"),
            in_place_edits=("deputy_update_resource",),
            writes_without_cleanup=("deputy_add_employee", "deputy_create_resource"),
        ),
    ),
    "employment-hero": ConnectorChecks(
        "employment-hero",
        reads=(
            _declared_l3("employment-hero"),
            # The list's own `data` envelope, as Employment Hero documents it:
            # {"items": [...], ...}. The server has not run against a live
            # account, so a live answer that differs fails here first.
            ReadCheck(
                "employment-hero.list_organisations",
                "employment_hero_list_organisations",
                {"item_per_page": 1},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("organisations",)),
                    IsType(("organisations",), "object"),
                    HasKeys(("organisations",), ("items",)),
                    IsType(("organisations", "items"), "array"),
                    Length(("organisations", "items"), max=1),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_TIER_2,
            reads=(
                "employment_hero_get_employee",
                "employment_hero_get_leave_balances",
                "employment_hero_list_employees",
                "employment_hero_list_leave_categories",
                "employment_hero_list_leave_requests",
                "employment_hero_list_team_employees",
                "employment_hero_list_teams",
                "employment_hero_list_timesheet_entries",
            ),
            in_place_edits=("employment_hero_approve_leave_request",),
            writes_without_cleanup=("employment_hero_create_leave_request",),
        ),
    ),
    "xero": ConnectorChecks(
        "xero",
        reads=(
            _declared_l3("xero"),
            _xero_list_check("accounts", "list_accounts", paged=False),
            _xero_list_check("contact_groups", "list_contact_groups", paged=False),
            _xero_list_check("items", "list_items", paged=False),
            _xero_list_check("tax_rates", "list_tax_rates", paged=False),
            _xero_list_check(
                "tracking_categories", "list_tracking_categories", paged=False
            ),
            _xero_list_check("contacts", "list_contacts", paged=True),
            _xero_list_check("invoices", "list_invoices", paged=True),
            _xero_list_check("credit_notes", "list_credit_notes", paged=True),
            _xero_list_check("bank_transactions", "list_bank_transactions", paged=True),
            _xero_list_check("payments", "list_payments", paged=True),
            _xero_list_check("quotes", "list_quotes", paged=True),
            _xero_list_check("manual_journals", "list_manual_journals", paged=True),
        ),
        exemptions=_exemptions(
            issue=_TIER_2,
            reads=(
                "list_aged_payables_by_contact",
                "list_aged_receivables_by_contact",
                "list_invoice_history",
                "list_overdue_invoices",
                "list_paid_invoices",
                "list_profit_and_loss",
                "list_receivables_ageing",
                "list_report_balance_sheet",
                "list_trial_balance",
            ),
            in_place_edits=(
                "update_bank_transaction",
                "update_contact",
                "update_credit_note",
                "update_invoice",
                "update_item",
                "update_manual_journal",
                "update_quote",
                "update_tracking_category",
                "update_tracking_options",
            ),
            writes_without_cleanup=(
                "create_bank_transaction",
                "create_contact",
                "create_credit_note",
                "create_invoice",
                "create_item",
                "create_manual_journal",
                "create_payment",
                "create_quote",
                "create_tracking_category",
                "create_tracking_options",
            ),
        ),
    ),
}


# Builtin connectors that have not adopted V2 yet (#3565). The coverage test
# skips them; every other builtin connector is in CONNECTOR_CHECKS, and each
# tool of an in-repo (``python -m``) one has a check or an exemption. The list
# only shrinks: a connector leaves it in the PR that adds its declaration, and
# a new connector ships with its declaration instead of joining it.
LEGACY_CONNECTORS: frozenset[str] = frozenset(
    {
        "aws",
        "chartmogul",
        "chrome-devtools",
        "facebook",
        "fireflies",
        "freshdesk",
        "github",
        "google-maps",
        "google-search-console",
        "granola",
        "hubspot",
        "instagram",
        "linkedin",
        "magento",
        "meta-ads",
        "mixpanel",
        "myob",
        "posthog",
        "rocketlane",
        "salesforce",
        "shopify",
        "stripe",
        "whatsapp",
    }
)


def connector_checks_for(app_id: str) -> ConnectorChecks | None:
    """The declaration the engine runs for ``app_id``, or None for an app
    with neither V2 checks nor an L3 declaration (a custom catalog app)."""
    declared = CONNECTOR_CHECKS.get(app_id)
    if declared is not None:
        return declared
    check = l3_read_check(app_id)
    if check is None:
        return None
    return ConnectorChecks(app_id, reads=(check,))
