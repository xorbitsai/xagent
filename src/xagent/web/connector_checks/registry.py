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
    ConnectorChecks,
    Contains,
    DeclarationError,
    EachHasKeys,
    Equals,
    HasKeys,
    IsType,
    L3Envelope,
    Length,
    ReadCheck,
    StatusSuccess,
    ToolExemption,
    UnprobeableCheck,
)
from .fixtures import FixtureRef

# Why a tool of an in-repo connector has no check yet (#3565). A reason for
# work still to come names the issue of the connector's family that decides
# it; a write whose object no tool can delete is a reason of its own.
_GOOGLE_TIER_1 = "#3567"
_MICROSOFT_365_TIER_1 = "#3569"
# Tier 2 runs structural reads only: no seeded objects and no write checks.
_TIER_2 = "#3573"
WRITE_WITHOUT_CLEANUP = (
    "write that creates an object no tool of this server deletes, so a write "
    "check has nothing to clean up with"
)
# Gmail's one label tool reads and writes labels. A label write check's verify
# or cleanup guard would therefore call the tool the check creates and deletes
# with, which the read-step rule of #3653 refuses; #3567 decided not to add the
# separate label read such a check needs, so the tool stays exempt.
_GMAIL_LABEL_WITHOUT_READ = (
    "read and write in one tool: it is the only Gmail tool that reads a label "
    "and the only one that creates or deletes one, so a label write check's "
    "verify or cleanup guard would call its own create and cleanup tool, which "
    "#3653 refuses; the separate label read such a check needs is not added, "
    "as decided in #3567"
)


def _exemptions(
    *,
    issue: str,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
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
            ToolExemption(tool, WRITE_WITHOUT_CLEANUP)
            for tool in writes_without_cleanup
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
                    HasKeys((), ("worksheets", "truncated")),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_MICROSOFT_365_TIER_1,
            reads=(
                "excel_get_range",
                "excel_get_used_range",
                "excel_count_values",
                "excel_find_rows",
                "excel_list_tables",
                "excel_list_table_rows",
            ),
            writes=(
                "excel_add_worksheet",
                "excel_delete_worksheet",
                "excel_update_range",
                "excel_delete_columns",
                "excel_delete_rows",
                "excel_clear_range",
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
                    Equals(
                        ("query", "property_id"),
                        FixtureRef("google-analytics.property_id"),
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
            writes_without_cleanup=(
                "google_docs_append_text",
                "google_docs_batch_update",
                "google_docs_create_document",
                "google_docs_replace_text",
            ),
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
                    Length(("sheets",), min=1),
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
                    IsType(("slide_count",), "number"),
                    EachHasKeys(("slides",), ("slide_number", "object_id", "text")),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_GOOGLE_TIER_1,
            reads=("google_slides_get_slide_thumbnail",),
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
    ),
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
    "powerpoint": ConnectorChecks(
        "powerpoint",
        reads=(
            ReadCheck(
                "powerpoint.read_fixture",
                "powerpoint_list_slides",
                {"file_path": FixtureRef("powerpoint.file_path")},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("slides", "etag")),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_MICROSOFT_365_TIER_1,
            reads=(
                "powerpoint_get_presentation_text",
                "powerpoint_get_slide_text",
                "powerpoint_list_slide_layouts",
            ),
            writes=(
                "powerpoint_set_shape_text",
                "powerpoint_add_slide",
                "powerpoint_delete_slide",
            ),
            writes_without_cleanup=("powerpoint_create_presentation",),
        ),
    ),
    "word": ConnectorChecks(
        "word",
        reads=(
            ReadCheck(
                "word.read_fixture",
                "word_get_document_text",
                {"file_path": FixtureRef("word.file_path")},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("text", "generation")),
                ),
            ),
        ),
        exemptions=_exemptions(
            issue=_MICROSOFT_365_TIER_1,
            reads=("word_list_paragraphs",),
            writes=(
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
        "deputy",
        "employment-hero",
        "facebook",
        "fireflies",
        "freshdesk",
        "github",
        "google-ads",
        "google-maps",
        "google-search-console",
        "granola",
        "hubspot",
        "instagram",
        "intercom",
        "jira",
        "linear",
        "linkedin",
        "magento",
        "meta-ads",
        "mixpanel",
        "myob",
        "onedrive",
        "outlook",
        "planner",
        "posthog",
        "rocketlane",
        "salesforce",
        "sharepoint",
        "shopify",
        "slack",
        "stripe",
        "teams",
        "whatsapp",
        "xero",
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
