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
    HasKeys,
    L3Envelope,
    ReadCheck,
    StatusSuccess,
    ToolExemption,
    UnprobeableCheck,
)
from .fixtures import FixtureRef

# Why a tool of an in-repo connector has no check yet (#3565).
_READ_PENDING = "read; no read check is declared for it yet"
_WRITE_PENDING = "write; write checks wait for the safety rails of #3564"
_WRITE_WITHOUT_CLEANUP = (
    "write that creates an object no tool of this server deletes, so a write "
    "check has nothing to clean up with"
)


def _exemptions(
    *,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    writes_without_cleanup: tuple[str, ...] = (),
) -> tuple[ToolExemption, ...]:
    return (
        *(ToolExemption(tool, _READ_PENDING) for tool in reads),
        *(ToolExemption(tool, _WRITE_PENDING) for tool in writes),
        *(
            ToolExemption(tool, _WRITE_WITHOUT_CLEANUP)
            for tool in writes_without_cleanup
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
                    HasKeys((), ("worksheets", "truncated")),
                ),
            ),
        ),
        exemptions=_exemptions(
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
    "google-docs": ConnectorChecks(
        "google-docs",
        reads=(
            ReadCheck(
                "google-docs.read_fixture",
                "google_docs_get_document",
                {"document_id": FixtureRef("google-docs.document_id")},
                assertions=(
                    StatusSuccess(),
                    HasKeys((), ("document_id", "title")),
                ),
            ),
        ),
        exemptions=_exemptions(
            reads=(),
            writes=(
                "google_docs_append_text",
                "google_docs_replace_text",
                "google_docs_batch_update",
            ),
            writes_without_cleanup=("google_docs_create_document",),
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
                    HasKeys((), ("spreadsheet_id", "title", "sheets")),
                ),
            ),
        ),
        exemptions=_exemptions(
            reads=("google_sheets_read_range",),
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
                    HasKeys((), ("presentation_id", "title", "slides")),
                ),
            ),
        ),
        exemptions=_exemptions(
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
        "gmail",
        "google-ads",
        "google-analytics",
        "google-calendar",
        "google-drive",
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
        "zendesk",
        "zoom",
    }
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
