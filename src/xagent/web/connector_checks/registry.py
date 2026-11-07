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
    UnprobeableCheck,
)
from .fixtures import FixtureRef

# Connectors that have adopted V2, by catalog app id.
CONNECTOR_CHECKS: Mapping[str, ConnectorChecks] = {
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
    ),
}


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
