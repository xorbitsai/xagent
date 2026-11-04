"""Google Drive Picker scope classification and app id checks."""

import logging

import pytest

from xagent.web.services import google_picker
from xagent.web.services.google_picker import (
    GOOGLE_PICKER_COMPANION_SCOPES,
    classify_google_drive_picker_scopes,
    google_picker_app_id_matches_client,
)

DRIVE = "https://www.googleapis.com/auth/drive"
DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"
DRIVE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_METADATA = "https://www.googleapis.com/auth/drive.metadata.readonly"
USERINFO_EMAIL = "https://www.googleapis.com/auth/userinfo.email"
USERINFO_PROFILE = "https://www.googleapis.com/auth/userinfo.profile"
GMAIL = "https://www.googleapis.com/auth/gmail.modify"


@pytest.mark.parametrize(
    ("scopes", "minimal", "expected"),
    [
        # Stored scope strings are whitespace separated.
        (DRIVE_FILE, False, "drive_file"),
        (f"{USERINFO_EMAIL} {DRIVE_FILE} {USERINFO_PROFILE}", True, "drive_file"),
        (f"  {DRIVE_FILE}\t{USERINFO_EMAIL}\n", True, "drive_file"),
        # Iterables, as found on Credentials.scopes.
        ([DRIVE_FILE, USERINFO_EMAIL], False, "drive_file"),
        ({DRIVE_FILE, "openid", "email", "profile"}, True, "drive_file"),
        ((DRIVE_FILE, ""), True, "drive_file"),
        # Full Drive access.
        ({DRIVE}, False, "full_drive"),
        ({DRIVE_READONLY}, False, "full_drive"),
        ({DRIVE_FILE, DRIVE_READONLY}, False, "full_drive"),
        (f"{DRIVE} {USERINFO_EMAIL}", True, "full_drive"),
        # Other Drive mixes.
        ({DRIVE_METADATA}, False, "other"),
        ({DRIVE_FILE, DRIVE_METADATA}, False, "other"),
        # No Drive scope at all.
        (None, False, "drive_missing"),
        ("", False, "drive_missing"),
        ([], True, "drive_missing"),
        ({USERINFO_EMAIL, USERINFO_PROFILE}, False, "drive_missing"),
        ({USERINFO_EMAIL, GMAIL}, True, "drive_missing"),
        # Non-identity scopes only matter for a minimal grant.
        ({DRIVE_FILE, GMAIL}, False, "drive_file"),
        ({DRIVE_FILE, GMAIL}, True, "other"),
        ({DRIVE, GMAIL}, False, "full_drive"),
        ({DRIVE, GMAIL}, True, "other"),
    ],
)
def test_classify_google_drive_picker_scopes(scopes, minimal, expected) -> None:
    assert classify_google_drive_picker_scopes(scopes, minimal=minimal) == expected


def test_classify_defaults_to_the_non_minimal_check() -> None:
    assert classify_google_drive_picker_scopes({DRIVE_FILE, GMAIL}) == "drive_file"


def test_companion_scopes_are_identity_only() -> None:
    assert GOOGLE_PICKER_COMPANION_SCOPES == {
        "openid",
        "email",
        "profile",
        USERINFO_EMAIL,
        USERINFO_PROFILE,
    }


@pytest.fixture(autouse=True)
def no_reported_app_id_mismatches(monkeypatch) -> None:
    monkeypatch.setattr(google_picker, "_REPORTED_APP_ID_MISMATCHES", set())


@pytest.mark.parametrize(
    ("app_id", "client_id"),
    [
        (None, "123456789012-abc.apps.googleusercontent.com"),
        ("", "123456789012-abc.apps.googleusercontent.com"),
        ("123456789012", "123456789012-abc.apps.googleusercontent.com"),
        (" 123456789012 ", "123456789012-abc.apps.googleusercontent.com"),
        ("123456789012", None),
        ("123456789012", ""),
        ("123456789012", "client-without-project-number"),
    ],
)
def test_picker_app_id_matches_when_it_cannot_disagree(
    app_id, client_id, caplog
) -> None:
    with caplog.at_level(logging.WARNING, logger="xagent.web.services.google_picker"):
        assert google_picker_app_id_matches_client(app_id, client_id) is True
    assert caplog.records == []


def test_picker_app_id_for_another_project_is_reported(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="xagent.web.services.google_picker"):
        matches = google_picker_app_id_matches_client(
            "555555555555", "123456789012-abc.apps.googleusercontent.com"
        )

    assert matches is False
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "555555555555" in record.getMessage()
    assert "123456789012" in record.getMessage()


def test_picker_app_id_mismatch_is_reported_once_per_pair(caplog) -> None:
    client = "123456789012-abc.apps.googleusercontent.com"
    other_client = "123456789012-other.apps.googleusercontent.com"

    with caplog.at_level(logging.WARNING, logger="xagent.web.services.google_picker"):
        results = [
            google_picker_app_id_matches_client("555555555555", client),
            google_picker_app_id_matches_client(" 555555555555 ", client),
            google_picker_app_id_matches_client("555555555555", other_client),
            google_picker_app_id_matches_client("666666666666", client),
            google_picker_app_id_matches_client("555555555555", client),
        ]

    # Every mismatch is still reported to the caller.
    assert results == [False] * 5
    assert [record.getMessage().split()[4] for record in caplog.records] == [
        "555555555555",
        "555555555555",
        "666666666666",
    ]
