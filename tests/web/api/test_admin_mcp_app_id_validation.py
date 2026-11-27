"""Tests for catalog app_id format validation (issue #3270).

A catalog app_id is printed verbatim in smoke-check output and log
lines.  An id containing a newline or other control character can
forge extra output lines and mislead log readers.  These tests verify
that the write models reject unsafe ids at validation time.
"""

import pytest
from pydantic import ValidationError

from xagent.web.api.admin_mcp import PublicMCPAppCreate, PublicMCPAppUpdate


class TestAppIdValidation:
    """Validate that app_id only accepts safe characters."""

    @pytest.mark.parametrize(
        "app_id",
        [
            "outlook",
            "google-drive",
            "google_drive",
            "slack",
            "a",
            "abc123",
            "my-app-v2",
        ],
    )
    def test_valid_app_ids_are_accepted(self, app_id):
        """Ids matching [a-z0-9_-]+ should pass validation."""
        base = {"app_id": app_id, "name": "Test", "transport": "oauth"}
        PublicMCPAppCreate(**base)
        PublicMCPAppUpdate(app_id=app_id)

    @pytest.mark.parametrize(
        "app_id",
        [
            "outlook\n",
            "fake\nPASS l2:fake",
            "app\r\nid",
            "UPPERCASE",
            "app with spaces",
            "app.id",
            "app/id",
            "app:id",
            "app;id",
            "app|id",
            "app\tid",
            "",
            "\n",
            " ",
        ],
    )
    def test_invalid_app_ids_are_rejected_on_create(self, app_id):
        """Ids outside [a-z0-9_-] should be rejected on create."""
        base = {"app_id": app_id, "name": "Test", "transport": "oauth"}
        with pytest.raises(ValidationError):
            PublicMCPAppCreate(**base)

    @pytest.mark.parametrize(
        "app_id",
        [
            "outlook\n",
            "fake\nPASS l2:fake",
            "UPPERCASE",
            "app with spaces",
            "app.id",
            "",
        ],
    )
    def test_invalid_app_ids_are_rejected_on_update(self, app_id):
        """Ids outside [a-z0-9_-] should be rejected on update."""
        with pytest.raises(ValidationError):
            PublicMCPAppUpdate(app_id=app_id)

    def test_none_app_id_is_accepted_on_update(self):
        """Update model allows None (no change to app_id)."""
        PublicMCPAppUpdate(app_id=None)
