import base64
import json
from email import policy
from email.parser import BytesParser
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import gmail

_parse_message = BytesParser(policy=policy.default).parsebytes


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)


def _mock_gmail_service(monkeypatch, users_mock):
    service = Mock()
    service.users.return_value = users_mock
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: service)
    return service


def _sent_raw_message(users_mock):
    raw = users_mock.messages.return_value.send.call_args.kwargs["body"]["raw"]
    return _parse_message(base64.urlsafe_b64decode(raw))


def _drafted_raw_message(users_mock):
    raw = users_mock.drafts.return_value.create.call_args.kwargs["body"]["message"][
        "raw"
    ]
    return _parse_message(base64.urlsafe_b64decode(raw))


def test_send_messages_without_attachments_still_works(monkeypatch):
    users = Mock()
    users.messages.return_value.send.return_value.execute.return_value = {"id": "m1"}
    _mock_gmail_service(monkeypatch, users)

    result = json.loads(
        gmail.gmail_send_messages(
            [{"to": "a@example.com", "subject": "Hi", "body": "hello"}],
            action="send",
        )
    )

    assert result["status"] == "success"
    assert result["results"] == [{"status": "sent", "id": "m1", "attachments": []}]


def test_send_messages_attaches_workspace_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    pdf_path = tmp_path / "deck.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf bytes")

    users = Mock()
    users.messages.return_value.send.return_value.execute.return_value = {"id": "m1"}
    _mock_gmail_service(monkeypatch, users)

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "hazel@example.com",
                    "subject": "Deck",
                    "body": "Please find attached.",
                    "attachments": [str(pdf_path)],
                }
            ],
            action="send",
        )
    )

    assert result["status"] == "success"
    assert result["results"][0]["attachments"] == [
        {"filename": "deck.pdf", "size": len(b"%PDF-1.4 fake pdf bytes")}
    ]

    sent = _sent_raw_message(users)
    assert sent.is_multipart()
    attachment_parts = [p for p in sent.iter_attachments()]
    assert len(attachment_parts) == 1
    assert attachment_parts[0].get_filename() == "deck.pdf"
    assert attachment_parts[0].get_content_type() == "application/pdf"
    assert attachment_parts[0].get_payload(decode=True) == b"%PDF-1.4 fake pdf bytes"


def test_draft_messages_also_supports_attachments(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    file_path = tmp_path / "notes.txt"
    file_path.write_text("some notes")

    users = Mock()
    users.drafts.return_value.create.return_value.execute.return_value = {"id": "d1"}
    _mock_gmail_service(monkeypatch, users)

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "a@example.com",
                    "subject": "S",
                    "body": "B",
                    "attachments": [str(file_path)],
                }
            ]
        )
    )

    assert result["status"] == "success"
    assert result["results"] == [
        {
            "status": "drafted",
            "id": "d1",
            "attachments": [{"filename": "notes.txt", "size": 10}],
        }
    ]
    drafted = _drafted_raw_message(users)
    assert [p.get_filename() for p in drafted.iter_attachments()] == ["notes.txt"]


def test_send_messages_rejects_attachment_outside_allowed_dirs(monkeypatch, tmp_path):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret")
    allowed_dir = tmp_path / "workspace"
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(allowed_dir))

    users = Mock()
    _mock_gmail_service(monkeypatch, users)

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "a@example.com",
                    "subject": "S",
                    "body": "B",
                    "attachments": [str(outside_file)],
                }
            ],
            action="send",
        )
    )

    assert result["status"] == "error"
    assert "outside the allowed" in result["message"]
    assert str(outside_file) not in result["message"]
    users.messages.return_value.send.assert_not_called()


def test_send_messages_rejects_missing_attachment(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    users = Mock()
    _mock_gmail_service(monkeypatch, users)

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "a@example.com",
                    "subject": "S",
                    "body": "B",
                    "attachments": [str(tmp_path / "missing.pdf")],
                }
            ],
            action="send",
        )
    )

    assert result["status"] == "error"
    assert "not found" in result["message"]
    users.messages.return_value.send.assert_not_called()


def test_send_messages_rejects_empty_attachment(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    users = Mock()
    _mock_gmail_service(monkeypatch, users)

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "a@example.com",
                    "subject": "S",
                    "body": "B",
                    "attachments": [str(empty_file)],
                }
            ],
            action="send",
        )
    )

    assert result["status"] == "error"
    assert "empty" in result["message"]
    users.messages.return_value.send.assert_not_called()


def test_send_messages_rejects_oversized_attachments(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    big_file = tmp_path / "big.bin"
    big_file.write_bytes(b"\x00")
    monkeypatch.setattr(gmail, "_MAX_ATTACHMENT_BYTES", 0)

    users = Mock()
    _mock_gmail_service(monkeypatch, users)

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "a@example.com",
                    "subject": "S",
                    "body": "B",
                    "attachments": [str(big_file)],
                }
            ],
            action="send",
        )
    )

    assert result["status"] == "error"
    assert "limit" in result["message"]
    users.messages.return_value.send.assert_not_called()


def test_send_messages_validates_all_attachments_before_sending_any(
    monkeypatch, tmp_path
):
    """Regression guard for the production incident this feature closes:
    a batch with one good message and one bad attachment must fail entirely
    up front, rather than silently sending the good message and only then
    failing on the second — which previously looked to the caller like a
    fully-succeeded batch was actually half-sent with no way to tell."""
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    good_file = tmp_path / "good.pdf"
    good_file.write_bytes(b"real content")

    users = Mock()
    users.messages.return_value.send.return_value.execute.return_value = {"id": "m1"}
    _mock_gmail_service(monkeypatch, users)

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "hazel@example.com",
                    "subject": "S1",
                    "body": "B1",
                    "attachments": [str(good_file)],
                },
                {
                    "to": "bright@example.com",
                    "subject": "S2",
                    "body": "B2",
                    "attachments": [str(tmp_path / "missing.pdf")],
                },
            ],
            action="send",
        )
    )

    assert result["status"] == "error"
    users.messages.return_value.send.assert_not_called()


def test_send_messages_returns_error_payload_on_api_failure(monkeypatch):
    users = Mock()
    users.messages.return_value.send.return_value.execute.side_effect = RuntimeError(
        "boom"
    )
    _mock_gmail_service(monkeypatch, users)

    result = json.loads(
        gmail.gmail_send_messages(
            [{"to": "a@example.com", "subject": "S", "body": "B"}], action="send"
        )
    )

    assert result["status"] == "error"
    assert "boom" in result["message"]
