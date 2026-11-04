import json
from unittest.mock import Mock

import pytest
from googleapiclient.errors import HttpError

from xagent.web.tools.mcp import google_docs


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")


class _HttpResponse:
    def __init__(self, status: int, reason: str = "error"):
        self.status = status
        self.reason = reason


def _http_error(status: int, body: dict) -> HttpError:
    return HttpError(
        _HttpResponse(status),
        json.dumps(body).encode("utf-8"),
        uri="https://docs.googleapis.com/v1/documents/doc123?alt=json",
    )


def _not_found() -> HttpError:
    return _http_error(
        404,
        {
            "error": {
                "code": 404,
                "message": "Requested entity was not found.",
                "status": "NOT_FOUND",
            }
        },
    )


def _mock_docs_service(monkeypatch):
    service = Mock()
    get_service = Mock(return_value=service)
    monkeypatch.setattr(google_docs, "get_docs_service", get_service)
    return service, get_service


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("doc123", "doc123"),
        ("https://docs.google.com/document/d/doc123/edit", "doc123"),
        ("https://docs.google.com/document/u/1/d/doc123/edit?tab=t.0", "doc123"),
    ],
)
def test_get_document_accepts_bare_id_and_link_forms(monkeypatch, value, expected):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.return_value = {
        "documentId": expected,
        "title": "Plan",
        "body": {"content": []},
    }

    result = json.loads(google_docs.google_docs_get_document(value))

    assert result["status"] == "success"
    assert service.documents.return_value.get.call_args.kwargs == {
        "documentId": expected
    }


def test_get_document_rejects_a_title_without_calling_the_api(monkeypatch):
    _, get_service = _mock_docs_service(monkeypatch)

    result = json.loads(google_docs.google_docs_get_document("Q3 planning notes"))

    assert result["status"] == "error"
    message = result["message"]
    assert "'Q3 planning notes' is not a Google Docs link" in message
    assert "cannot search for or list documents by name" in message
    assert "If google_drive_search is available, use it to find the document's id" in (
        message
    )
    assert "otherwise, or if it finds nothing, ask the user to paste" in message
    assert "https://docs.google.com/document/d/" in message
    assert "google_docs_create_document" in message
    assert "Connecting Google Drive is not needed" in message
    get_service.assert_not_called()


def test_get_document_error_keeps_non_ascii_input_readable(monkeypatch):
    _mock_docs_service(monkeypatch)

    raw = google_docs.google_docs_get_document("季度预算 报告")

    assert "'季度预算 报告' is not a Google Docs link" in raw
    assert "\\u" not in raw


def test_get_document_rejects_a_published_link_instead_of_reading_e_as_the_id(
    monkeypatch,
):
    _, get_service = _mock_docs_service(monkeypatch)

    result = json.loads(
        google_docs.google_docs_get_document(
            "https://docs.google.com/document/d/e/2PACX-1vTabc123/pub"
        )
    )

    assert result["status"] == "error"
    message = result["message"]
    assert "is a published-to-the-web link" in message
    assert "https://docs.google.com/document/d/..." in message
    get_service.assert_not_called()


def test_get_document_names_a_link_to_another_kind_of_google_file(monkeypatch):
    _, get_service = _mock_docs_service(monkeypatch)

    result = json.loads(
        google_docs.google_docs_get_document(
            "https://docs.google.com/spreadsheets/d/abc123/edit"
        )
    )

    assert result["status"] == "error"
    message = result["message"]
    assert "is a Google Sheets link, not a Google Docs link" in message
    assert "Open it with the Google Sheets tools" in message
    assert "by name" not in message
    get_service.assert_not_called()


def test_get_document_decodes_a_percent_encoded_wrapped_link(monkeypatch):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.return_value = {
        "documentId": "doc123",
        "title": "Plan",
        "body": {"content": []},
    }

    result = json.loads(
        google_docs.google_docs_get_document(
            "https://www.google.com/url?q=https%3A%2F%2Fdocs.google.com%2F"
            "document%2Fd%2Fdoc123%2Fedit&sa=D"
        )
    )

    assert result["status"] == "success"
    assert service.documents.return_value.get.call_args.kwargs == {
        "documentId": "doc123"
    }


def test_get_document_maps_not_found_to_an_actionable_message(monkeypatch):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.side_effect = _not_found()

    result = json.loads(google_docs.google_docs_get_document("doc123"))

    assert result["status"] == "error"
    message = result["message"]
    assert message.startswith("Google Docs could not open this document")
    assert "does not exist" in message
    assert "google_docs_create_document" in message
    assert "Connecting Google Drive would not change this access." in message
    assert message.endswith(
        "Google API response: HTTP 404 Requested entity was not found."
    )


@pytest.mark.parametrize("title", ["budget", "Q3-report"])
def test_get_document_not_found_for_a_one_word_title_says_names_are_not_searched(
    monkeypatch, title
):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.side_effect = _not_found()

    result = json.loads(google_docs.google_docs_get_document(title))

    assert service.documents.return_value.get.call_args.kwargs == {"documentId": title}
    message = result["message"]
    assert message.startswith("Google Docs could not open this document")
    assert "cannot search for or list documents by name" in message
    assert "If this value is the document's name rather than its id" in message
    assert "If google_drive_search is available" in message
    assert "https://docs.google.com/document/d/..." in message


def test_get_document_maps_permission_denied_to_an_actionable_message(monkeypatch):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.side_effect = _http_error(
        403,
        {
            "error": {
                "code": 403,
                "message": "The caller does not have permission",
                "status": "PERMISSION_DENIED",
            }
        },
    )

    result = json.loads(google_docs.google_docs_get_document("doc123"))

    assert result["message"].startswith("Google Docs could not open this document")
    assert "HTTP 403 The caller does not have permission" in result["message"]


@pytest.mark.parametrize(
    "reason_body",
    [
        {"errors": [{"reason": "rateLimitExceeded"}]},
        {"details": [{"reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}]},
        {"details": [{"reason": "SERVICE_DISABLED"}]},
    ],
)
def test_get_document_keeps_raw_error_for_non_access_403(monkeypatch, reason_body):
    service, _ = _mock_docs_service(monkeypatch)
    error = _http_error(
        403, {"error": {"code": 403, "message": "Denied", **reason_body}}
    )
    service.documents.return_value.get.return_value.execute.side_effect = error

    result = json.loads(google_docs.google_docs_get_document("doc123"))

    assert result["message"] == str(error)


def test_get_document_keeps_raw_error_for_other_failures(monkeypatch):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.side_effect = RuntimeError(
        "boom"
    )

    result = json.loads(google_docs.google_docs_get_document("doc123"))

    assert result == {"status": "error", "message": "boom"}


def test_append_text_maps_not_found_to_an_actionable_message(monkeypatch):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.side_effect = _not_found()

    result = json.loads(google_docs.google_docs_append_text("doc123", "more"))

    assert result["message"].startswith("Google Docs could not open this document")
    service.documents.return_value.batchUpdate.assert_not_called()


@pytest.mark.parametrize(
    "call",
    [
        lambda: google_docs.google_docs_append_text("Weekly report", "x"),
        lambda: google_docs.google_docs_replace_text("Weekly report", "a", "b"),
        lambda: google_docs.google_docs_batch_update("Weekly report", "[]"),
    ],
)
def test_editing_tools_reject_a_title_without_calling_the_api(monkeypatch, call):
    _, get_service = _mock_docs_service(monkeypatch)

    result = json.loads(call())

    assert result["status"] == "error"
    assert "is not a Google Docs link" in result["message"]
    get_service.assert_not_called()


_VIEW_ONLY_403 = {
    "error": {
        "code": 403,
        "message": "The caller does not have permission",
        "status": "PERMISSION_DENIED",
    }
}


def _assert_refused_edit(result):
    assert result["status"] == "error"
    message = result["message"]
    assert message.startswith("Google Docs could not make this change")
    assert "only view or comment access" in message
    assert "can edit the document" in message
    assert "could not open" not in message
    assert "Connecting Google Drive would not change this access." in message
    assert message.endswith(
        "Google API response: HTTP 403 The caller does not have permission"
    )


def test_append_text_explains_a_refused_edit_on_a_readable_document(monkeypatch):
    service, _ = _mock_docs_service(monkeypatch)
    documents = service.documents.return_value
    documents.get.return_value.execute.return_value = {
        "documentId": "doc123",
        "body": {"content": [{"endIndex": 5}]},
    }
    documents.batchUpdate.return_value.execute.side_effect = _http_error(
        403, _VIEW_ONLY_403
    )

    result = json.loads(google_docs.google_docs_append_text("doc123", "more"))

    _assert_refused_edit(result)
    documents.batchUpdate.assert_called_once()


@pytest.mark.parametrize(
    "call",
    [
        lambda: google_docs.google_docs_replace_text("doc123", "a", "b"),
        lambda: google_docs.google_docs_batch_update(
            "doc123", '[{"insertText": {"location": {"index": 1}, "text": "x"}}]'
        ),
    ],
)
def test_editing_tools_explain_a_refused_edit(monkeypatch, call):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.batchUpdate.return_value.execute.side_effect = (
        _http_error(403, _VIEW_ONLY_403)
    )

    _assert_refused_edit(json.loads(call()))


@pytest.mark.parametrize(
    "call",
    [
        lambda: google_docs.google_docs_replace_text("doc123", "a", "b"),
        lambda: google_docs.google_docs_batch_update("doc123", "[]"),
    ],
)
def test_editing_tools_map_not_found_to_an_actionable_message(monkeypatch, call):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.batchUpdate.return_value.execute.side_effect = (
        _not_found()
    )

    result = json.loads(call())

    assert result["message"].startswith("Google Docs could not open this document")


async def test_get_document_description_says_drive_is_not_needed():
    tools = {tool.name: tool for tool in await google_docs.mcp.list_tools()}

    description = " ".join(tools["google_docs_get_document"].description.split())
    assert "cannot search for or list documents by name" in description
    assert "Connecting Google Drive is not needed" in description
    assert (
        "use google_drive_search to find its id if that tool is available"
        in description
    )
    assert "otherwise, or if it finds nothing, ask the user to paste the link" in (
        description
    )


# The Sheets API's Office-file 400 (its first sentence). The Docs tools
# use the same check, in case the Docs API answers a Word file
# stored in Drive the same way.
_OFFICE_FILE_400 = {
    "error": {
        "code": 400,
        "message": "This operation is not supported for this document",
        "status": "FAILED_PRECONDITION",
    }
}


# Every Docs tool that opens an existing document, with the request that gets
# the 400: an editing tool must pass editing=True to the error message too,
# not only to the id resolver.
@pytest.mark.parametrize(
    ("call", "request_of", "editing"),
    [
        pytest.param(
            lambda: google_docs.google_docs_get_document("doc123"),
            lambda documents: documents.get,
            False,
            id="get_document",
        ),
        pytest.param(
            lambda: google_docs.google_docs_append_text("doc123", "more"),
            lambda documents: documents.get,
            True,
            id="append_text",
        ),
        pytest.param(
            lambda: google_docs.google_docs_replace_text("doc123", "a", "b"),
            lambda documents: documents.batchUpdate,
            True,
            id="replace_text",
        ),
        pytest.param(
            lambda: google_docs.google_docs_batch_update(
                "doc123", '[{"insertText": {"location": {"index": 1}, "text": "x"}}]'
            ),
            lambda documents: documents.batchUpdate,
            True,
            id="batch_update",
        ),
    ],
)
def test_tools_explain_a_word_file_stored_in_drive(
    monkeypatch, call, request_of, editing
):
    service, _ = _mock_docs_service(monkeypatch)
    request = request_of(service.documents.return_value)
    request.return_value.execute.side_effect = _http_error(400, _OFFICE_FILE_400)

    result = json.loads(call())

    assert result["status"] == "error"
    assert request.call_args.kwargs["documentId"] == "doc123"
    message = result["message"]
    # Google Docs itself opens the file in Office compatibility mode; only
    # its API cannot.
    assert message.startswith(
        "The Google Docs tools cannot open this file: it is most likely a Word "
        "file (.docx) stored in Google Drive rather than a Google Docs document. "
        "Google Docs can open such a file in Office compatibility mode"
    )
    assert "File > Save as Google Docs" in message
    assert "is a Google Sheets or Google Slides file instead" in message
    # A tool that changes the file is not sent to read a downloaded copy, and
    # is told that saving as a Google Doc makes a new copy.
    assert ("read the downloaded copy with read_file" in message) is not editing
    assert ("To make this change with these tools" in message) is editing
    assert (
        "this creates a new document: the change will be made in that copy" in message
    ) is editing
    assert message.endswith(
        "Google API response: HTTP 400 This operation is not supported for this "
        "document"
    )


@pytest.mark.parametrize(
    ("call", "error_body"),
    [
        # The Docs API answers the id of an Excel file with a 400 that says
        # nothing about the file.
        pytest.param(
            lambda: google_docs.google_docs_get_document("doc123"),
            {
                "error": {
                    "code": 400,
                    "message": "Request contains an invalid argument.",
                    "status": "INVALID_ARGUMENT",
                }
            },
            id="get_document-invalid-argument",
        ),
        # A request built by the caller can fail a precondition for reasons
        # that have nothing to do with an Office file.
        pytest.param(
            lambda: google_docs.google_docs_batch_update(
                "doc123", '[{"deleteContentRange": {}}]'
            ),
            {
                "error": {
                    "code": 400,
                    "message": "Precondition check failed.",
                    "status": "FAILED_PRECONDITION",
                }
            },
            id="batch_update-failed-precondition",
        ),
    ],
)
def test_tools_keep_the_raw_error_for_another_400(monkeypatch, call, error_body):
    service, _ = _mock_docs_service(monkeypatch)
    error = _http_error(400, error_body)
    documents = service.documents.return_value
    documents.get.return_value.execute.side_effect = error
    documents.batchUpdate.return_value.execute.side_effect = error

    result = json.loads(call())

    assert result["status"] == "error"
    assert result["message"] == str(error)


@pytest.mark.parametrize(
    ("call", "editing"),
    [
        pytest.param(google_docs.google_docs_get_document, False, id="get_document"),
        pytest.param(
            lambda doc: google_docs.google_docs_append_text(doc, "more"),
            True,
            id="append_text",
        ),
        pytest.param(
            lambda doc: google_docs.google_docs_replace_text(doc, "a", "b"),
            True,
            id="replace_text",
        ),
        pytest.param(
            lambda doc: google_docs.google_docs_batch_update(doc, "[]"),
            True,
            id="batch_update",
        ),
    ],
)
def test_tools_give_their_own_next_steps_for_a_drive_file_link(
    monkeypatch, call, editing
):
    _, get_service = _mock_docs_service(monkeypatch)

    result = json.loads(call("https://drive.google.com/file/d/doc123/view"))

    assert result["status"] == "error"
    message = result["message"]
    assert "is a Google Drive file link, not a Google Docs link" in message
    assert "File > Save as Google Docs" in message
    # A tool that changes the file is not sent to read a downloaded copy, and
    # is told that saving as a Google Doc makes a new copy.
    assert ("read the downloaded copy with read_file" in message) is not editing
    assert (
        "this creates a new document: the change will be made in that copy" in message
    ) is editing
    get_service.assert_not_called()


# A 400 whose message names a cause is about the request, so it keeps the
# raw error for an id taken from a Drive open?id= or uc?id= link too.
@pytest.mark.parametrize(
    "link",
    [
        "https://drive.google.com/open?id=doc123",
        "https://drive.google.com/uc?export=download&id=doc123",
    ],
)
@pytest.mark.parametrize(
    ("call", "error_body"),
    [
        pytest.param(
            lambda doc: google_docs.google_docs_batch_update(
                doc, '[{"deleteContentRange": {}}]'
            ),
            {
                "error": {
                    "code": 400,
                    "message": "Precondition check failed.",
                    "status": "FAILED_PRECONDITION",
                }
            },
            id="batch_update-failed-precondition",
        ),
        pytest.param(
            lambda doc: google_docs.google_docs_replace_text(doc, "", "b"),
            {
                "error": {
                    "code": 400,
                    "message": (
                        "Invalid requests[0].replaceAllText: The containsText "
                        "text must not be empty."
                    ),
                    "status": "INVALID_ARGUMENT",
                }
            },
            id="replace_text-request-field",
        ),
    ],
)
def test_tools_keep_the_raw_error_for_a_400_with_a_cause_for_a_drive_open_or_uc_link(
    monkeypatch, link, call, error_body
):
    service, _ = _mock_docs_service(monkeypatch)
    error = _http_error(400, error_body)
    batch_update = service.documents.return_value.batchUpdate
    batch_update.return_value.execute.side_effect = error

    result = json.loads(call(link))

    assert result["status"] == "error"
    assert batch_update.call_args.kwargs["documentId"] == "doc123"
    assert result["message"] == str(error)


# A Drive open?id= or uc?id= link can name a native document or an uploaded
# file, so its id goes to the API. The Docs API answers the id of an
# uploaded file (an Excel file, at least) with a 400 that does not name the
# cause, so every tool must pass its own argument to the error message, which
# then adds the next steps for an uploaded file.
@pytest.mark.parametrize(
    "link",
    [
        "https://drive.google.com/open?id=doc123",
        "https://drive.google.com/uc?export=download&id=doc123",
    ],
)
@pytest.mark.parametrize(
    ("call", "request_of", "editing"),
    [
        pytest.param(
            google_docs.google_docs_get_document,
            lambda documents: documents.get,
            False,
            id="get_document",
        ),
        pytest.param(
            lambda doc: google_docs.google_docs_append_text(doc, "more"),
            lambda documents: documents.get,
            True,
            id="append_text",
        ),
        pytest.param(
            lambda doc: google_docs.google_docs_replace_text(doc, "a", "b"),
            lambda documents: documents.batchUpdate,
            True,
            id="replace_text",
        ),
        pytest.param(
            lambda doc: google_docs.google_docs_batch_update(
                doc, '[{"insertText": {"location": {"index": 1}, "text": "x"}}]'
            ),
            lambda documents: documents.batchUpdate,
            True,
            id="batch_update",
        ),
    ],
)
def test_tools_add_next_steps_to_a_400_for_a_drive_open_or_uc_link(
    monkeypatch, link, call, request_of, editing
):
    service, _ = _mock_docs_service(monkeypatch)
    request = request_of(service.documents.return_value)
    request.return_value.execute.side_effect = _http_error(
        400,
        {
            "error": {
                "code": 400,
                "message": "Request contains an invalid argument.",
                "status": "INVALID_ARGUMENT",
            }
        },
    )

    result = json.loads(call(link))

    assert result["status"] == "error"
    assert request.call_args.kwargs["documentId"] == "doc123"
    message = result["message"]
    assert message.startswith(
        "Google Docs rejected this request (Google API response: HTTP 400 "
        "Request contains an invalid argument.). The document id was taken "
        "from a Google Drive link"
    )
    assert "File > Save as Google Docs" in message
    assert ("read the downloaded copy with read_file" in message) is not editing
    assert ("To make this change with these tools" in message) is editing
    assert message.endswith("the error is about the request itself.")


# Once a tool has read the document, it is a Google Doc whatever link named
# it, so a 400 for the tool's later request keeps Google's own error, even
# one that names no cause.
@pytest.mark.parametrize(
    "link",
    [
        "https://drive.google.com/open?id=doc123",
        "https://drive.google.com/uc?export=download&id=doc123",
    ],
)
def test_append_text_keeps_the_raw_400_after_the_document_opened(monkeypatch, link):
    service, _ = _mock_docs_service(monkeypatch)
    documents = service.documents.return_value
    documents.get.return_value.execute.return_value = {
        "documentId": "doc123",
        "body": {"content": [{"endIndex": 5}]},
    }
    error = _http_error(
        400,
        {
            "error": {
                "code": 400,
                "message": "Request contains an invalid argument.",
                "status": "INVALID_ARGUMENT",
            }
        },
    )
    documents.batchUpdate.return_value.execute.side_effect = error

    result = json.loads(google_docs.google_docs_append_text(link, "more"))

    assert result["status"] == "error"
    assert documents.get.call_args.kwargs["documentId"] == "doc123"
    assert documents.batchUpdate.call_args.kwargs["documentId"] == "doc123"
    assert result["message"] == str(error)
