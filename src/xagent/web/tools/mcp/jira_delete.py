"""Error hints for the Jira delete-issue tool.

Kept apart from jira.py so the delete-specific wording does not grow that
module. The DELETE request itself stays in jira.py's tool body: xagent-saas
classifies Jira write tools by the mutating method each tool body passes to
_request, and cannot see one issued from a helper.
"""

from ....core.utils import public_requests as requests

DELETE_MAY_HAVE_COMPLETED = (
    "the deletion may have completed -- call jira_get_issue on this issue "
    "before retrying, and tell the user the outcome is unconfirmed"
)

DELETE_HAS_SUBTASKS_HINT = (
    "Jira refused to delete the issue because it has subtasks -- ask the user "
    "whether to delete its subtasks too, and call again with "
    "delete_subtasks=true only if they agree"
)

DELETE_STATUS_HINTS = {
    403: (
        "Jira denied the deletion -- the connected Jira account lacks the "
        '"Delete issues" permission in this issue\'s project, which many '
        "projects grant only to admins. This is a Jira permission, not a broken "
        "connection: ask a Jira admin to grant it, or offer to transition the "
        'issue (e.g. to "Won\'t Do" or "Closed") instead'
    ),
    404: (
        "Jira could not find the issue or the site, or the connected account "
        "cannot view the issue -- confirm the exact issue key with the user; do "
        "not retry with a guessed or corrected key, which could delete a "
        "different issue"
    ),
    # A gateway answers after the request was sent, so Jira may have acted.
    502: DELETE_MAY_HAVE_COMPLETED,
    503: DELETE_MAY_HAVE_COMPLETED,
    504: DELETE_MAY_HAVE_COMPLETED,
}

_OUTCOME_UNKNOWN_ERRORS = (
    requests.Timeout,
    requests.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.JSONDecodeError,
)


def delete_error_hint(exc: Exception, *, delete_subtasks: bool) -> str | None:
    """An actionable hint for a failed DELETE request, or None.

    Only call this for an exception raised by the DELETE itself, not by the
    site lookup before it. A ConnectTimeout is the one transport failure
    known to have sent nothing, so it gets no hint. _request_absolute raises
    RuntimeError(...) from the HTTPError for an HTTP error status, which is
    where the failing response lives.
    """
    if isinstance(exc, _OUTCOME_UNKNOWN_ERRORS) and not isinstance(
        exc, requests.ConnectTimeout
    ):
        return DELETE_MAY_HAVE_COMPLETED
    response = getattr(exc.__cause__, "response", None)
    status = getattr(response, "status_code", None)
    if not isinstance(status, int):
        return None
    # Jira documents 400 for this endpoint only for an issue with subtasks
    # deleted without deleteSubtasks; with the flag already sent, the hint
    # would just repeat the call.
    if status == 400:
        return None if delete_subtasks else DELETE_HAS_SUBTASKS_HINT
    return DELETE_STATUS_HINTS.get(status)
