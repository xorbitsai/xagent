import json
import os
import sys
import urllib.parse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from xagent.web.tools.mcp import zoom

MEETING_WRITE_FLAG = "XAGENT_ZOOM_MEETING_WRITE_ENABLED"


class MockResponse:
    def __init__(self, json_data=None, text="", status_code=200, url=""):
        self._json_data = json_data
        self.text = text or (json.dumps(json_data) if json_data else "")
        self.status_code = status_code
        self.content = self.text.encode()
        self.url = url

    def json(self):
        # Like requests.Response.json(): without explicit json_data the body
        # text is parsed, and a non-JSON body raises ValueError.
        if self._json_data is not None:
            return self._json_data
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            # Mirror real requests behavior: str(HTTPError) embeds the URL.
            raise requests.HTTPError(
                f"{self.status_code} Client Error for url: {self.url}", response=self
            )


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("ZOOM_ACCESS_TOKEN", "access-token")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("ZOOM_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="ZOOM_ACCESS_TOKEN"):
        zoom._headers()


def test_headers_include_bearer_token():
    assert zoom._headers() == {"Authorization": "Bearer access-token"}


def test_encode_meeting_id_leaves_plain_numeric_id_alone():
    assert zoom._encode_meeting_id("123456789") == "123456789"


def test_encode_meeting_id_double_encodes_uuid_with_leading_slash():
    raw = "/ajXp112QmuoKj4854875=="
    expected = urllib.parse.quote(urllib.parse.quote(raw, safe=""), safe="")
    assert zoom._encode_meeting_id(raw) == expected


def test_encode_meeting_id_double_encodes_uuid_with_double_slash():
    raw = "abc//def"
    expected = urllib.parse.quote(urllib.parse.quote(raw, safe=""), safe="")
    assert zoom._encode_meeting_id(raw) == expected


def test_request_wraps_http_error_with_message_and_status(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=400, text='{"message": "bad id"}')),
    )

    with pytest.raises(zoom._ZoomApiError, match="bad id") as excinfo:
        zoom._request("GET", "/users/me/meetings")
    assert excinfo.value.status_code == 400


def test_request_uses_only_the_message_of_a_json_error_body(monkeypatch):
    """Zoom errors are {"code": ..., "message": ...}; only the message is
    kept, not the raw body."""
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=404,
                json_data={"code": 3001, "message": "Meeting does not exist: 1."},
            )
        ),
    )

    with pytest.raises(zoom._ZoomApiError) as excinfo:
        zoom._request("GET", "/meetings/1")

    assert str(excinfo.value).endswith(" - Meeting does not exist: 1.")
    assert "3001" not in str(excinfo.value)


def test_request_falls_back_to_raw_text_for_unstructured_error_body(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="upstream 500")),
    )

    with pytest.raises(RuntimeError, match="upstream 500"):
        zoom._request("GET", "/users/me/meetings")


def test_request_truncates_long_unstructured_error_body(monkeypatch):
    """An HTML gateway error page (or similar) landing in an unstructured
    error body must not be forwarded to the LLM/logs verbatim and
    unbounded."""
    long_body = "x" * 5000
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        zoom._request("GET", "/users/me/meetings")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_vtt_to_text_strips_scaffolding():
    vtt = (
        "WEBVTT\n"
        "\n"
        "1\n"
        "00:00:01.000 --> 00:00:04.000\n"
        "Alice: Let's start the meeting.\n"
        "\n"
        "2\n"
        "00:00:05.000 --> 00:00:09.000\n"
        "Bob: 我们先过一下上周的行动项。\n"
    )
    assert zoom._vtt_to_text(vtt) == (
        "Alice: Let's start the meeting.\nBob: 我们先过一下上周的行动项。"
    )


def test_vtt_to_text_strips_short_form_timestamp_and_its_cue_index():
    """WebVTT allows the hours group to be omitted (MM:SS.mmm); both that
    timestamp line and the cue-index line preceding it must still be
    recognized as scaffolding and dropped."""
    vtt = "WEBVTT\n\n1\n00:01.000 --> 00:04.000\nAlice: quick clip.\n"
    assert zoom._vtt_to_text(vtt) == "Alice: quick clip."


def test_vtt_to_text_keeps_spoken_digit_only_line():
    """A cue-index line is digit-only AND immediately followed by a
    timestamp line; a spoken line that happens to be all digits (a PIN, an
    order number, a year read aloud) is not, and must survive."""
    vtt = (
        "WEBVTT\n"
        "\n"
        "1\n"
        "00:00:01.000 --> 00:00:04.000\n"
        "2024\n"
        "\n"
        "2\n"
        "00:00:05.000 --> 00:00:09.000\n"
        "Thanks for confirming the year.\n"
    )
    assert zoom._vtt_to_text(vtt) == ("2024\nThanks for confirming the year.")


def test_vtt_to_text_drops_trailing_bare_digit_line():
    """A digit-only line with no following line at all (a truncated/partial
    VTT download ending mid-cue) can never be complete spoken content — it
    must still be dropped as a cue index, not kept as a stray line."""
    vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nhi\n\n2\n"
    assert zoom._vtt_to_text(vtt) == "hi"


def test_list_meetings_returns_meetings_and_page_token(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"meetings": [{"id": 1}], "next_page_token": "tok-2"}
            )
        ),
    )

    result = json.loads(zoom.zoom_list_meetings())

    assert result["status"] == "success"
    assert result["meetings"] == [{"id": 1}]
    assert result["next_page_token"] == "tok-2"


def test_list_meetings_accepts_upcoming_and_page_token(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"meetings": [], "next_page_token": ""})
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(
        zoom.zoom_list_meetings(meeting_type="upcoming", page_token="tok-2")
    )

    assert result["status"] == "success"
    params = mock_request.call_args.kwargs["params"]
    assert params["type"] == "upcoming"
    assert params["next_page_token"] == "tok-2"


def test_list_meetings_rejects_previous_meetings_as_unsupported(monkeypatch):
    """List Meetings never returns past meetings — Zoom staff have confirmed
    the endpoint only accepts scheduled/live/upcoming; there is no
    "previous_meetings" type. Reject it instead of forwarding it to Zoom."""
    mock_request = Mock()
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_list_meetings(meeting_type="previous_meetings"))

    assert result["status"] == "error"
    assert "scheduled" in result["message"]
    mock_request.assert_not_called()


def test_list_meetings_rejects_unknown_meeting_type(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_list_meetings(meeting_type="past"))

    assert result["status"] == "error"
    assert "scheduled" in result["message"]
    mock_request.assert_not_called()


def test_list_meetings_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(
            return_value=MockResponse(status_code=401, text='{"message": "bad token"}')
        ),
    )

    result = json.loads(zoom.zoom_list_meetings())

    assert result["status"] == "error"
    assert "bad token" in result["message"]


def test_get_meeting_falls_back_to_past_meetings_on_404(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(status_code=404, text='{"message": "meeting not found"}'),
            MockResponse(json_data={"id": 123, "topic": "Ended meeting"}),
        ]
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_get_meeting("123"))

    assert result["status"] == "success"
    assert result["meeting"]["topic"] == "Ended meeting"
    assert mock_request.call_count == 2
    first_call, second_call = mock_request.call_args_list
    assert first_call.kwargs["url"].endswith("/meetings/123")
    assert second_call.kwargs["url"].endswith("/past_meetings/123")


def test_get_meeting_reports_not_found_when_both_legs_404(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(status_code=404, text='{"message": "not found"}')
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_get_meeting("999"))

    assert result["status"] == "error"
    assert "Meeting 999 not found" in result["message"]
    assert "past" in result["message"]
    assert mock_request.call_count == 2


def test_get_meeting_does_not_fall_back_on_non_404_error(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(status_code=500, text='{"message": "server error"}')
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_get_meeting("123"))

    assert result["status"] == "error"
    assert "server error" in result["message"]
    assert mock_request.call_count == 1


def test_get_meeting_does_not_treat_404_mention_in_body_as_not_found(monkeypatch):
    """A 500 whose error body merely mentions "404" must not trigger the
    past-meetings fallback — 404 detection is on the status code, not the text."""
    mock_request = Mock(
        return_value=MockResponse(
            status_code=500, text='{"message": "upstream proxy saw 404"}'
        )
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_get_meeting("123"))

    assert result["status"] == "error"
    assert mock_request.call_count == 1


def test_list_recordings_returns_recording_files(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"recording_files": [{"file_type": "MP4"}]}
            )
        ),
    )

    result = json.loads(zoom.zoom_list_recordings("123"))

    assert result["status"] == "success"
    assert result["recording_files"] == [{"file_type": "MP4"}]


def test_list_recordings_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=403, text='{"message": "insufficient scope"}'
            )
        ),
    )

    result = json.loads(zoom.zoom_list_recordings("123"))

    assert result["status"] == "error"
    assert "insufficient scope" in result["message"]


def test_get_meeting_transcript_uses_dedicated_endpoint_and_cleans_vtt(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"download_url": "https://download.zoom.us/transcript.vtt"}
        )
    )
    mock_get = Mock(
        return_value=MockResponse(
            text="WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nAlice: transcript text\n"
        )
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)
    monkeypatch.setattr(zoom.requests, "get", mock_get)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "success"
    assert result["transcript"] == "Alice: transcript text"
    mock_request.assert_called_once()
    assert mock_request.call_args.kwargs["url"].endswith("/meetings/123/transcript")
    assert mock_get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer access-token"
    }


def test_get_meeting_transcript_surfaces_restriction_reason(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"download_restriction_reason": "IP_ADDRESS_RESTRICTED"}
        )
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "error"
    assert "restricted" in result["message"]
    assert "IP_ADDRESS_RESTRICTED" in result["message"]
    mock_request.assert_called_once()


def test_get_meeting_transcript_reports_not_ready_instead_of_restricted(monkeypatch):
    """NOT_READY means the transcript is still processing — a retry-later
    condition — and must not be reported as "restricted" like a genuine
    restriction (DELETED_OR_TRASHED, UNSUPPORTED)."""
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "download_restriction_reason": "NOT_READY",
                "can_download": False,
            }
        )
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "error"
    assert "processing" in result["message"].lower()
    assert "restricted" not in result["message"].lower()


def test_get_meeting_transcript_respects_can_download_false_without_reason(
    monkeypatch,
):
    """can_download is a signal, but the recordings listing is consulted
    first: only once it independently finds no transcript file either is
    can_download trusted to report "restricted" rather than a silent
    downgrade of a transcript the recordings endpoint could still serve."""
    mock_request = Mock(return_value=MockResponse(json_data={"can_download": False}))
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "error"
    assert "restricted" in result["message"].lower()
    assert mock_request.call_count == 2


def test_get_meeting_transcript_can_download_false_still_falls_back_to_recordings(
    monkeypatch,
):
    """can_download: false on the transcript-shortcut endpoint must not
    short-circuit before the recordings listing is checked: the two
    endpoints can disagree, and a transcript file the recordings endpoint
    can still serve must not be misreported as restricted."""
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"can_download": False}),
            MockResponse(
                json_data={
                    "recording_files": [
                        {
                            "file_type": "TRANSCRIPT",
                            "download_url": "https://download.zoom.us/transcript.vtt",
                        },
                    ]
                }
            ),
        ]
    )
    mock_get = Mock(
        return_value=MockResponse(
            text="WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nstill downloadable\n"
        )
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)
    monkeypatch.setattr(zoom.requests, "get", mock_get)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "success"
    assert result["transcript"] == "still downloadable"
    assert mock_request.call_count == 2


def test_get_meeting_transcript_falls_back_to_recording_files_on_404(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(status_code=404, text='{"message": "no transcript"}'),
            MockResponse(
                json_data={
                    "recording_files": [
                        {"file_type": "MP4", "download_url": "https://x/video.mp4"},
                        {
                            "file_type": "TRANSCRIPT",
                            "download_url": "https://download.zoom.us/transcript.vtt",
                        },
                    ]
                }
            ),
        ]
    )
    mock_get = Mock(
        return_value=MockResponse(
            text="WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nfallback transcript\n"
        )
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)
    monkeypatch.setattr(zoom.requests, "get", mock_get)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "success"
    assert result["transcript"] == "fallback transcript"
    assert mock_get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer access-token"
    }


def test_get_meeting_transcript_falls_back_to_recordings_on_empty_success_response(
    monkeypatch,
):
    """A 200 transcript response with neither download_url nor
    download_restriction_reason (not a 404) must still fall through to the
    recordings lookup — only the 404-triggered entry into that same
    fallback was previously covered."""
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={}),
            MockResponse(
                json_data={
                    "recording_files": [
                        {
                            "file_type": "TRANSCRIPT",
                            "download_url": "https://download.zoom.us/transcript.vtt",
                        },
                    ]
                }
            ),
        ]
    )
    mock_get = Mock(
        return_value=MockResponse(
            text="WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nfallback transcript\n"
        )
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)
    monkeypatch.setattr(zoom.requests, "get", mock_get)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "success"
    assert result["transcript"] == "fallback transcript"
    assert mock_request.call_count == 2
    first_call, second_call = mock_request.call_args_list
    assert first_call.kwargs["url"].endswith("/meetings/123/transcript")
    assert second_call.kwargs["url"].endswith("/meetings/123/recordings")


def test_get_meeting_transcript_reports_not_found_when_both_legs_404(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(status_code=404, text='{"message": "not found"}')
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_get_meeting_transcript("999"))

    assert result["status"] == "error"
    assert "Meeting 999 not found" in result["message"]
    assert mock_request.call_count == 2


def test_get_meeting_transcript_reports_error_when_no_transcript_file_exists(
    monkeypatch,
):
    mock_request = Mock(
        side_effect=[
            MockResponse(status_code=404, text='{"message": "no transcript"}'),
            MockResponse(
                json_data={
                    "recording_files": [
                        {"file_type": "MP4", "download_url": "https://x/video.mp4"}
                    ]
                }
            ),
        ]
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "error"
    assert "No transcript found" in result["message"]


def test_get_meeting_transcript_download_failure_leaks_no_url(monkeypatch):
    """A failing transcript download must not leak the download URL (which can
    carry an access token as a query param) into the tool response."""
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "download_url": "https://download.zoom.us/rec/x?access_token=SECRET"
            }
        )
    )
    mock_get = Mock(
        return_value=MockResponse(
            status_code=401,
            text="unauthorized",
            url="https://download.zoom.us/rec/x?access_token=SECRET",
        )
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)
    monkeypatch.setattr(zoom.requests, "get", mock_get)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "error"
    assert "401" in result["message"]
    assert "download.zoom.us" not in result["message"]
    assert "SECRET" not in result["message"]


def test_get_meeting_transcript_download_connection_error_leaks_no_url(monkeypatch):
    """A raw connection failure (no HTTP response at all) must not leak the
    download URL either — requests.ConnectionError embeds the full request
    URL, including any access_token query param, in str(exc)."""
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "download_url": "https://download.zoom.us/rec/x?access_token=SECRET"
            }
        )
    )
    mock_get = Mock(
        side_effect=requests.ConnectionError(
            "HTTPSConnectionPool: Failed to establish a new connection "
            "to https://download.zoom.us/rec/x?access_token=SECRET"
        )
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)
    monkeypatch.setattr(zoom.requests, "get", mock_get)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "error"
    assert "download.zoom.us" not in result["message"]
    assert "SECRET" not in result["message"]
    assert "ConnectionError" in result["message"]


def test_download_text_wraps_connection_error_without_leaking_url(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "get",
        Mock(
            side_effect=requests.Timeout(
                "Read timed out for https://download.zoom.us/t?token=SECRET"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Timeout") as excinfo:
        zoom._download_text("https://download.zoom.us/t?token=SECRET")
    assert "SECRET" not in str(excinfo.value)
    assert "download.zoom.us" not in str(excinfo.value)


def test_download_text_decodes_utf8_regardless_of_headers(monkeypatch):
    """requests falls back to ISO-8859-1 for text/* without a charset; the
    decode must be explicit UTF-8 so Chinese transcripts don't mojibake."""
    response = MockResponse()
    response.content = "会议纪要：讨论了下季度目标".encode("utf-8")
    monkeypatch.setattr(zoom.requests, "get", Mock(return_value=response))

    assert (
        zoom._download_text("https://download.zoom.us/t.vtt")
        == "会议纪要：讨论了下季度目标"
    )


def test_download_text_strips_utf8_bom(monkeypatch):
    """A BOM-prefixed VTT download must decode without leaking the BOM into
    the first line, or it fails _vtt_to_text's "WEBVTT" header check."""
    response = MockResponse()
    response.content = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nhi\n".encode(
        "utf-8-sig"
    )
    monkeypatch.setattr(zoom.requests, "get", Mock(return_value=response))

    text = zoom._download_text("https://download.zoom.us/t.vtt")

    assert text.startswith("WEBVTT")
    assert zoom._vtt_to_text(text) == "hi"


def test_download_text_rejects_non_zoom_host(monkeypatch):
    """The Zoom bearer token must never be attached to a request for a
    download_url pointing somewhere other than Zoom's own domain."""
    mock_get = Mock()
    monkeypatch.setattr(zoom.requests, "get", mock_get)

    with pytest.raises(RuntimeError, match="unexpected host"):
        zoom._download_text("https://evil.example.com/t.vtt?token=SECRET")
    mock_get.assert_not_called()


def test_download_text_allows_bare_zoom_us_host(monkeypatch):
    """The exact-match branch (host == "zoom.us", as opposed to a
    subdomain matching the ".zoom.us" suffix) must actually allow the
    request through rather than only being reachable in theory."""
    monkeypatch.setattr(
        zoom.requests, "get", Mock(return_value=MockResponse(text="WEBVTT\n"))
    )

    text = zoom._download_text("https://zoom.us/t.vtt")

    assert text == "WEBVTT\n"


def test_get_current_user_returns_profile(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"id": "u1", "email": "a@b.com"})),
    )

    result = json.loads(zoom.zoom_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["email"] == "a@b.com"


def test_get_current_user_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=401, text='{"message": "expired"}')),
    )

    result = json.loads(zoom.zoom_get_current_user())

    assert result["status"] == "error"
    assert "expired" in result["message"]


def test_get_current_user_reports_error_on_empty_payload(monkeypatch):
    """A 200/204 with no body is a data problem, not a found-empty-profile
    success — surfacing it as status=success with an empty user object would
    mislead the agent into thinking a profile was found."""
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=204)),
    )

    result = json.loads(zoom.zoom_get_current_user())

    assert result["status"] == "error"


def test_get_meeting_reports_error_on_empty_payload(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=204)),
    )

    result = json.loads(zoom.zoom_get_meeting("123"))

    assert result["status"] == "error"
    assert "no data" in result["message"].lower()


def test_zoom_app_registry_requests_past_meeting_scope():
    """zoom_get_meeting falls back to /past_meetings/{id} whenever
    /meetings/{id} 404s — the normal outcome for any already-ended meeting,
    and the primary path now that list-based past-meeting discovery has been
    removed. That fallback requires meeting:read:past_meeting; without it,
    the fallback surfaces a raw scope error instead of degrading."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    zoom_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "zoom"
    )
    assert "meeting:read:past_meeting" in zoom_app["oauth_scopes"]


_CREATED_MEETING_RESPONSE = {
    "id": 85012345678,
    "uuid": "aDYlohsHRtCd4ii1uC2+hA==",
    "host_id": "host-1",
    "host_email": "host@example.com",
    "topic": "Roadmap sync",
    "type": 2,
    "start_time": "2026-10-09T07:00:00Z",
    "duration": 30,
    "timezone": "Asia/Singapore",
    "agenda": "Q4 plan",
    "join_url": "https://us02web.zoom.us/j/85012345678?pwd=abc",
    "start_url": "https://us02web.zoom.us/s/85012345678?zak=host-secret",
    "password": "123456",
    "h323_password": "123456",
    "settings": {"waiting_room": True},
}

_NO_LINKLESS_EVENT = "Do not create a calendar event without a confirmed Zoom link"


def _create_meeting_error(**kwargs) -> str:
    """Call zoom_create_meeting expecting a failure and return its message.

    Failures are raised as ToolError so they reach the agent as MCP isError
    results rather than as a successful call carrying an error payload.
    """
    with pytest.raises(ToolError) as excinfo:
        zoom.zoom_create_meeting(**kwargs)
    return str(excinfo.value)


def test_create_meeting_posts_a_scheduled_meeting_and_returns_join_url(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data=_CREATED_MEETING_RESPONSE, status_code=201)
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(
        zoom.zoom_create_meeting(
            topic=" Roadmap sync ",
            start_time="2026-10-09T15:00:00+08:00",
            duration_minutes=30,
            timezone="Asia/Singapore",
            agenda="Q4 plan",
        )
    )

    assert result["status"] == "success"
    mock_request.assert_called_once()
    kwargs = mock_request.call_args.kwargs
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == f"{zoom.ZOOM_BASE_URL}/users/me/meetings"
    assert kwargs["headers"] == {"Authorization": "Bearer access-token"}
    assert kwargs["json"] == {
        "topic": "Roadmap sync",
        "type": 2,
        "start_time": "2026-10-09T07:00:00Z",
        "duration": 30,
        "timezone": "Asia/Singapore",
        "agenda": "Q4 plan",
    }
    assert result["meeting"] == {
        "id": 85012345678,
        "topic": "Roadmap sync",
        "start_time": "2026-10-09T07:00:00Z",
        "duration": 30,
        "timezone": "Asia/Singapore",
        "join_url": "https://us02web.zoom.us/j/85012345678?pwd=abc",
        "password": "123456",
    }
    # The host start link must never reach the model, a chat, or an invite.
    assert "host-secret" not in json.dumps(result)
    assert "meeting_link" in result["message"]


def test_create_meeting_converts_a_local_time_in_timezone_to_utc(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data=_CREATED_MEETING_RESPONSE, status_code=201)
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    zoom.zoom_create_meeting(
        topic="Roadmap sync",
        start_time="2026-10-09T15:00:00",
        duration_minutes=45,
        timezone="America/New_York",
    )

    body = mock_request.call_args.kwargs["json"]
    assert body["start_time"] == "2026-10-09T19:00:00Z"
    assert body["timezone"] == "America/New_York"
    assert body["duration"] == 45
    assert "agenda" not in body


def test_create_meeting_accepts_a_utc_time_without_timezone(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data=_CREATED_MEETING_RESPONSE, status_code=201)
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    zoom.zoom_create_meeting(
        topic="Roadmap sync", start_time="2026-10-09T07:00:00.250Z", duration_minutes=30
    )

    body = mock_request.call_args.kwargs["json"]
    assert body["start_time"] == "2026-10-09T07:00:00Z"
    assert "timezone" not in body


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"start_time": "2026-10-09T15:00:00", "duration_minutes": 30},
            "UTC offset",
        ),
        ({"start_time": "2026-10-09", "duration_minutes": 30}, "RFC3339"),
        ({"start_time": "next Tuesday at 3pm", "duration_minutes": 30}, "RFC3339"),
        ({"start_time": "2026-13-09T15:00:00Z", "duration_minutes": 30}, "month"),
        ({"start_time": "2026-10-09T07:00:00Z", "duration_minutes": 0}, "positive"),
        ({"start_time": "2026-10-09T07:00:00Z", "duration_minutes": -15}, "positive"),
        ({"start_time": "2026-10-09T07:00:00Z", "duration_minutes": True}, "positive"),
        (
            {
                "start_time": "2026-10-09T15:00:00",
                "duration_minutes": 30,
                "timezone": "Mars/Olympus_Mons",
            },
            "IANA",
        ),
        (
            {
                "topic": "   ",
                "start_time": "2026-10-09T07:00:00Z",
                "duration_minutes": 30,
            },
            "topic",
        ),
        # 02:30 does not exist in New York on the day clocks spring forward.
        (
            {
                "start_time": "2026-03-08T02:30:00",
                "duration_minutes": 30,
                "timezone": "America/New_York",
            },
            "daylight-saving",
        ),
        # A valid RFC3339 value whose UTC equivalent is past year 9999.
        (
            {"start_time": "9999-12-31T23:00:00-05:00", "duration_minutes": 30},
            "supported date range",
        ),
    ],
)
def test_create_meeting_rejects_invalid_input_without_calling_zoom(
    monkeypatch, kwargs, expected
):
    mock_request = Mock()
    monkeypatch.setattr(zoom.requests, "request", mock_request)
    call_kwargs = {"topic": "Roadmap sync", **kwargs}

    message = _create_meeting_error(**call_kwargs)

    assert expected in message
    assert "No Zoom meeting was created" in message
    assert "Correct the arguments" in message
    assert _NO_LINKLESS_EVENT in message
    mock_request.assert_not_called()


_MISSING_SCOPE_BODY = {
    "code": 4711,
    "message": "Invalid access token, does not contain "
    "scopes:[meeting:write:meeting, meeting:write:meeting:admin].",
}


@pytest.mark.parametrize("body", ["json", "text"])
@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_create_meeting_missing_scope_asks_the_user_to_reconnect(
    monkeypatch, status_code, body
):
    """A grant made before meeting:write:meeting was requested cannot create
    meetings; the model must be told to get the user to reconnect, and must
    not fall back to a calendar event without a link. Zoom's own error body
    is JSON, read through _extract_error_detail; a body that is not JSON
    falls back to the raw text and must be recognized the same way."""
    response = (
        MockResponse(status_code=status_code, json_data=_MISSING_SCOPE_BODY)
        if body == "json"
        else MockResponse(status_code=status_code, text=_MISSING_SCOPE_BODY["message"])
    )
    if body == "text":
        # Keep this case on the raw-text fallback, not the JSON branch.
        assert zoom._extract_error_detail(response) is None
    monkeypatch.setattr(zoom.requests, "request", Mock(return_value=response))

    message = _create_meeting_error(
        topic="Roadmap sync",
        start_time="2026-10-09T07:00:00Z",
        duration_minutes=30,
    )

    # A connected Zoom offers no separate reconnect action, so the message
    # spells out the steps.
    assert "disconnect Zoom and connect it again" in message
    assert "meeting:write:meeting" in message
    assert "No Zoom meeting was created" in message
    assert _NO_LINKLESS_EVENT in message


@pytest.mark.parametrize("status_code", [401, 403])
def test_create_meeting_auth_error_without_scope_is_not_a_reconnect_hint(
    monkeypatch, status_code
):
    """An expired or revoked token is not a missing scope: it gets the
    generic message with Zoom's own reason, not the reconnect steps."""
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=status_code,
                json_data={"code": 124, "message": "Access token is expired."},
            )
        ),
    )

    message = _create_meeting_error(
        topic="Roadmap sync",
        start_time="2026-10-09T07:00:00Z",
        duration_minutes=30,
    )

    assert message.startswith("Zoom could not create the meeting: ")
    assert "Access token is expired." in message
    # Only the message of Zoom's JSON error body is used, not the raw body.
    assert '"code"' not in message
    assert "No Zoom meeting was created" in message
    assert "connect it again" not in message
    assert "meeting:write:meeting" not in message
    assert _NO_LINKLESS_EVENT in message


def test_create_meeting_client_error_says_no_meeting_was_created(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=429,
                text='{"code": 429, "message": "You have reached the maximum '
                "per-day number of 'Create a meeting' API requests\"}",
            )
        ),
    )

    message = _create_meeting_error(
        topic="Roadmap sync",
        start_time="2026-10-09T07:00:00Z",
        duration_minutes=30,
    )

    assert "maximum per-day number" in message
    assert "No Zoom meeting was created" in message
    assert "connect it again" not in message
    assert _NO_LINKLESS_EVENT in message


def test_create_meeting_server_error_says_the_meeting_may_exist(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=502, text="Bad Gateway")),
    )

    message = _create_meeting_error(
        topic="Roadmap sync",
        start_time="2026-10-09T07:00:00Z",
        duration_minutes=30,
    )

    assert "may or may not have been created" in message
    assert "zoom_list_meetings" in message
    assert "No Zoom meeting was created" not in message
    assert _NO_LINKLESS_EVENT in message


@pytest.mark.parametrize(
    "exc", [requests.Timeout("read timed out"), requests.ConnectionError("reset")]
)
def test_create_meeting_network_failure_says_the_meeting_may_exist(monkeypatch, exc):
    """Zoom may already have created the meeting when the response is lost;
    retrying blindly would create a second one."""
    monkeypatch.setattr(zoom.requests, "request", Mock(side_effect=exc))

    message = _create_meeting_error(
        topic="Roadmap sync",
        start_time="2026-10-09T07:00:00Z",
        duration_minutes=30,
    )

    assert "may or may not have been created" in message
    assert "zoom_list_meetings" in message
    assert "No Zoom meeting was created" not in message
    assert _NO_LINKLESS_EVENT in message


def test_create_meeting_without_join_url_in_response_is_an_error(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"id": 1}, status_code=201)),
    )

    message = _create_meeting_error(
        topic="Roadmap sync",
        start_time="2026-10-09T07:00:00Z",
        duration_minutes=30,
    )

    assert "no join_url" in message
    assert "zoom_list_meetings" in message
    assert _NO_LINKLESS_EVENT in message


def test_create_meeting_without_access_token_says_no_meeting_was_created(
    monkeypatch,
):
    monkeypatch.delenv("ZOOM_ACCESS_TOKEN")
    mock_request = Mock()
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    message = _create_meeting_error(
        topic="Roadmap sync",
        start_time="2026-10-09T07:00:00Z",
        duration_minutes=30,
    )

    assert "No Zoom meeting was created" in message
    assert _NO_LINKLESS_EVENT in message
    mock_request.assert_not_called()


_CREATE_MEETING_ARGS = {
    "topic": "Roadmap sync",
    "start_time": "2026-10-09T07:00:00Z",
    "duration_minutes": 30,
}


def _meeting_write_server() -> FastMCP:
    """A server with the tools zoom.py registers when meeting writes are on."""
    server = FastMCP("zoom-mcp-meeting-write")
    zoom._register_meeting_write_tools(server)
    return server


async def _call_create_meeting_over_mcp(arguments):
    """Call the tool through a real MCP client session, as the agent does,
    and return the CallToolResult the client receives."""
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(
        _meeting_write_server()._mcp_server
    ) as session:
        return await session.call_tool("zoom_create_meeting", arguments)


@pytest.mark.parametrize(
    ("response", "arguments", "expected"),
    [
        (
            MockResponse(status_code=503, text="Service unavailable"),
            _CREATE_MEETING_ARGS,
            "may or may not have been created",
        ),
        (
            MockResponse(status_code=400, text='{"code": 300, "message": "Bad"}'),
            _CREATE_MEETING_ARGS,
            "No Zoom meeting was created",
        ),
        (None, {**_CREATE_MEETING_ARGS, "duration_minutes": 0}, "positive"),
    ],
)
async def test_create_meeting_failure_reaches_the_agent_as_a_failed_call(
    monkeypatch, response, arguments, expected
):
    """A failure must be an MCP isError result. As a successful call with an
    error payload, the agent would record it as completed, and the same-turn
    duplicate-write guard would answer an identical retry with "already
    succeeded" without running it."""
    from xagent.core.agent.result import tool_result_succeeded
    from xagent.core.tools.adapters.vibe.mcp_adapter import (
        _normalized_mcp_call_result,
    )

    mock_request = Mock(return_value=response)
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = await _call_create_meeting_over_mcp(arguments)

    assert result.isError is True
    assert expected in result.content[0].text
    assert _NO_LINKLESS_EVENT in result.content[0].text
    assert tool_result_succeeded(_normalized_mcp_call_result(result)) is False


async def test_created_meeting_reaches_the_agent_as_a_successful_call(monkeypatch):
    from xagent.core.agent.result import tool_result_succeeded
    from xagent.core.tools.adapters.vibe.mcp_adapter import (
        _normalized_mcp_call_result,
    )

    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data=_CREATED_MEETING_RESPONSE, status_code=201
            )
        ),
    )

    result = await _call_create_meeting_over_mcp(_CREATE_MEETING_ARGS)

    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "success"
    assert payload["meeting"]["join_url"] == _CREATED_MEETING_RESPONSE["join_url"]
    assert tool_result_succeeded(_normalized_mcp_call_result(result)) is True


def test_read_requests_do_not_send_a_json_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"meetings": [], "next_page_token": ""})
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    zoom.zoom_list_meetings()

    assert "json" not in mock_request.call_args.kwargs


def test_create_meeting_is_annotated_as_non_idempotent_write():
    """idempotentHint=False enrolls the tool in the ReAct duplicate-write
    guard, so an identical repeat after a successful create in the same turn
    returns the first meeting instead of creating a second one. Failures are
    MCP errors, so a retry after one still runs."""
    tool = _meeting_write_server()._tool_manager.get_tool("zoom_create_meeting")

    assert tool.annotations is not None
    assert tool.annotations.idempotentHint is False
    assert tool.annotations.destructiveHint is False


def test_create_meeting_description_explains_the_calendar_hand_off():
    tool = _meeting_write_server()._tool_manager.get_tool("zoom_create_meeting")

    assert "does not invite or email anyone" in tool.description
    assert "meeting_link" in tool.description
    assert "do not create a calendar event" in tool.description
    # A calendar conflict that moves the meeting must not lead to a second
    # Zoom meeting; there is no tool to delete the first one.
    assert "keep this join_url" in tool.description


def test_create_meeting_description_says_to_check_the_calendar_first():
    """The calendar's conflict check runs only after the Zoom meeting exists,
    and the meeting cannot be moved or deleted here, so the description asks
    for the window to be checked before creating it."""
    tool = _meeting_write_server()._tool_manager.get_tool("zoom_create_meeting")
    description = " ".join(tool.description.split())

    assert "check the time before calling this" in description
    assert "google_calendar_search_events" in description
    assert "attendees are still checked when the calendar event is created" in (
        description
    )


_READ_TOOLS = {
    "zoom_list_meetings",
    "zoom_get_meeting",
    "zoom_list_recordings",
    "zoom_get_meeting_transcript",
    "zoom_get_current_user",
}


@pytest.mark.parametrize("value", [None, "false", "true"])
async def test_create_meeting_is_offered_only_when_meeting_write_is_enabled(
    monkeypatch, value
):
    """Launch the Zoom MCP server the way the host does (registry launch
    config, host-built environment) and list its tools over stdio. Without
    the flag, the scope is not requested, so the tool must not be offered."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
    from xagent.web.tools.config import WebToolConfig

    if value is None:
        monkeypatch.delenv(MEETING_WRITE_FLAG, raising=False)
    else:
        monkeypatch.setenv(MEETING_WRITE_FLAG, value)
    app_info = get_builtin_public_mcp_app("zoom")
    transport = WebToolConfig(
        db=None, request=None
    )._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Zoom"),
        app_info={"launch_config": app_info["launch_config"]},
        access_token="access-token",
    )
    env = dict(transport["env"])
    # Only what the test process needs to import xagent; the flag itself
    # must come from the host-built environment above.
    for name in ("PATH", "PYTHONPATH"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    params = StdioServerParameters(
        command=sys.executable, args=transport["args"], env=env
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {tool.name for tool in (await session.list_tools()).tools}

    assert _READ_TOOLS <= tools
    assert ("zoom_create_meeting" in tools) is (value == "true")
