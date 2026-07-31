import json
import urllib.parse
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import zoom


class MockResponse:
    def __init__(self, json_data=None, text="", status_code=200, url=""):
        self._json_data = json_data if json_data is not None else {}
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.status_code = status_code
        self.content = self.text.encode()
        self.url = url

    def json(self):
        return self._json_data

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
