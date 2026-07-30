import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import zoom


class MockResponse:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json_data = json_data if json_data is not None else {}
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.status_code = status_code
        self.content = self.text.encode()

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


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
    once = zoom._encode_meeting_id(raw)
    import urllib.parse

    expected = urllib.parse.quote(urllib.parse.quote(raw, safe=""), safe="")
    assert once == expected


def test_encode_meeting_id_double_encodes_uuid_with_double_slash():
    raw = "abc//def"
    import urllib.parse

    expected = urllib.parse.quote(urllib.parse.quote(raw, safe=""), safe="")
    assert zoom._encode_meeting_id(raw) == expected


def test_request_wraps_http_error_with_message(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=400, text='{"message": "bad id"}')),
    )

    with pytest.raises(RuntimeError, match="bad id"):
        zoom._request("GET", "/users/me/meetings")


def test_request_falls_back_to_raw_text_for_unstructured_error_body(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="upstream 500")),
    )

    with pytest.raises(RuntimeError, match="upstream 500"):
        zoom._request("GET", "/users/me/meetings")


def test_list_meetings_returns_meetings(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"meetings": [{"id": 1}]})),
    )

    result = json.loads(zoom.zoom_list_meetings())

    assert result["status"] == "success"
    assert result["meetings"] == [{"id": 1}]


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


def test_get_meeting_does_not_fall_back_on_non_404_error(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(status_code=500, text='{"message": "server error"}')
    )
    monkeypatch.setattr(zoom.requests, "request", mock_request)

    result = json.loads(zoom.zoom_get_meeting("123"))

    assert result["status"] == "error"
    assert "server error" in result["message"]
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


def test_get_meeting_transcript_uses_dedicated_endpoint(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"download_url": "https://download.zoom.us/transcript.vtt"}
        )
    )
    mock_get = Mock(return_value=MockResponse(text="WEBVTT\n\ntranscript text"))
    monkeypatch.setattr(zoom.requests, "request", mock_request)
    monkeypatch.setattr(zoom.requests, "get", mock_get)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "success"
    assert "transcript text" in result["transcript"]
    mock_request.assert_called_once()
    assert mock_request.call_args.kwargs["url"].endswith("/meetings/123/transcript")
    assert mock_get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer access-token"
    }


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
                            "download_url": "https://x/transcript.vtt",
                        },
                    ]
                }
            ),
        ]
    )
    mock_get = Mock(return_value=MockResponse(text="WEBVTT\n\nfallback transcript"))
    monkeypatch.setattr(zoom.requests, "request", mock_request)
    monkeypatch.setattr(zoom.requests, "get", mock_get)

    result = json.loads(zoom.zoom_get_meeting_transcript("123"))

    assert result["status"] == "success"
    assert "fallback transcript" in result["transcript"]
    assert mock_get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer access-token"
    }


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


def test_get_current_user_returns_profile(monkeypatch):
    monkeypatch.setattr(
        zoom.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"id": "u1", "email": "a@b.com"})),
    )

    result = json.loads(zoom.zoom_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["email"] == "a@b.com"
