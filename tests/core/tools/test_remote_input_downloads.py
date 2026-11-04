"""Real download and caller boundaries, without contacting external hosts."""

import asyncio
import base64
import gzip
import json
import logging
import socket
from unittest.mock import AsyncMock

import httpx
import pytest

from xagent.core.model.image.gemini import GeminiImageModel
from xagent.core.model.image.openai import OpenAIImageModel
from xagent.core.model.tts.base import BaseTTS, TTSResult
from xagent.core.tools.core.audio_tool import AudioToolCore
from xagent.core.utils import security
from xagent.core.utils.public_download import PublicDownloadError, download_public_input
from xagent.core.workspace import TaskWorkspace

PUBLIC_IP = "93.184.216.34"


@pytest.mark.asyncio
async def test_openai_reference_download_uses_bounded_public_input_without_api_key(
    network,
):
    _, seen, replies, _ = network
    replies.append(httpx.Response(200, content=b"image-bytes"))
    model = OpenAIImageModel(model_name="gpt-image-1", api_key="provider-secret")
    path = await model._download_url("https://example.com/reference.png")
    from pathlib import Path

    try:
        assert Path(path).read_bytes() == b"image-bytes"
        assert seen[0].url.host == PUBLIC_IP
        assert "authorization" not in seen[0].headers
        assert "provider-secret" not in str(seen[0].url)
    finally:
        Path(path).unlink()


@pytest.mark.asyncio
async def test_openai_private_reference_never_reaches_socket(network):
    _, seen, _, _ = network
    model = OpenAIImageModel(model_name="gpt-image-1", api_key="provider-secret")
    with pytest.raises(PublicDownloadError):
        await model._download_url("http://192.168.1.1/reference.png")
    assert not seen


class FakeTTS(BaseTTS):
    def __init__(self):
        self.calls = []

    @property
    def abilities(self):
        return ["tts", "voice_cloning"]

    async def synthesize(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return TTSResult(audio=b"test-audio", format="mp3")


class ObservedStream(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0):
        self.chunks = chunks
        self.delay = delay
        self.read_count = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            self.read_count += 1
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.fixture
def network(monkeypatch):
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "XAGENT_TRUSTED_EGRESS_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
    dns = AsyncMock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443)),
        ]
    )
    # Model policies resolve synchronously inside a worker; input download
    # policies resolve through to_thread directly. Keep both on the same fake
    # DNS snapshot without bypassing either real destination policy.
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: dns.return_value)
    real_to_thread = asyncio.to_thread

    async def to_thread(function, *args, **kwargs):
        if function is socket.getaddrinfo:
            return await dns(*args, **kwargs)
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(security.asyncio, "to_thread", to_thread)
    seen, replies, options = [], [], []

    async def handle(request):
        seen.append(request)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def transport(**kwargs):
        options.append(kwargs)
        return httpx.MockTransport(handle)

    # Keep the real client, DNS policy and IP rewrite. Only sockets are fake.
    monkeypatch.setattr("httpx._client.AsyncHTTPTransport", transport)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    return dns, seen, replies, options


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.1.2.3/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://localhost/",
        "https://user:private-secret@example.com/",
        "file:///etc/passwd",
        "ftp://example.com/",
        "https://example.com:0/",
        "https://example.com/bad\x00path?signature=private-signature",
    ],
)
async def test_untrusted_input_targets_never_reach_a_socket(url, network):
    _, seen, _, _ = network
    with pytest.raises(PublicDownloadError, match="policy"):
        await download_public_input(url, timeout=1, max_bytes=100)
    assert seen == []


@pytest.mark.asyncio
async def test_mixed_dns_answers_fail_closed(network):
    dns, seen, _, _ = network
    dns.return_value.append(
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))
    )
    with pytest.raises(PublicDownloadError):
        await download_public_input("https://example.com/", timeout=1, max_bytes=100)
    assert seen == []


@pytest.mark.asyncio
async def test_real_download_pins_ip_host_and_sni_without_ambient_proxy(network):
    dns, seen, replies, options = network
    replies.append(httpx.Response(200, json={"segments": []}))
    result = await download_public_input(
        "https://example.com:8443/input", timeout=1, max_bytes=100
    )
    assert result.url == "https://example.com:8443/input"
    assert json.loads(result.content) == {"segments": []}
    assert dns.await_count == 1
    assert seen[0].url.host == PUBLIC_IP
    assert seen[0].headers["host"] == "example.com:8443"
    assert seen[0].extensions["sni_hostname"] == "example.com"
    assert options and all(item["trust_env"] is False for item in options)
    assert all(item.get("proxy") is None for item in options)
    assert seen[0].headers["accept-encoding"] == "identity"


@pytest.mark.asyncio
async def test_untrusted_ambient_proxy_fails_before_dispatch(network, monkeypatch):
    dns, seen, _, options = network
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy-user:proxy-secret@127.0.0.1:3128")
    with pytest.raises(PublicDownloadError, match="policy") as caught:
        await download_public_input("https://example.com/", timeout=1, max_bytes=100)
    assert "proxy-secret" not in str(caught.value)
    assert not seen and not options
    dns.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_trusted_proxy_preserves_hostname_and_validates_redirects(
    network, monkeypatch
):
    dns, seen, replies, options = network
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    replies.extend(
        [
            httpx.Response(302, headers={"Location": "https://example.net/next"}),
            httpx.Response(302, headers={"Location": "https://127.0.0.1/private"}),
        ]
    )
    with pytest.raises(PublicDownloadError):
        await download_public_input("https://example.com/", timeout=1, max_bytes=100)
    assert dns.await_count == 2
    assert [request.url.host for request in seen] == ["example.com", "example.net"]
    assert any(item.get("proxy") is not None for item in options)
    assert all(item["trust_env"] is False for item in options)
    assert all("sni_hostname" not in request.extensions for request in seen)


@pytest.mark.asyncio
async def test_https_redirect_cannot_downgrade_to_public_http(network):
    _, seen, replies, _ = network
    replies.append(
        httpx.Response(302, headers={"Location": "http://example.net/input"})
    )
    with pytest.raises(PublicDownloadError):
        await download_public_input("https://example.com/", timeout=1, max_bytes=100)
    assert len(seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/",
        "file:///etc/passwd",
        "https://example.com:0/",
        "https://user:redirect-secret@example.com/",
    ],
)
async def test_every_redirect_is_checked_and_never_echoes_location(location, network):
    _, seen, replies, _ = network
    replies.append(httpx.Response(302, headers={"Location": location}))
    with pytest.raises(PublicDownloadError) as caught:
        await download_public_input("https://example.com/", timeout=1, max_bytes=100)
    assert len(seen) == 1
    assert location not in str(caught.value)


@pytest.mark.asyncio
async def test_redirect_cannot_use_a_rebound_dns_answer(network):
    dns, seen, replies, _ = network
    dns.side_effect = [
        dns.return_value,
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ]
    replies.append(httpx.Response(302, headers={"Location": "/next"}))
    with pytest.raises(PublicDownloadError):
        await download_public_input("https://example.com/", timeout=1, max_bytes=100)
    assert dns.await_count == 2
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_public_redirect_is_pinned_again(network):
    _, seen, replies, _ = network
    replies.extend(
        [
            httpx.Response(302, headers={"Location": "https://other.example/final"}),
            httpx.Response(200, content=b"ok"),
        ]
    )
    result = await download_public_input(
        "https://example.com/", timeout=1, max_bytes=100
    )
    assert result.url == "https://other.example/final"
    assert [r.url.host for r in seen] == [PUBLIC_IP, PUBLIC_IP]
    assert [r.headers["host"] for r in seen] == ["example.com", "other.example"]


@pytest.mark.asyncio
async def test_redirect_chain_is_bounded(network):
    _, seen, replies, _ = network
    replies.extend(httpx.Response(302, headers={"Location": "/loop"}) for _ in range(6))
    with pytest.raises(PublicDownloadError):
        await download_public_input("https://example.com/", timeout=1, max_bytes=100)
    assert len(seen) == 6


@pytest.mark.asyncio
async def test_declared_size_is_rejected_without_reading_body(network):
    _, _, replies, _ = network
    stream = ObservedStream([b"not-read"])
    replies.append(
        httpx.Response(200, headers={"Content-Length": "999"}, stream=stream)
    )
    with pytest.raises(PublicDownloadError):
        await download_public_input("https://example.com/", timeout=1, max_bytes=10)
    assert stream.read_count == 0
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [False, True])
async def test_streamed_decoded_size_is_bounded_and_response_closed(encoded, network):
    _, _, replies, _ = network
    payload = b"x" * 100000
    compressed = gzip.compress(payload)
    assert len(compressed) < 1000 < len(payload)
    stream = ObservedStream([compressed if encoded else payload])
    replies.append(
        httpx.Response(
            200, headers={"Content-Encoding": "gzip"} if encoded else {}, stream=stream
        )
    )
    with pytest.raises(PublicDownloadError):
        await download_public_input("https://example.com/", timeout=1, max_bytes=1000)
    assert stream.read_count == (0 if encoded else 1)
    assert stream.closed


@pytest.mark.asyncio
async def test_shared_downloader_decoded_limit_is_not_only_a_wire_length_limit(network):
    _, _, replies, _ = network
    compressed = gzip.compress(b"x" * 100000)
    assert len(compressed) < 1000
    stream = ObservedStream([compressed])
    replies.append(
        httpx.Response(
            200,
            headers={
                "Content-Encoding": "gzip",
                "Content-Length": str(len(compressed)),
            },
            stream=stream,
        )
    )
    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(ValueError, match="maximum size"):
            await security.fetch_public_http_bytes(
                client, "https://example.com/", timeout=1, max_content_bytes=1000
            )
    assert stream.read_count == 1
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [201, 204, 206])
async def test_non_200_success_status_is_rejected_before_read(status, network):
    _, _, replies, _ = network
    stream = ObservedStream([b"partial-or-unexpected"])
    replies.append(httpx.Response(status, stream=stream))
    with pytest.raises(PublicDownloadError, match=f"HTTP {status}"):
        await download_public_input("https://example.com/", timeout=1, max_bytes=100)
    assert stream.read_count == 0
    assert stream.closed


@pytest.mark.asyncio
async def test_dns_is_inside_the_total_deadline(network):
    dns, seen, _, _ = network

    async def delayed(*args, **kwargs):
        await asyncio.sleep(1)
        return dns.return_value

    dns.side_effect = delayed
    with pytest.raises(PublicDownloadError, match="timed out"):
        await download_public_input("https://example.com/", timeout=0.03, max_bytes=100)
    assert seen == []


@pytest.mark.asyncio
async def test_stream_body_is_inside_the_total_deadline(network):
    _, _, replies, _ = network
    stream = ObservedStream([b"late"], delay=1)
    replies.append(httpx.Response(200, stream=stream))
    with pytest.raises(PublicDownloadError, match="timed out"):
        await download_public_input("https://example.com/", timeout=0.03, max_bytes=100)
    assert stream.closed


@pytest.mark.asyncio
async def test_status_errors_do_not_echo_signed_urls_or_response_text(network):
    _, _, replies, _ = network
    replies.append(httpx.Response(403, text="server-secret"))
    with pytest.raises(PublicDownloadError) as caught:
        await download_public_input(
            "https://example.com/input?signature=caller-secret",
            timeout=1,
            max_bytes=100,
        )
    assert str(caught.value) == "Remote input returned HTTP 403"


@pytest.mark.asyncio
async def test_audio_json_uses_real_public_download_without_logging_signed_url(
    network, caplog
):
    _, seen, replies, _ = network
    replies.append(httpx.Response(200, json={"segments": [{"text": "hello"}]}))
    tts = FakeTTS()
    tool = AudioToolCore(tts_models={"fake": tts})
    with caplog.at_level(logging.INFO, logger="xagent.core.tools.core.audio_tool"):
        result = await tool.synthesize_speech_json(
            file_id="https://example.com/batch?signature=private-signature"
        )
    assert result["success"] is True
    assert tts.calls[0]["text"] == "hello"
    assert len(seen) == 1
    application_logs = "\n".join(
        record.message for record in caplog.records if record.name.startswith("xagent.")
    )
    assert "private-signature" not in application_logs


@pytest.mark.asyncio
async def test_audio_remote_json_limit_is_applied_before_tts(network, monkeypatch):
    _, _, replies, _ = network
    monkeypatch.setenv("XAGENT_REMOTE_JSON_DOWNLOAD_MAX_BYTES", "10")
    replies.append(httpx.Response(200, json={"segments": [{"text": "hello"}]}))
    tts = FakeTTS()
    result = await AudioToolCore(tts_models={"fake": tts}).synthesize_speech_json(
        file_id="https://example.com/batch"
    )
    assert result["success"] is False
    assert tts.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("use_symlink", [False, True])
async def test_audio_json_cannot_fall_back_to_a_host_file(tmp_path, use_symlink):
    workspace = TaskWorkspace("audio-boundary", base_dir=str(tmp_path / "uploads"))
    secret = tmp_path / "platform-secrets.json"
    secret.write_text(json.dumps({"segments": [{"text": "must-not-be-read"}]}))
    target = secret
    if use_symlink:
        target = workspace.input_dir / "host-link.json"
        target.symlink_to(secret)
    tts = FakeTTS()
    result = await AudioToolCore(
        tts_models={"fake": tts}, workspace=workspace
    ).synthesize_speech_json(file_id=str(target))
    assert result["success"] is False
    assert any(
        "workspace" in error.lower()
        or "outside" in error.lower()
        or "access" in error.lower()
        for error in result.get("errors", [str(result.get("error", ""))])
    )
    assert "must-not-be-read" not in json.dumps(result)
    assert tts.calls == []


@pytest.mark.asyncio
async def test_registered_workspace_json_still_synthesizes(tmp_path):
    workspace = TaskWorkspace("audio-valid", base_dir=str(tmp_path))
    source = workspace.input_dir / "batch.json"
    source.write_text(json.dumps({"segments": [{"text": "authorized"}]}))
    file_id = workspace.register_file(str(source))
    tts = FakeTTS()
    result = await AudioToolCore(
        tts_models={"fake": tts}, workspace=workspace
    ).synthesize_speech_json(file_id=file_id)
    assert result["success"] is True
    assert tts.calls[0]["text"] == "authorized"


@pytest.mark.asyncio
async def test_batch_voice_reference_cannot_bypass_workspace(tmp_path):
    workspace = TaskWorkspace("audio-reference", base_dir=str(tmp_path / "uploads"))
    secret = tmp_path / "host-secret.wav"
    secret.write_bytes(b"private-host-data")
    tts = FakeTTS()
    result = await AudioToolCore(
        tts_models={"fake": tts}, workspace=workspace
    ).synthesize_speech_json(
        json_data={"segments": [{"text": "hello", "reference_audio": str(secret)}]}
    )
    assert result["success"] is False
    assert any(
        "workspace" in error.lower()
        or "outside" in error.lower()
        or "access" in error.lower()
        for error in result["errors"]
    )
    assert tts.calls == []


@pytest.mark.asyncio
async def test_workspace_relative_json_and_voice_reference_reach_tts(tmp_path):
    workspace = TaskWorkspace("audio-relative", base_dir=str(tmp_path))
    reference = workspace.input_dir / "voice.wav"
    reference.write_bytes(b"authorized-voice")
    source = workspace.input_dir / "batch.json"
    source.write_text(
        json.dumps(
            {"segments": [{"text": "authorized", "reference_audio": "input/voice.wav"}]}
        )
    )
    tts = FakeTTS()
    result = await AudioToolCore(
        tts_models={"fake": tts}, workspace=workspace
    ).synthesize_speech_json(file_id="input/batch.json")
    assert result["success"] is True, result
    assert tts.calls[0]["reference_audio"] == str(reference)


@pytest.mark.asyncio
async def test_batch_preserves_provider_reference_url(tmp_path):
    workspace = TaskWorkspace("audio-provider-reference", base_dir=str(tmp_path))
    tts = FakeTTS()
    reference = "https://provider.example/reference.wav"
    result = await AudioToolCore(
        tts_models={"fake": tts}, workspace=workspace
    ).synthesize_speech_json(
        json_data={"segments": [{"text": "hello", "reference_audio": reference}]}
    )
    assert result["success"] is True, result
    assert tts.calls[0]["reference_audio"] == reference


@pytest.mark.asyncio
async def test_gemini_partial_reference_does_not_reach_billed_model_call(network):
    _, seen, replies, _ = network
    stream = ObservedStream([b"partial-image"])
    replies.append(httpx.Response(206, stream=stream))
    model = GeminiImageModel(model_name="gemini-3-pro-image", api_key="provider-secret")
    with pytest.raises(RuntimeError, match="HTTP 206"):
        await model.edit_image("https://example.com/reference", "edit")
    assert len(seen) == 1
    assert stream.read_count == 0
    assert stream.closed


@pytest.mark.asyncio
async def test_gemini_reference_is_public_only_and_does_not_receive_provider_key(
    network,
):
    _, seen, replies, _ = network
    replies.extend(
        [
            httpx.Response(
                200, content=b"image-bytes", headers={"Content-Type": "image/png"}
            ),
            httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "![Image](https://example.com/result.png)"}
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ]
                },
            ),
        ]
    )
    model = GeminiImageModel(
        model_name="gemini-3-pro-image",
        api_key="provider-secret",
        base_url="https://model.example/v1beta",
    )
    result = await model.edit_image("https://example.com/reference", "edit")
    assert result["image_url"] == "https://example.com/result.png"
    assert len(seen) == 2
    assert seen[0].url.host == PUBLIC_IP
    assert "provider-secret" not in str(seen[0].url)
    assert "authorization" not in seen[0].headers
    assert seen[1].url.host == PUBLIC_IP
    assert seen[1].headers["Host"] == "model.example"
    assert seen[1].headers["authorization"] == "Bearer provider-secret"
    part = json.loads(seen[1].content)["contents"][0]["parts"][0]["inlineData"]
    assert part == {
        "mimeType": "image/png",
        "data": base64.b64encode(b"image-bytes").decode(),
    }


@pytest.mark.asyncio
async def test_gemini_private_input_fails_before_billed_model_call(network):
    _, seen, _, _ = network
    model = GeminiImageModel(model_name="gemini-3-pro-image", api_key="provider-secret")
    with pytest.raises(RuntimeError, match="public-network"):
        await model.edit_image(
            "http://169.254.169.254/?signature=private-signature", "edit"
        )
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_bytes,timeout",
    [
        (0, 1),
        (-1, 1),
        (True, 1),
        (1.5, 1),
        (100, 0),
        (100, -1),
        (100, float("nan")),
        (100, float("inf")),
    ],
)
async def test_invalid_limits_fail_before_network(max_bytes, timeout, network):
    dns, seen, _, options = network
    with pytest.raises(ValueError, match="positive and finite"):
        await download_public_input(
            "https://example.com/", max_bytes=max_bytes, timeout=timeout
        )
    assert not options
    assert not seen
    dns.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("length", ["-1", "invalid"])
async def test_invalid_declared_length_is_rejected_without_body_read(length, network):
    _, _, replies, _ = network
    stream = ObservedStream([b"not-read"])
    replies.append(
        httpx.Response(200, headers={"Content-Length": length}, stream=stream)
    )
    with pytest.raises(PublicDownloadError):
        await download_public_input("https://example.com/", timeout=1, max_bytes=100)
    assert stream.read_count == 0
    assert stream.closed


@pytest.mark.asyncio
async def test_empty_body_is_rejected(network):
    _, _, replies, _ = network
    replies.append(httpx.Response(200, content=b""))
    with pytest.raises(PublicDownloadError):
        await download_public_input("https://example.com/", timeout=1, max_bytes=100)


@pytest.mark.asyncio
async def test_caller_cancellation_is_not_converted_to_download_failure(network):
    dns, seen, _, _ = network
    started = asyncio.Event()

    async def delayed(*args, **kwargs):
        started.set()
        await asyncio.Future()

    dns.side_effect = delayed
    task = asyncio.create_task(
        download_public_input("https://example.com/", timeout=10, max_bytes=100)
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not seen


@pytest.mark.asyncio
async def test_gemini_reference_size_limit_is_applied_before_provider_call(
    network, monkeypatch
):
    _, seen, replies, _ = network
    monkeypatch.setenv("XAGENT_IMAGE_DOWNLOAD_MAX_BYTES", "2")
    replies.append(httpx.Response(200, content=b"too-large"))
    model = GeminiImageModel(model_name="gemini-3-pro-image", api_key="provider-secret")
    with pytest.raises(RuntimeError, match="size policy"):
        await model.edit_image("https://example.com/reference", "edit")
    assert len(seen) == 1
    assert seen[0].method == "GET"


@pytest.mark.asyncio
@pytest.mark.parametrize("literal", [False, True])
async def test_public_ipv6_is_pinned_without_losing_brackets_or_host(literal, network):
    dns, seen, replies, _ = network
    address = "2606:4700:4700::1111"
    dns.return_value = [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, 8443, 0, 0))
    ]
    host = f"[{address}]" if literal else "example.com"
    replies.append(httpx.Response(200, content=b"ok"))
    await download_public_input(f"https://{host}:8443/input", timeout=1, max_bytes=10)
    assert seen[0].url.host == address
    assert seen[0].headers["host"] == f"{host}:8443"
    assert seen[0].extensions["sni_hostname"] == (address if literal else host)
