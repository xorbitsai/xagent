"""Private model exceptions are exact, operator-owned and transport-bound."""

import socket

import httpx
import pytest

from xagent import config
from xagent.core.model.network_policy import ModelNetworkPolicy
from xagent.core.model.public_http import (
    create_public_model_async_client,
    create_public_model_sync_client,
)
from xagent.core.utils.public_http_transport import (
    UnsafePublicHttpTarget,
    create_public_sync_client,
)
from xagent.core.utils.security import PrivateNetworkHostError


def answers(monkeypatch, values):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (
                socket.AF_INET6 if ":" in value else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (value, args[1]),
            )
            for value in values
        ],
    )


def approve(
    monkeypatch, endpoint="http://inference.example:9997", cidrs="127.0.0.1/32"
):
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_ENDPOINTS", endpoint)
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_CIDRS", cidrs)


def test_config_default_deny_and_operator_lists(monkeypatch):
    assert config.get_model_private_endpoints() == []
    assert config.get_model_private_cidrs() == []
    approve(
        monkeypatch,
        " http://localhost:9997, http://inference.example:9997 ",
        " 127.0.0.1/32, ::1/128 ",
    )
    assert config.get_model_private_endpoints() == [
        "http://localhost:9997",
        "http://inference.example:9997",
    ]
    assert config.get_model_private_cidrs() == ["127.0.0.1/32", "::1/128"]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9997/",
        "http://localhost:9997/",
        "http://10.1.2.3/",
        "https://[::1]/",
    ],
)
def test_private_defaults_remain_blocked(url):
    with pytest.raises(PrivateNetworkHostError):
        ModelNetworkPolicy().destination(url)


@pytest.mark.parametrize(
    "endpoint,cidrs,url,values",
    [
        (
            "http://inference.example:9997",
            "127.0.0.1/32",
            "http://inference.example:9998/",
            ["127.0.0.1"],
        ),
        (
            "https://inference.example:9997",
            "127.0.0.1/32",
            "http://inference.example:9997/",
            ["127.0.0.1"],
        ),
        (
            "http://inference.example:9997",
            "10.1.2.3/32",
            "http://inference.example:9997/",
            ["127.0.0.1"],
        ),
        (
            "http://inference.example:9997",
            "127.0.0.1/32",
            "http://inference.example:9997/",
            ["127.0.0.1", "10.0.0.1"],
        ),
        (
            "http://inference.example:9997",
            "",
            "http://inference.example:9997/",
            ["127.0.0.1"],
        ),
        ("", "127.0.0.1/32", "http://inference.example:9997/", ["127.0.0.1"]),
    ],
)
def test_both_exact_origin_and_every_dns_answer_must_be_approved(
    monkeypatch, endpoint, cidrs, url, values
):
    approve(monkeypatch, endpoint, cidrs)
    answers(monkeypatch, values)
    with pytest.raises(PrivateNetworkHostError):
        ModelNetworkPolicy().resolve_sync(url)


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254",
        "169.254.1.1",
        "fd00:ec2::254",
        "168.63.129.16",
        "100.100.100.200",
        "0.0.0.0",
        "224.0.0.1",
        "::ffff:169.254.169.254",
    ],
)
def test_metadata_and_unsafe_answers_cannot_be_approved(monkeypatch, address):
    approve(monkeypatch, cidrs="127.0.0.1/32,fc00::/7")
    answers(monkeypatch, [address])
    with pytest.raises(PrivateNetworkHostError):
        ModelNetworkPolicy().resolve_sync("http://inference.example:9997/")


@pytest.mark.parametrize(
    "value",
    [
        "http://localhost:0",
        "http://user:secret@localhost:9997",
        "http://localhost:9997/api",
        "http://localhost:9997?x=1",
        "http://*.example:9997",
    ],
)
def test_invalid_operator_origin_is_rejected(monkeypatch, value):
    approve(monkeypatch, value)
    with pytest.raises(ValueError):
        ModelNetworkPolicy()


@pytest.mark.asyncio
async def test_approved_direct_httpx_connections_still_pin_checked_ip(monkeypatch):
    approve(monkeypatch)
    answers(monkeypatch, ["127.0.0.1"])
    seen = []

    def transport(**kwargs):
        return httpx.MockTransport(
            lambda request: seen.append(request) or httpx.Response(200)
        )

    monkeypatch.setattr(httpx, "HTTPTransport", transport)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    with create_public_model_sync_client() as client:
        client.post("http://inference.example:9997/v1", content=b"model")
    async with create_public_model_async_client() as client:
        await client.post("http://inference.example:9997/v1", content=b"model")
    assert [request.url.host for request in seen] == ["127.0.0.1", "127.0.0.1"]
    assert all(request.headers["Host"] == "inference.example:9997" for request in seen)
    with create_public_sync_client() as client:
        with pytest.raises(UnsafePublicHttpTarget):
            client.get("http://inference.example:9997/")
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_aiohttp_uses_exact_request_origin_not_just_host_port(monkeypatch):
    import aiohappyeyeballs

    from xagent.core.model.public_http import create_public_model_aiohttp_session

    approve(monkeypatch, "https://inference.example:9997")
    answers(monkeypatch, ["127.0.0.1"])
    seen = []

    class Checked(Exception):
        pass

    async def connect(**kwargs):
        seen.append(kwargs["addr_infos"])
        raise Checked

    monkeypatch.setattr(aiohappyeyeballs, "start_connection", connect)
    async with create_public_model_aiohttp_session() as session:
        with pytest.raises(PrivateNetworkHostError):
            await session.get("http://inference.example:9997/")
        assert not seen
        with pytest.raises(Checked):
            await session.get("https://inference.example:9997/")
    assert seen[0][0][4][0] == "127.0.0.1"


def test_requests_model_policy_is_scoped_and_pins_approved_dns(monkeypatch):
    import io

    from urllib3.response import HTTPResponse

    from xagent.core.model import public_requests as model_requests
    from xagent.core.utils import public_requests

    approve(monkeypatch)
    answers(monkeypatch, ["127.0.0.1"])
    seen = []

    def urlopen(pool, *args, **kwargs):
        seen.append(pool.host)
        return HTTPResponse(body=io.BytesIO(b"{}"), status=200, preload_content=False)

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen)
    assert (
        model_requests.post("http://inference.example:9997/v1", json={}).status_code
        == 200
    )
    with pytest.raises(public_requests.RequestException):
        public_requests.post("http://inference.example:9997/v1", json={})
    assert seen == ["127.0.0.1"]


def test_private_xinference_default_can_be_explicitly_approved(monkeypatch):
    import io

    from urllib3.response import HTTPResponse

    from xagent.core.model.public_xinference import PublicXinferenceClient

    approve(monkeypatch, "http://localhost:9997")
    answers(monkeypatch, ["127.0.0.1"])
    seen = []

    def urlopen(pool, *args, **kwargs):
        seen.append(pool.host)
        return HTTPResponse(
            body=io.BytesIO(b'{"auth": false}'), status=200, preload_content=False
        )

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen)
    client = PublicXinferenceClient("http://localhost:9997")
    try:
        assert seen == ["127.0.0.1"]
    finally:
        client.close()


def test_private_bedrock_session_uses_same_operator_policy(monkeypatch):
    import io

    from botocore.awsrequest import AWSPreparedRequest
    from urllib3.response import HTTPResponse

    from xagent.core.utils.public_botocore import PublicBotocoreSession
    from xagent.core.utils.public_http_transport import UnsafePublicHttpTarget

    approve(monkeypatch)
    answers(monkeypatch, ["127.0.0.1"])
    seen = []

    def urlopen(pool, **kwargs):
        seen.append(pool.host)
        return HTTPResponse(body=io.BytesIO(b"{}"), status=200, preload_content=False)

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen)
    request = AWSPreparedRequest(
        method="POST",
        url="http://inference.example:9997/",
        headers={},
        body=b"{}",
        stream_output=True,
    )
    session = PublicBotocoreSession(timeout=1, _policy=ModelNetworkPolicy())
    generic = PublicBotocoreSession(timeout=1)
    try:
        session.send(request).raw.close()
        with pytest.raises(UnsafePublicHttpTarget):
            generic.send(request)
        assert seen == ["127.0.0.1"]
    finally:
        session.close()
        generic.close()


@pytest.mark.asyncio
async def test_video_reference_download_is_bounded_and_has_no_model_credentials(
    monkeypatch,
):
    from xagent.core.model.video.xinference import XinferenceVideoModel
    from xagent.core.utils.public_download import PublicDownloadError

    answers(monkeypatch, ["93.184.216.34"])
    monkeypatch.setenv("XAGENT_IMAGE_DOWNLOAD_MAX_BYTES", "4")
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(
                307, headers={"Location": "https://other.example/image.png"}
            )
        return httpx.Response(200, content=b"large", headers={"Content-Length": "5"})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    model = XinferenceVideoModel(
        "video", base_url="http://localhost:9997", api_key="never-forward"
    )
    with pytest.raises(PublicDownloadError):
        await model._read_media_bytes("https://image.example/image.png")
    assert len(seen) == 2
    assert all("Authorization" not in request.headers for request in seen)
    assert all(request.url.host == "93.184.216.34" for request in seen)
