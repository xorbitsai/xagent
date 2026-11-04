import httpx
import pytest
from elevenlabs.client import AsyncElevenLabs, ElevenLabs

from xagent.core.model.public_elevenlabs import (
    close_public_elevenlabs_http_client,
    close_public_elevenlabs_sync_http_client,
    create_public_elevenlabs_client,
)
from xagent.core.utils.public_http_transport import UnsafePublicHttpTarget


def test_real_sync_sdk_cannot_connect_to_private_endpoint_and_owned_client_closes(
    monkeypatch,
):
    seen = []
    monkeypatch.setattr(
        httpx,
        "HTTPTransport",
        lambda **kwargs: httpx.MockTransport(lambda request: seen.append(request)),
    )
    client = create_public_elevenlabs_client(
        ElevenLabs,
        asynchronous=False,
        api_key="test-key",
        base_url="http://192.168.1.1/",
    )
    try:
        with pytest.raises(UnsafePublicHttpTarget):
            client.models.list()
        assert not seen
    finally:
        close_public_elevenlabs_sync_http_client(client)
    assert client._xagent_public_http_client.is_closed


@pytest.mark.asyncio
async def test_real_async_sdk_cannot_connect_to_private_endpoint_and_owned_client_closes(
    monkeypatch,
):
    seen = []
    monkeypatch.setattr(
        httpx,
        "AsyncHTTPTransport",
        lambda **kwargs: httpx.MockTransport(lambda request: seen.append(request)),
    )
    client = create_public_elevenlabs_client(
        AsyncElevenLabs,
        asynchronous=True,
        api_key="test-key",
        base_url="http://169.254.169.254/",
    )
    try:
        with pytest.raises(UnsafePublicHttpTarget):
            await client.models.list()
        assert not seen
    finally:
        await close_public_elevenlabs_http_client(client)
    assert client._xagent_public_http_client.is_closed
