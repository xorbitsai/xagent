"""Tests for shared security helpers."""

import socket
from unittest.mock import patch

import pytest

from xagent.core.utils.security import (
    PrivateNetworkHostError,
    redact_sensitive_text,
    redact_url_credentials_for_logging,
    reject_private_network_host,
    validate_public_http_url,
)


def test_reject_private_network_host_rejects_cgnat_range() -> None:
    with pytest.raises(PrivateNetworkHostError):
        reject_private_network_host("100.64.0.1")


@pytest.mark.asyncio
async def test_validate_public_http_url_rejects_private_dns_result() -> None:
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    with patch("socket.getaddrinfo", return_value=resolved):
        with pytest.raises(PrivateNetworkHostError):
            await validate_public_http_url("https://public.example/logo.png")


@pytest.mark.asyncio
async def test_validate_public_http_url_accepts_only_public_dns_results() -> None:
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    with patch("socket.getaddrinfo", return_value=resolved):
        await validate_public_http_url("https://public.example/logo.png")


def test_redact_url_credentials_for_logging_masks_sensitive_query_values() -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSySecret&v=1"
    redacted = redact_url_credentials_for_logging(url)

    assert "AIzaSySecret" not in redacted
    assert "key=%2A%2A%2A" in redacted
    assert "v=1" in redacted


def test_redact_sensitive_text_masks_bearer_and_header_keys() -> None:
    text = (
        "Authorization: Bearer sk-secret-value "
        "x-goog-api-key: AIzaSyHeaderSecret "
        "url=https://example.com/path?api_key=my_api_key"
    )
    redacted = redact_sensitive_text(text)

    assert "sk-secret-value" not in redacted
    assert "AIzaSyHeaderSecret" not in redacted
    assert "my_api_key" not in redacted


def test_redact_sensitive_text_masks_assignment_style_secrets() -> None:
    text = "api_key=sk-super-secret timeout=30"
    redacted = redact_sensitive_text(text)

    assert "sk-super-secret" not in redacted
    assert "api_key=***" in redacted
    assert "timeout=30" in redacted
