"""Unit tests for DNS-resolving egress authorization (Phase 3).

A fake resolver is injected so these stay pure unit tests with no real DNS.
"""

from __future__ import annotations

import pytest

from xagent.core.ssh import SshError, SshErrorCode
from xagent.core.ssh.egress import EgressPolicyConfig
from xagent.core.ssh.egress_io import resolve_and_authorize


def _resolver(*addrs: str):
    async def resolve(hostname: str, port: int) -> list[str]:
        return list(addrs)

    return resolve


async def test_all_public_addresses_allowed() -> None:
    ips = await resolve_and_authorize(
        "example.com", 22, EgressPolicyConfig(), resolver=_resolver("93.184.216.34")
    )
    assert ips == ["93.184.216.34"]


async def test_any_private_address_denies_whole_host() -> None:
    # An attacker adding a private A record must not let the connection proceed:
    # if ANY resolved address is denied, the host is rejected.
    with pytest.raises(SshError) as exc:
        await resolve_and_authorize(
            "rebind.example",
            22,
            EgressPolicyConfig(),
            resolver=_resolver("93.184.216.34", "10.0.0.5"),
        )
    assert exc.value.code == SshErrorCode.EGRESS_DENIED


async def test_no_addresses_denied() -> None:
    with pytest.raises(SshError) as exc:
        await resolve_and_authorize(
            "nxdomain.example", 22, EgressPolicyConfig(), resolver=_resolver()
        )
    assert exc.value.code == SshErrorCode.EGRESS_DENIED
