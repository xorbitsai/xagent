"""Exercise the installed BoxLite SDK without starting a VM or requiring KVM."""

from __future__ import annotations

import asyncio
import ipaddress
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

try:
    import boxlite
except ImportError:
    pytest.skip("BoxLite SDK unavailable on this platform", allow_module_level=True)

from xagent.sandbox.base import SandboxConfig, SandboxInfo, SandboxTemplate
from xagent.sandbox.boxlite_sandbox import (
    PUBLIC_EGRESS_NETWORKS,
    BoxliteSandboxService,
    MemBoxliteStore,
)


def test_public_egress_ranges_exclude_local_and_special_destinations():
    allowed = [ipaddress.ip_network(network) for network in PUBLIC_EGRESS_NETWORKS]

    for address in (
        "10.0.0.1",
        "100.64.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "172.16.0.1",
        "192.168.0.1",
        "198.18.0.1",
        "::1",
        "2001:2::1",
        "2001:db8::1",
        "2002::1",
        "3fff::1",
        "fc00::1",
        "fe80::1",
    ):
        candidate = ipaddress.ip_address(address)
        assert not any(
            candidate.version == network.version and candidate in network
            for network in allowed
        )

    for address in ("1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"):
        candidate = ipaddress.ip_address(address)
        assert any(
            candidate.version == network.version and candidate in network
            for network in allowed
        )


@pytest.fixture
def service_and_runtime():
    metadata = SimpleNamespace(
        name="sdk-contract-test",
        image="python:3.12-slim",
        cpus=1,
        memory_mib=256,
        state=SimpleNamespace(status="running"),
        created_at="2026-01-01T00:00:00+00:00",
    )
    raw_box = SimpleNamespace(
        __aenter__=AsyncMock(),
        info=AsyncMock(return_value=metadata),
    )
    runtime = SimpleNamespace(
        get=AsyncMock(return_value=None),
        get_or_create=AsyncMock(return_value=(raw_box, True)),
    )
    # Bypass only native runtime initialization. The real SimpleBox constructor,
    # start() and async info() methods still run against this fake runtime.
    service = object.__new__(BoxliteSandboxService)
    service._runtime = runtime
    service._store = MemBoxliteStore()
    service._locks = {}
    service._locks_lock = asyncio.Lock()
    return service, runtime, raw_box, metadata


@pytest.mark.asyncio
async def test_creation_awaits_simple_box_info_and_persists_metadata(
    service_and_runtime,
):
    service, runtime, raw_box, metadata = service_and_runtime

    sandbox = await service.get_or_create(
        metadata.name,
        template=SandboxTemplate(image=metadata.image),
        config=SandboxConfig(cpus=1, memory=256),
    )

    assert isinstance(sandbox._box, boxlite.SimpleBox)
    raw_box.__aenter__.assert_awaited_once()
    raw_box.info.assert_awaited_once()
    runtime.get_or_create.assert_awaited_once()
    persisted = service._store.get_info(metadata.name)
    assert persisted is sandbox._info
    assert persisted.created_at == metadata.created_at
    assert persisted.state == "running"


@pytest.mark.asyncio
async def test_missing_store_metadata_awaits_native_box_info(service_and_runtime):
    service, runtime, raw_box, metadata = service_and_runtime
    runtime.get.return_value = raw_box
    runtime.get_or_create.return_value = (raw_box, False)

    sandbox = await service.get_or_create(metadata.name)

    raw_box.info.assert_awaited_once()
    assert sandbox._info.created_at == metadata.created_at
    assert sandbox._info.template.image == metadata.image
    assert sandbox._info.config.memory == metadata.memory_mib
    assert sandbox._info.state == "running"


@pytest.mark.asyncio
async def test_reuse_preserves_stored_configuration(service_and_runtime):
    service, runtime, raw_box, metadata = service_and_runtime
    runtime.get.return_value = raw_box
    runtime.get_or_create.return_value = (raw_box, False)
    stored = SandboxInfo(
        name=metadata.name,
        state="stopped",
        template=SandboxTemplate(image=metadata.image),
        config=SandboxConfig(network_isolated=True),
        created_at=metadata.created_at,
    )
    service._store.add_info(metadata.name, stored)

    sandbox = await service.get_or_create(metadata.name)

    assert sandbox._info is stored
    raw_box.info.assert_not_awaited()
    options = runtime.get_or_create.await_args.args[0]
    assert options.network.outbound.mode == "disabled"
    assert options.network.inbound.mode == "disabled"


@pytest.mark.asyncio
@pytest.mark.parametrize("isolated", [True, False, None])
async def test_guest_network_policy_uses_native_box_options(
    service_and_runtime, isolated
):
    service, runtime, _, metadata = service_and_runtime

    await service.get_or_create(
        metadata.name,
        template=SandboxTemplate(image=metadata.image),
        config=SandboxConfig(network_isolated=isolated),
    )

    options = runtime.get_or_create.await_args.args[0]
    assert isinstance(options, boxlite.BoxOptions)
    if isolated:
        assert options.network.outbound.mode == "disabled"
        assert options.network.inbound.mode == "disabled"
        assert options.advanced is None
    else:
        assert options.network.outbound.mode == "enabled"
        assert set(options.network.outbound.allow_net) == set(PUBLIC_EGRESS_NETWORKS)
        assert options.network.inbound.mode == "disabled"
        assert "10.0.0.0/8" not in options.network.outbound.allow_net
        assert "127.0.0.0/8" not in options.network.outbound.allow_net
        assert "fc00::/7" not in options.network.outbound.allow_net


@pytest.mark.asyncio
async def test_isolated_guest_rejects_published_ports_before_creation(
    service_and_runtime,
):
    service, runtime, _, metadata = service_and_runtime
    with pytest.raises(ValueError, match="Published ports.*isolated"):
        await service.get_or_create(
            metadata.name,
            config=SandboxConfig(network_isolated=True, ports=[(8080, 80)]),
        )
    runtime.get_or_create.assert_not_awaited()
