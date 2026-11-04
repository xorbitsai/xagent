"""Pin FakeSandboxService's contract against the real SandboxService base.

These are pins, not behavior tests: they exist so that any future edit to
FakeSandboxService (or to SandboxService itself) that would silently change
what manager tests are actually exercising fails loudly here first, rather
than showing up as a mysteriously-misrouted assertion three files away.
"""

from __future__ import annotations

import inspect as std_inspect

import pytest

from tests.web.sandbox_fakes import FakeSandboxService
from xagent.sandbox.base import SandboxReconcileUnsupportedError, SandboxService


class TestSpecReconciliationDefaults:
    """A plain FakeSandboxService() must behave like a real legacy-only
    backend on the four spec-reconciliation methods: never silently
    routed as capable."""

    @pytest.mark.asyncio
    async def test_supports_runtime_spec_defaults_to_real_false(self):
        service = FakeSandboxService()
        result = await service.supports_runtime_spec()

        assert result is False
        # Not just falsy: an unconfigured AsyncMock() would also satisfy
        # `not result`, but its return value is a truthy child Mock, which
        # is exactly the silent-misroute failure mode this pin exists to
        # catch.
        assert type(result) is bool

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method_name,args",
        [
            ("inspect", ("box",)),
            ("create", ("box", None, None)),
            ("start_existing", ("box",)),
            ("stop_existing", ("box",)),
        ],
    )
    async def test_unconfigured_reconciliation_methods_raise(self, method_name, args):
        service = FakeSandboxService()
        method = getattr(service, method_name)

        with pytest.raises(SandboxReconcileUnsupportedError):
            await method(*args)

    def test_reconciliation_methods_are_not_overridden_at_class_level(self):
        """FakeSandboxService must not shadow the four reconciliation
        methods (or supports_runtime_spec) at the class level: only
        per-instance opt-in (via constructor kwargs) may do so."""
        for name in (
            "supports_runtime_spec",
            "inspect",
            "create",
            "start_existing",
            "stop_existing",
        ):
            assert getattr(FakeSandboxService, name) is getattr(SandboxService, name)

    @pytest.mark.asyncio
    async def test_runtime_spec_supported_kwarg_opts_in_without_touching_class(self):
        opted_in = FakeSandboxService(runtime_spec_supported=True)
        default = FakeSandboxService()

        assert await opted_in.supports_runtime_spec() is True
        assert await default.supports_runtime_spec() is False


class TestLegacyLifecycleSignatures:
    """FakeSandboxService's concrete overrides of the legacy abstract
    methods must keep SandboxService's async signature shape (mirrors the
    DockerSandboxService pin in test_docker_lifecycle_api.py)."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "get_or_create",
            "list_sandboxes",
            "delete",
            "supports_snapshots",
            "create_snapshot",
            "list_snapshots",
            "delete_snapshot",
        ],
    )
    def test_fake_override_matches_base_signature(self, method_name):
        base_method = getattr(SandboxService, method_name)
        fake_method = getattr(FakeSandboxService, method_name)

        assert std_inspect.iscoroutinefunction(fake_method)

        base_params = list(std_inspect.signature(base_method).parameters)
        fake_params = list(std_inspect.signature(fake_method).parameters)
        assert fake_params == base_params


class TestLegacyLifecycleDefaults:
    """The legacy methods keep full unittest.mock call-tracking while
    defaulting to an in-memory container registry."""

    @pytest.mark.asyncio
    async def test_get_or_create_tracks_containers_and_calls(self):
        service = FakeSandboxService()

        sandbox = await service.get_or_create("box::1")

        assert sandbox.name == "box::1"
        assert "box::1" in service.containers
        service.get_or_create.assert_awaited_once_with("box::1")

    @pytest.mark.asyncio
    async def test_delete_untracks_and_records(self):
        service = FakeSandboxService(("box::1",))

        await service.delete("box::1")

        assert "box::1" not in service.containers
        assert service.deleted == ["box::1"]
        service.delete.assert_awaited_once_with("box::1")

    @pytest.mark.asyncio
    async def test_list_sandboxes_reflects_current_containers(self):
        service = FakeSandboxService(("box::1", "box::2"))

        listed = await service.list_sandboxes()

        assert sorted(sb.name for sb in listed) == ["box::1", "box::2"]

    @pytest.mark.asyncio
    async def test_return_value_override_takes_precedence_over_default(self):
        """A test can still fully replace the default behavior, matching
        how the ad hoc AsyncMock() fakes it replaces were used."""
        service = FakeSandboxService()
        service.list_sandboxes.return_value = []

        assert await service.list_sandboxes() == []
        assert service.containers == set()

    @pytest.mark.asyncio
    async def test_supports_snapshots_defaults_false(self):
        service = FakeSandboxService()

        assert await service.supports_snapshots() is False
