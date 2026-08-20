"""Guards for the Docker host-port allocation used by the e2e container fixtures.

These run without a Docker daemon: they drive ``run_container_with_dynamic_ports``
against a fake client and assert the contract that removed the port-collision
race — the fixtures must never name a host port themselves.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.minio_harness import run_container_with_dynamic_ports


class _FakeContainer:
    def __init__(self, ports: dict[str, Any] | None) -> None:
        self.ports = ports
        self.removed = False
        self.reload_calls = 0

    def reload(self) -> None:
        self.reload_calls += 1

    def remove(self, force: bool = False) -> None:
        del force
        self.removed = True


class _FakeContainers:
    def __init__(self, container: _FakeContainer) -> None:
        self._container = container
        self.run_kwargs: dict[str, Any] = {}

    def run(
        self, image: str, command: str | None = None, **kwargs: Any
    ) -> _FakeContainer:
        self.run_kwargs = {"image": image, "command": command, **kwargs}
        return self._container


class _FakeClient:
    def __init__(self, container: _FakeContainer) -> None:
        self.containers = _FakeContainers(container)


def test_publishes_with_docker_chosen_loopback_ports() -> None:
    """The fixture must hand Docker the choice, not a port it picked itself.

    A concrete host port here would reintroduce the check-then-bind window that
    produced ``port is already allocated`` failures in CI.
    """
    container = _FakeContainer(
        {
            "9000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "51001"}],
            "9001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "51002"}],
        }
    )
    client = _FakeClient(container)

    returned, host_ports = run_container_with_dynamic_ports(
        client,
        "minio/minio",
        "server /data",
        name="fixture-container",
        container_ports=("9000/tcp", "9001/tcp"),
        tmpfs={"/data": "size=64m"},
    )

    assert returned is container
    assert host_ports == {"9000/tcp": 51001, "9001/tcp": 51002}
    assert client.containers.run_kwargs["ports"] == {
        "9000/tcp": ("127.0.0.1", None),
        "9001/tcp": ("127.0.0.1", None),
    }
    assert client.containers.run_kwargs["detach"] is True
    assert client.containers.run_kwargs["tmpfs"] == {"/data": "size=64m"}
    assert container.reload_calls == 1
    assert not container.removed


@pytest.mark.parametrize(
    "ports",
    [
        pytest.param(None, id="no-inspect-data"),
        pytest.param({}, id="no-published-ports"),
        pytest.param({"5432/tcp": []}, id="empty-binding-list"),
        pytest.param(
            {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}]}, id="unbound"
        ),
    ],
)
def test_unpublished_port_raises_and_removes_the_container(
    ports: dict[str, Any] | None,
) -> None:
    """A container we cannot reach is a hard error, and must not be leaked."""
    container = _FakeContainer(ports)
    client = _FakeClient(container)

    with pytest.raises(RuntimeError, match="did not publish a host port"):
        run_container_with_dynamic_ports(
            client,
            "postgres:16-bookworm",
            name="fixture-container",
            container_ports=("5432/tcp",),
        )

    assert container.removed, "the unusable container must be cleaned up"
