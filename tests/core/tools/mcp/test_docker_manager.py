from types import SimpleNamespace
from typing import cast

import docker.models.containers

from xagent.core.tools.core.mcp.data_config import ContainerInfo
from xagent.core.tools.core.mcp.docker_manager import DockerManager


class DummyImage:
    def __init__(self, tags=None, image_id="sha256:dummy") -> None:
        self.tags = tags or ["mineru-mcp:latest"]
        self.id = image_id


def test_create_container_info_includes_ports_and_mounts():
    manager = DockerManager()

    container = cast(
        docker.models.containers.Container,
        SimpleNamespace(
            id="container-id",
            name="mineru-mcp",
            image=DummyImage(),
            attrs={
                "NetworkSettings": {
                    "Ports": {
                        "8001/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8001"}],
                        "9000/tcp": None,
                    }
                },
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": "/host/downloads",
                        "Destination": "/app/downloads",
                        "Mode": "rw",
                        "RW": True,
                    }
                ],
            },
        ),
    )
    info: ContainerInfo = manager._create_container_info(container)

    assert info.container_id == "container-id"
    assert info.container_name == "mineru-mcp"
    assert info.image == "mineru-mcp:latest"
    assert info.image_id == "sha256:dummy"
    assert info.ports == {"8001/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8001"}]}
    assert info.mounts == [
        {
            "Type": "bind",
            "Source": "/host/downloads",
            "Destination": "/app/downloads",
            "Mode": "rw",
            "RW": True,
        }
    ]
