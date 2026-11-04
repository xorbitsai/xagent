"""
Docker Manager for MCP Server Container Operations

This module provides Docker operations for MCP servers, handling container
lifecycle management, status monitoring, and resource usage tracking.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import docker.errors
import docker.models.containers
import requests
import requests.exceptions

from .data_config import ContainerInfo, ContainerStatus, ResourceUsage

logger = logging.getLogger(__name__)


class DockerManager:
    """
    Docker Manager for handling container operations per server.

    This class manages Docker clients for different servers, allowing each
    server to connect to its own Docker daemon if needed.
    """

    def __init__(self) -> None:
        self._clients: Dict[str, docker.DockerClient] = {}

    def get_client(self, docker_url: Optional[str] = None) -> docker.DockerClient:
        """
        Get or create a Docker client for the given URL.

        Args:
            docker_url: Docker daemon URL. If None, uses default.

        Returns:
            Docker client instance.
        """
        # Use a key for the client cache
        key = docker_url or "default"

        if key not in self._clients:
            try:
                self._clients[key] = docker.DockerClient(base_url=docker_url)
            except Exception as e:
                raise ValueError(
                    f"Failed to connect to Docker daemon at {docker_url or 'default'}: {e}"
                ) from e

        return self._clients[key]

    def close_client(self, docker_url: Optional[str] = None) -> None:
        """Close a Docker client."""
        key = docker_url or "default"
        if key in self._clients:
            try:
                self._clients[key].close()
            except Exception:
                pass  # Ignore errors when closing
            del self._clients[key]

    def close_all(self) -> None:
        """Close all Docker clients."""
        for client in self._clients.values():
            try:
                client.close()
            except Exception:
                pass  # Ignore errors when closing
        self._clients.clear()

    def start_container(
        self,
        name: str,
        docker_url: Optional[str],
        image: str,
        volumes: Optional[Union[Dict[str, Dict[str, str]], List[str]]] = None,
        bind_ports: Optional[Dict[str, Union[int, str]]] = None,
        environment: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
        restart_policy: str = "no",
    ) -> ContainerInfo:
        """
        Start a Docker container for an MCP server.

        Args:
            name: Server name (used for container naming).
            docker_url: Docker daemon URL.
            image: Docker image to use.
            volumes: Volume mappings.
            bind_ports: Port mappings (format: {'container_port': host_port}).
            environment: Environment variables (format: {'KEY': 'value'}).
            working_dir: Working directory inside the container.
            restart_policy: Container restart policy.

        Returns:
            ContainerInfo with container details.

        Raises:
            ValueError: If container creation or start fails.
        """
        docker_client = self.get_client(docker_url)
        container_name = name

        # Validate that we have image
        if not image:
            raise ValueError(f"Image must be provided for server '{name}'")

        # Check if container already exists
        try:
            existing_container = docker_client.containers.get(container_name)
            if existing_container.status == "running":
                logger.info(f"Container for server '{name}' is already running")
                return self._create_container_info(existing_container)
            else:
                # Start existing stopped container
                logger.info(f"Starting existing stopped container '{container_name}'")
                existing_container.start()
                logger.info(
                    f"Successfully started existing container '{container_name}'"
                )
                return self._create_container_info(existing_container)
        except docker.errors.NotFound:
            pass  # Container doesn't exist, proceed with creation

        # Method to start container
        try:
            assert image is not None, "image should not be None at this point"
            logger.info(f"Starting container with image: {image}")
            return self._start_with_config(
                name=name,
                image=image,
                volumes=volumes,
                bind_ports=bind_ports,
                environment=environment,
                working_dir=working_dir,
                restart_policy=restart_policy,
                docker_client=docker_client,
            )

        except Exception as e:
            logger.error(f"Failed to start container for server '{name}': {e}")
            raise ValueError(
                f"Failed to start container for server '{name}': {e}"
            ) from e

    def stop_container(self, name: str, docker_url: Optional[str]) -> bool:
        """
        Stop a Docker container.

        Args:
            name: Server name.
            docker_url: Docker daemon URL.

        Returns:
            True if successfully stopped.

        Raises:
            ValueError: If container not found or stop fails.
        """
        docker_client = self.get_client(docker_url)
        container_name = name

        try:
            container = docker_client.containers.get(container_name)
            container.stop(timeout=30)  # 30 second timeout
            logger.info(f"Stopped container '{container_name}'")
            return True
        except docker.errors.NotFound:
            logger.warning(f"Container '{container_name}' not found")
            return True  # Consider it stopped if not found
        except Exception as e:
            logger.error(f"Failed to stop container '{container_name}': {e}")
            raise ValueError(
                f"Failed to stop container for server '{name}': {e}"
            ) from e

    def restart_container(self, name: str, docker_url: Optional[str]) -> bool:
        """
        Restart a Docker container.

        Args:
            name: Server name.
            docker_url: Docker daemon URL.

        Returns:
            True if successfully restarted.

        Raises:
            ValueError: If container not found or restart fails.
        """
        docker_client = self.get_client(docker_url)
        container_name = name

        try:
            container = docker_client.containers.get(container_name)
            container.restart(timeout=30)  # 30 second timeout
            logger.info(f"Restarted container '{container_name}'")
            return True
        except docker.errors.NotFound:
            raise ValueError(f"Container for server '{name}' not found")
        except Exception as e:
            logger.error(f"Failed to restart container '{container_name}': {e}")
            raise ValueError(
                f"Failed to restart container for server '{name}': {e}"
            ) from e

    def get_container_status(
        self, name: str, docker_url: Optional[str]
    ) -> Tuple[ContainerStatus, Optional[ResourceUsage], Optional[str]]:
        """
        Get container status and resource usage.

        Args:
            name: Server name.
            docker_url: Docker daemon URL.

        Returns:
            Tuple of (status, resource_usage, uptime).
        """
        docker_client = self.get_client(docker_url)
        container_name = name

        try:
            container = docker_client.containers.get(container_name)
            status = ContainerStatus.from_docker_status(container.status)

            # Get resource usage
            resource_usage = None
            try:
                stats = container.stats(stream=False)
                if stats:
                    # Handle union type: Iterator[dict[str, Any]] | dict[str, Any]
                    if isinstance(stats, dict):
                        stats_dict = stats
                    else:
                        # Assume it's an iterator, get first item
                        stats_dict = next(stats, {})
                    if stats_dict:  # Check if dict is not empty
                        resource_usage = self._parse_resource_stats(stats_dict)
            except Exception as e:
                logger.warning(
                    f"Failed to get resource stats for container '{container_name}': {e}"
                )

            # Get uptime
            uptime = None
            try:
                started_at = container.attrs.get("State", {}).get("StartedAt")
                if started_at:
                    uptime = "running"  # Simplified implementation
            except Exception:
                pass

            return status, resource_usage, uptime

        except docker.errors.NotFound:
            return ContainerStatus.STOPPED, None, None
        except Exception as e:
            logger.error(f"Failed to get status for container '{container_name}': {e}")
            return ContainerStatus.UNKNOWN, None, None

    def get_container_logs(
        self, name: str, docker_url: Optional[str], lines: int = 100
    ) -> Optional[List[str]]:
        """
        Get container logs.

        Args:
            name: Server name.
            docker_url: Docker daemon URL.
            lines: Number of log lines to retrieve (1-1000).

        Returns:
            List of log lines or None if not available.
        """
        if not (1 <= lines <= 1000):
            raise ValueError(f"lines must be between 1 and 1000, got {lines}")

        docker_client = self.get_client(docker_url)
        container_name = name

        try:
            container = docker_client.containers.get(container_name)
            logs = container.logs(stdout=True, stderr=True, tail=lines)

            # Decode and split logs
            if isinstance(logs, bytes):
                log_lines = logs.decode("utf-8", errors="replace").split("\n")
            else:
                log_lines = str(logs).split("\n")

            # Filter out empty lines
            log_lines = [line for line in log_lines if line.strip()]
            return log_lines

        except docker.errors.NotFound:
            logger.warning(f"Container '{container_name}' not found")
            return []
        except Exception as e:
            logger.error(f"Failed to get logs for container '{container_name}': {e}")
            return []

    def check_external_server_health(
        self, url: str
    ) -> Tuple[ContainerStatus, Optional[str]]:
        """
        Check health of an external server.

        Args:
            url: Server URL to check.

        Returns:
            Tuple of (status, health_message).
        """
        try:
            headers = {}
            timeout = 5  # Default timeout

            # Handle different transport protocols
            if "/sse" in url.lower():
                # For SSE endpoints, try with appropriate headers
                headers = {
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
                response = requests.get(
                    url, headers=headers, timeout=timeout, stream=True
                )
            elif url.lower().startswith("ws://") or url.lower().startswith("wss://"):
                # For WebSocket endpoints, we can't use requests directly
                # Try to convert to HTTP and test basic connectivity
                http_url = url.replace("ws://", "http://").replace("wss://", "https://")
                response = requests.get(http_url, timeout=timeout)
            elif "/mcp" in url.lower() or "streamable_http" in url.lower():
                # For streamable_http MCP endpoints, send a proper initialize request
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                }
                # Send proper MCP initialize request
                initialize_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "health-check", "version": "1.0.0"},
                    },
                }
                try:
                    response = requests.post(
                        url, json=initialize_payload, headers=headers, timeout=timeout
                    )
                except requests.exceptions.RequestException:
                    # If POST fails, try GET for basic connectivity
                    response = requests.get(url, headers=headers, timeout=timeout)
            else:
                # For other HTTP-based endpoints, use standard health check
                response = requests.get(url, timeout=timeout)

            # Check response status
            if response.status_code == 200:
                return ContainerStatus.REACHABLE, "healthy"
            elif response.status_code < 500:
                # Any 2xx, 3xx, 4xx response means server is reachable
                return ContainerStatus.REACHABLE, "reachable"
            else:
                # 5xx server errors mean unreachable
                return (
                    ContainerStatus.UNREACHABLE,
                    f"server error (HTTP {response.status_code})",
                )

        except requests.exceptions.ConnectTimeout:
            return ContainerStatus.UNREACHABLE, "Connection timeout"
        except requests.exceptions.ConnectionError:
            return ContainerStatus.UNREACHABLE, "Connection refused"
        except requests.exceptions.RequestException as e:
            return ContainerStatus.UNREACHABLE, f"Request error: {e}"
        except Exception as e:
            return ContainerStatus.UNREACHABLE, f"Unexpected error: {e}"

    def _create_container_info(
        self, container: docker.models.containers.Container
    ) -> ContainerInfo:
        """Create ContainerInfo from Docker container object."""
        ports: Dict[str, List[Dict[str, str]]] = {}
        mounts: List[Dict[str, Any]] = []

        attrs: Dict[str, Any] = {}
        raw_attrs = getattr(container, "attrs", {})
        if isinstance(raw_attrs, dict):
            attrs = raw_attrs

        try:
            network_settings = attrs.get("NetworkSettings", {})
            raw_ports = network_settings.get("Ports") or {}
            if isinstance(raw_ports, dict):
                for container_port, bindings in raw_ports.items():
                    if not bindings:
                        continue
                    ports[container_port] = []
                    for binding in bindings:
                        if not isinstance(binding, dict):
                            continue
                        host_ip = str(binding.get("HostIp", ""))
                        host_port = str(binding.get("HostPort", ""))
                        ports[container_port].append(
                            {"HostIp": host_ip, "HostPort": host_port}
                        )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Failed to parse container ports for %s: %s", container.id, exc
            )

        try:
            raw_mounts = attrs.get("Mounts", [])
            if isinstance(raw_mounts, list):
                mounts = [m for m in raw_mounts if isinstance(m, dict)]
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Failed to parse container mounts for %s: %s", container.id, exc
            )

        return ContainerInfo(
            container_id=container.id,
            container_name=container.name,
            image=container.image.tags[0]
            if container.image and container.image.tags
            else (container.image.id if container.image else None),
            image_id=container.image.id if container.image else None,
            ports=ports,
            mounts=mounts,
        )

    def _parse_resource_stats(self, stats: Dict[str, Any]) -> Optional[ResourceUsage]:
        """Parse Docker stats into ResourceUsage object."""
        try:
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})
            memory_stats = stats.get("memory_stats", {})

            resource_usage = ResourceUsage()

            # Calculate CPU usage percentage
            cpu_delta = cpu_stats.get("cpu_usage", {}).get(
                "total_usage", 0
            ) - precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get(
                "system_cpu_usage", 0
            )
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (
                    (cpu_delta / system_delta)
                    * len(cpu_stats.get("cpu_usage", {}).get("percpu_usage", []))
                    * 100
                )
                resource_usage.cpu_usage = cpu_percent

            # Memory usage
            if memory_stats:
                resource_usage.memory_usage = memory_stats.get("usage", 0)

            return resource_usage
        except Exception:
            return None

    def _start_with_config(
        self,
        name: str,
        image: str,
        volumes: Optional[Union[Dict[str, Dict[str, str]], List[str]]],
        bind_ports: Optional[Dict[str, Union[int, str]]],
        environment: Optional[Dict[str, str]],
        working_dir: Optional[str],
        restart_policy: str,
        docker_client: docker.DockerClient,
    ) -> ContainerInfo:
        """
        Start container using configuration parameters.

        Args:
            name: Server name for container naming.
            image: Docker image to use.
            volumes: Volume mappings.
            ports: Port mappings (format: {'container_port': host_port}).
            environment: Environment variables (format: {'KEY': 'value'}).
            working_dir: Working directory inside the container.
            restart_policy: Container restart policy.
            docker_client: Docker client instance.

        Returns:
            ContainerInfo with container details.

        Raises:
            ValueError: If container creation or start fails.
        """
        container_name = name

        # Prepare container creation parameters
        container_kwargs: Dict[str, Any] = {
            "name": container_name,
            "image": image,
            "detach": True,
        }

        # Handle restart policy
        if restart_policy and restart_policy != "no":
            container_kwargs["restart_policy"] = {"Name": restart_policy}

        # Handle volumes
        if volumes:
            container_kwargs["volumes"] = volumes

        # Handle port mappings
        if bind_ports:
            container_kwargs["ports"] = bind_ports

        # Handle environment variables
        if environment:
            container_kwargs["environment"] = environment

        # Handle working directory
        if working_dir:
            container_kwargs["working_dir"] = working_dir

        # Create and start container
        container = docker_client.containers.create(**container_kwargs)
        container.start()

        logger.info(f"Started container '{container_name}' with ID {container.id}")
        return self._create_container_info(container)


__all__ = ["DockerManager"]
