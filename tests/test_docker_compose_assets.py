import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_SANDBOX_OVERLAY = REPO_ROOT / "docker" / "docker-compose.sandbox.docker.yml"


def _service_environment(service_name: str) -> dict[str, str]:
    entries = _overlay_services()[service_name]["environment"]
    return dict(entry.split("=", 1) for entry in entries)


def test_overlay_propagates_one_namespace_to_every_deployment_process():
    namespaces = {
        _service_environment(service)["XAGENT_SANDBOX_NAMESPACE"]
        for service in ("backend", "worker", "scheduler")
    }

    assert len(namespaces) == 1


def test_overlay_requires_a_resolved_compose_project_name():
    expected = (
        "${COMPOSE_PROJECT_NAME:?set COMPOSE_PROJECT_NAME to a stable unique value}"
    )

    for service in ("backend", "worker", "scheduler"):
        assert _service_environment(service)["XAGENT_SANDBOX_NAMESPACE"] == expected


MILVUS_ADDON = REPO_ROOT / "docker" / "docker-compose.milvus.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _milvus_services() -> dict[str, dict]:
    return yaml.safe_load(MILVUS_ADDON.read_text(encoding="utf-8"))["services"]


def _environment(service: dict) -> dict[str, str]:
    return dict(entry.split("=", 1) for entry in service["environment"])


def test_milvus_addon_points_every_deployment_process_at_milvus():
    services = _milvus_services()

    for name in ("backend", "worker", "scheduler"):
        environment = _environment(services[name])
        assert environment["XAGENT_VECTOR_BACKEND"] == "milvus"
        assert environment["MILVUS_URI"] == "http://milvus:19530"
    for name in ("backend", "worker"):
        assert services[name]["depends_on"] == {
            "milvus": {"condition": "service_healthy"}
        }
    assert "depends_on" not in services["scheduler"]


def test_milvus_addon_services_reach_each_other_by_service_name():
    services = _milvus_services()
    milvus = services["milvus"]
    environment = _environment(milvus)

    assert environment["ETCD_ENDPOINTS"] == "milvus-etcd:2379"
    assert environment["MINIO_ADDRESS"] == "milvus-minio:9000"
    assert (
        "-advertise-client-urls=http://milvus-etcd:2379"
        in (services["milvus-etcd"]["command"])
    )
    assert set(milvus["depends_on"]) == {"milvus-etcd", "milvus-minio"}
    assert all(
        dependency == {"condition": "service_healthy"}
        for dependency in milvus["depends_on"].values()
    )


def test_milvus_addon_pins_images_and_publishes_nothing():
    services = _milvus_services()
    engine = ("milvus-etcd", "milvus-minio", "milvus")

    assert services["milvus"]["image"] == "milvusdb/milvus:v2.6.25"
    for name in engine:
        assert not services[name]["image"].endswith((":latest", ":stable"))
        assert ":" in services[name]["image"]
        assert "ports" not in services[name]
        assert services[name]["networks"] == ["xagent_network"]
        assert services[name]["healthcheck"]["test"]


def test_milvus_addon_keeps_its_data_on_named_volumes():
    compose = yaml.safe_load(MILVUS_ADDON.read_text(encoding="utf-8"))
    mounted = {
        volume.split(":")[0]
        for name in ("milvus-etcd", "milvus-minio", "milvus")
        for volume in compose["services"][name]["volumes"]
    }

    assert mounted == set(compose["volumes"])


def test_the_milvus_ci_job_starts_the_addon_that_ships():
    ci = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    steps = {step["name"]: step for step in ci["jobs"]["pytest-milvus"]["steps"]}
    start = steps["Start Milvus standalone"]["run"]

    assert 'COMPOSE_FILE="docker-compose.yml:docker/docker-compose.milvus.yml:' in start
    assert '"19530:19530"' in start
    assert ci["jobs"]["pytest-milvus"]["env"]["MILVUS_URI"] == (
        "http://localhost:19530"
    )


def _overlay_services() -> dict[str, dict]:
    return yaml.safe_load(DOCKER_SANDBOX_OVERLAY.read_text(encoding="utf-8"))[
        "services"
    ]


def _storage_mount(service_name: str) -> dict:
    mounts = [
        volume
        for volume in _overlay_services()[service_name].get("volumes", [])
        if volume["target"] == "/root/.xagent"
    ]
    assert len(mounts) == 1
    return mounts[0]


def _services_mounting_storage() -> set[str]:
    base = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text("utf-8"))
    return {
        name
        for name, service in base["services"].items()
        if any(
            str(volume).split(":")[1] == "/root/.xagent"
            for volume in service.get("volumes", [])
        )
    }


def test_overlay_binds_one_host_storage_root_into_every_container_that_reads_it():
    source = "${XAGENT_HOST_STORAGE_ROOT:-/root/.xagent}"
    expected = _services_mounting_storage()

    assert expected
    bound = {
        name
        for name, service in _overlay_services().items()
        if any(v["target"] == "/root/.xagent" for v in service.get("volumes", []))
    }
    assert bound == expected
    for service in expected:
        mount = _storage_mount(service)
        assert (mount["type"], mount["source"]) == ("bind", source)


def test_overlay_keeps_nginx_storage_read_only_and_the_others_writable():
    assert _storage_mount("nginx")["read_only"] is True
    for service in ("backend", "worker", "scheduler"):
        assert not _storage_mount(service).get("read_only")


def _nginx_group_setup_command() -> str:
    nginx = _overlay_services()["nginx"]
    command = nginx["command"]

    assert command[:2] == ["/bin/sh", "-c"]
    return command[2].replace("$$", "$")


def _run_nginx_group_setup(
    tmp_path: Path, *, group_contents: str, nginx_contents: str
) -> tuple[subprocess.CompletedProcess[str], str, str, Path]:
    group_file = tmp_path / "group"
    nginx_conf = tmp_path / "nginx.conf"
    marker = tmp_path / "nginx-exec"
    bin_dir = tmp_path / "bin"
    fake_nginx = bin_dir / "nginx"

    group_file.write_text(group_contents, encoding="utf-8")
    nginx_conf.write_text(nginx_contents, encoding="utf-8")
    bin_dir.mkdir()
    fake_nginx.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" > "$XAGENT_NGINX_EXEC_MARKER"\n',
        encoding="utf-8",
    )
    fake_nginx.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "XAGENT_NGINX_GROUP_FILE": str(group_file),
            "XAGENT_NGINX_CONF": str(nginx_conf),
            "XAGENT_NGINX_EXEC_MARKER": str(marker),
        }
    )
    result = subprocess.run(
        ["/bin/sh", "-c", _nginx_group_setup_command()],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    return (
        result,
        group_file.read_text(encoding="utf-8"),
        nginx_conf.read_text(encoding="utf-8"),
        marker,
    )


def test_nginx_workers_use_existing_sandbox_group_as_primary_group(tmp_path):
    result, group_contents, nginx_contents, marker = _run_nginx_group_setup(
        tmp_path,
        group_contents="nginx:x:101:101:\nsandbox:x:1010:\n",
        nginx_contents="user nginx;\nevents {}\n",
    )

    assert result.returncode == 0, result.stderr
    assert group_contents.count(":x:1010:") == 1
    assert nginx_contents.startswith("user nginx sandbox;\n")
    assert marker.read_text(encoding="utf-8") == "-g daemon off;\n"


def test_nginx_workers_create_missing_sandbox_group(tmp_path):
    result, group_contents, nginx_contents, _marker = _run_nginx_group_setup(
        tmp_path,
        group_contents="nginx:x:101:101:\n",
        nginx_contents="user nginx;\nevents {}\n",
    )

    assert result.returncode == 0, result.stderr
    assert "xagent_sandbox:x:1010:\n" in group_contents
    assert nginx_contents.startswith("user nginx xagent_sandbox;\n")


def test_nginx_group_setup_fails_when_user_directive_is_missing(tmp_path):
    result, _group_contents, _nginx_contents, marker = _run_nginx_group_setup(
        tmp_path,
        group_contents="nginx:x:101:101:\nsandbox:x:1010:\n",
        nginx_contents="events {}\n",
    )

    assert result.returncode != 0
    assert "failed to find nginx user directive" in result.stderr
    assert not marker.exists()
