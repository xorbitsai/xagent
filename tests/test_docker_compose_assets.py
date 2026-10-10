from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_SANDBOX_OVERLAY = REPO_ROOT / "docker" / "docker-compose.sandbox.docker.yml"


def _service_environment(service_name: str) -> dict[str, str]:
    compose = yaml.safe_load(DOCKER_SANDBOX_OVERLAY.read_text(encoding="utf-8"))
    entries = compose["services"][service_name]["environment"]
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


def _storage_mount(service_name: str) -> dict:
    compose = yaml.safe_load(DOCKER_SANDBOX_OVERLAY.read_text(encoding="utf-8"))
    mounts = [
        volume
        for volume in compose["services"][service_name].get("volumes", [])
        if volume["target"] == "/root/.xagent"
    ]
    assert len(mounts) == 1
    return mounts[0]


def test_overlay_binds_one_host_storage_root_into_every_container_that_reads_it():
    source = "${XAGENT_HOST_STORAGE_ROOT:-/root/.xagent}"

    for service in ("backend", "worker", "scheduler", "nginx"):
        mount = _storage_mount(service)
        assert (mount["type"], mount["source"]) == ("bind", source)


def test_overlay_keeps_nginx_storage_read_only_and_the_others_writable():
    assert _storage_mount("nginx")["read_only"] is True
    for service in ("backend", "worker", "scheduler"):
        assert not _storage_mount(service).get("read_only")
