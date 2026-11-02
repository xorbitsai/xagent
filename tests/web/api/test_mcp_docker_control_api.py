"""Regression coverage for user-facing MCP Docker control boundaries."""

import pytest

import xagent.web.api.mcp as mcp_api
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


def test_create_rejects_internal_docker_management_fields() -> None:
    headers = _admin_headers()
    forbidden_values = {
        "managed": "internal",
        "docker_url": "tcp://attacker.example:2375",
        "docker_image": "attacker/image:latest",
        "docker_environment": {"SECRET": "value"},
        "docker_working_dir": "/host",
        "volumes": ["/:/host"],
        "bind_ports": {"80/tcp": 8080},
        "restart_policy": "always",
        "auto_start": True,
        # Prefix matching keeps future Docker options closed by default.
        "docker_future_option": "value",
    }

    for index, (field_name, field_value) in enumerate(forbidden_values.items()):
        server_name = f"forbidden-docker-{index}"
        response = client.post(
            "/api/mcp/servers",
            headers=headers,
            json={
                "name": server_name,
                "transport": "sse",
                "config": {
                    "url": "https://mcp.example.test/sse",
                    field_name: field_value,
                },
            },
        )

        assert response.status_code == 400, response.text
        assert (
            "Internal Docker management fields are not available"
            in response.json()["detail"]
        )
        assert field_name in response.json()["detail"]

        db = _direct_db_session()
        try:
            assert (
                db.query(MCPServer).filter(MCPServer.name == server_name).first()
                is None
            )
        finally:
            db.close()


def test_update_rejects_docker_management_fields_without_changing_server() -> None:
    headers = _admin_headers()
    created = client.post(
        "/api/mcp/servers",
        headers=headers,
        json={
            "name": "safe-remote",
            "transport": "sse",
            "config": {"url": "https://mcp.example.test/sse"},
        },
    )
    assert created.status_code == 201, created.text
    server_id = created.json()["id"]

    response = client.put(
        f"/api/mcp/servers/{server_id}",
        headers=headers,
        json={
            "config": {
                "url": "https://changed.example.test/sse",
                "docker_url": "tcp://attacker.example:2375",
            }
        },
    )

    assert response.status_code == 400, response.text
    assert "docker_url" in response.json()["detail"]

    db = _direct_db_session()
    try:
        server = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert server.managed == "external"
        assert server.url == "https://mcp.example.test/sse"
        assert server.docker_url is None
    finally:
        db.close()


def test_update_accepts_editable_config_after_response_metadata_is_projected_out() -> (
    None
):
    headers = _admin_headers()
    created = client.post(
        "/api/mcp/servers",
        headers=headers,
        json={
            "name": "editable-remote",
            "transport": "sse",
            "config": {"url": "https://mcp.example.test/sse"},
        },
    )
    assert created.status_code == 201, created.text
    server_id = created.json()["id"]

    response = client.put(
        f"/api/mcp/servers/{server_id}",
        headers=headers,
        json={"config": {"url": "https://rotated.example.test/sse"}},
    )

    assert response.status_code == 200, response.text
    db = _direct_db_session()
    try:
        server = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert server.managed == "external"
        assert server.url == "https://rotated.example.test/sse"
    finally:
        db.close()


def test_editing_legacy_internal_row_drops_trusted_docker_fields() -> None:
    headers = _admin_headers()
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        server = MCPServer(
            name="legacy-internal-edit",
            managed="internal",
            transport="sse",
            url="http://127.0.0.1:9000/sse",
            docker_url="unix:///var/run/docker.sock",
            docker_image="legacy/image:latest",
            restart_policy="no",
            auto_start=False,
        )
        db.add(server)
        db.flush()
        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.commit()
        server_id = server.id
    finally:
        db.close()

    response = client.put(
        f"/api/mcp/servers/{server_id}",
        headers=headers,
        json={"config": {"url": "https://converted.example.test/sse"}},
    )

    assert response.status_code == 200, response.text
    db = _direct_db_session()
    try:
        server = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert server.managed == "external"
        assert server.url == "https://converted.example.test/sse"
        assert server.docker_url is None
        assert server.docker_image is None
        assert server.auto_start is False
    finally:
        db.close()


def test_internal_logs_never_initialize_docker_manager(monkeypatch) -> None:
    headers = _admin_headers()

    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        server = MCPServer(
            name="legacy-internal",
            description="Persisted before the API boundary was enforced",
            managed="internal",
            transport="sse",
            url="http://127.0.0.1:9000/sse",
            docker_url="tcp://attacker.example:2375",
            docker_image="attacker/image:latest",
            restart_policy="no",
            auto_start=False,
        )
        db.add(server)
        db.flush()
        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.commit()
        server_id = server.id
    finally:
        db.close()

    def fail_if_initialized(*_args, **_kwargs):
        raise AssertionError("internal MCP logs must not initialize Docker manager")

    monkeypatch.setattr(mcp_api, "DatabaseMCPServerManager", fail_if_initialized)

    response = client.get(f"/api/mcp/servers/{server_id}/logs", headers=headers)

    assert response.status_code == 403, response.text
    assert response.json() == {"detail": "Internal MCP server logs are disabled"}
