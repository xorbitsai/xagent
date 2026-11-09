"""The key-based stdio branch must hand the built-in Jira Ticket server its
upload allowlist, and only that server.

Unlike OAuth connectors, key-based stdio servers get no file allowlist from
_build_mcp_server_config. Without one the Jira attachment guard falls back to
the process working directory, which is not a task workspace.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.mcp import MCPServer
from xagent.web.tools.config import WebToolConfig

ENV_VAR = "XAGENT_JIRA_FILE_ALLOWED_DIRS"
PROVENANCE = {"registry": "xagent", "app_id": "jira-ticket", "version": 1}


def _server(name: str = "jira-ticket", auth: Any = None) -> MCPServer:
    return MCPServer(
        id=1,
        name=name,
        description="d",
        managed="external",
        transport="stdio",
        command="python",
        args=["-m", "xagent.web.tools.mcp.jira_ticket"],
        auth=auth,
    )


def _build(cfg: WebToolConfig, server: MCPServer) -> dict[str, Any]:
    built = asyncio.run(
        cfg._build_mcp_server_config(
            server=server,
            user_env_by_id={},
            shared_env_by_id={},
            env_source_by_id={},
        )
    )
    return built["config"]


def _cfg(tmp_path: Path, **kwargs: Any) -> WebToolConfig:
    return WebToolConfig(
        db=None,
        request=None,
        user_id=1,
        task_id="task-123",
        workspace_base_dir=str(tmp_path),
        **kwargs,
    )


def test_builtin_jira_ticket_server_gets_the_task_workspace_allowlist(
    tmp_path: Path,
) -> None:
    config = _build(_cfg(tmp_path), _server(auth={"builtin_provenance": PROVENANCE}))
    expected = str((tmp_path / "task-123").resolve())
    assert json.loads(config["env"][ENV_VAR]) == [expected]


def test_builtin_jira_ticket_server_without_a_workspace_denies_all_uploads() -> None:
    """With no allowlist the connector would fall back to its working
    directory, so the hook sends an empty one (deny everything) instead."""
    cfg = WebToolConfig(db=None, request=None, user_id=1)
    config = _build(cfg, _server(auth={"builtin_provenance": PROVENANCE}))
    assert json.loads(config["env"][ENV_VAR]) == []


@pytest.mark.parametrize(
    "server",
    [
        # Same name, no provenance: a custom server that merely shares the name.
        _server(auth=None),
        _server(auth={}),
        # Marker for some other built-in.
        _server(auth={"builtin_provenance": {**PROVENANCE, "app_id": "shopify"}}),
        # Marker present but the server is not named for it.
        _server(name="other", auth={"builtin_provenance": PROVENANCE}),
        _server(name="shopify", auth=None),
    ],
)
def test_other_servers_get_no_jira_allowlist(tmp_path: Path, server: MCPServer) -> None:
    config = _build(_cfg(tmp_path), server)
    assert ENV_VAR not in (config.get("env") or {})


@pytest.mark.parametrize("name", ["jira-ticket", "Jira Ticket", " JIRA-TICKET "])
def test_name_is_matched_by_canonical_identity(tmp_path: Path, name: str) -> None:
    config = _build(
        _cfg(tmp_path), _server(name=name, auth={"builtin_provenance": PROVENANCE})
    )
    assert ENV_VAR in config["env"]


def test_connect_flow_names_and_stamps_the_server_the_hook_recognizes(
    tmp_path: Path,
) -> None:
    """End to end through the real catalog connect: the row it provisions is
    exactly what the hook accepts (name and stamped provenance), and the
    upload table finds the tool under that server's name."""
    from xagent.core.tools.adapters.vibe.mcp_adapter import _durable_upload_fields
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
    from xagent.web.models.database import Base
    from xagent.web.models.public_mcp import PublicMCPApp
    from xagent.web.models.user import User

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    user = User(username="alice", id=1, password_hash="x")
    db.add(user)
    row = get_builtin_public_mcp_app("jira-ticket")
    assert row is not None
    # The row ships hidden, and connect refuses a hidden app; this is a
    # deployment that has shown it after live verification (#3661).
    db.add(PublicMCPApp(**{**row, "is_visible_in_connector": True}))
    db.commit()

    connect_mcp_app(
        "jira-ticket",
        MCPAppConnectRequest(
            env={
                "JIRA_TICKET_SITE": "acme",
                "JIRA_TICKET_EMAIL": "intake@example.com",
                "JIRA_TICKET_API_TOKEN": "tok",
                "JIRA_TICKET_PROJECT_KEY": "SUP",
            }
        ),
        current_user=user,
        db=db,
    )

    server = db.query(MCPServer).one()
    assert server.auth == {"builtin_provenance": PROVENANCE}
    config = _build(
        _cfg(tmp_path),
        _server(name=str(server.name), auth=server.auth),
    )
    assert ENV_VAR in config["env"]
    assert _durable_upload_fields(str(server.name), "jira_ticket_create") == (
        "attachment_path",
        "attachment_path_2",
        "attachment_path_3",
    )
    db.close()


def test_staged_upload_is_accepted_through_the_real_hook_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staged file -> task workspace path -> hook-built env var -> read guard.

    Only the file-id lookup is stubbed (it needs a database); staging, the
    config hook and the connector's read guard are the real ones.
    """
    from xagent.core.workspace import TaskWorkspace
    from xagent.web.tools.mcp import jira_attachment

    workspace = TaskWorkspace(id="task-123", base_dir=str(tmp_path))
    source = workspace.input_dir / "report.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"hello")
    monkeypatch.setattr(workspace, "resolve_file_id_detached", lambda _: source)
    try:
        staged = workspace.stage_file_for_external_upload("file-id")

        config = _build(
            _cfg(tmp_path), _server(auth={"builtin_provenance": PROVENANCE})
        )
        monkeypatch.setenv(ENV_VAR, config["env"][ENV_VAR])

        name, data = jira_attachment.read_allowed_file(str(staged))
        assert (name, data) == ("report.txt", b"hello")
    finally:
        workspace.cleanup()
