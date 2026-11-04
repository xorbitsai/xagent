import asyncssh
import pytest

from tests.core.ssh.helpers import start_test_ssh_server

pytestmark = pytest.mark.integration


async def test_server_accepts_authorized_key_and_runs_command() -> None:
    server = await start_test_ssh_server()
    try:
        client_key = asyncssh.import_private_key(server.client_private_key)
        async with asyncssh.connect(
            server.host,
            port=server.port,
            client_keys=[client_key],
            known_hosts=None,  # strict verification is a Phase 3 concern
        ) as conn:
            result = await conn.run("uptime", check=True)
            assert result.stdout == "ran: uptime"
    finally:
        await server.close()


async def test_server_rejects_unknown_key() -> None:
    server = await start_test_ssh_server()
    try:
        stranger = asyncssh.generate_private_key("ssh-ed25519")
        with pytest.raises(asyncssh.PermissionDenied):
            await asyncssh.connect(
                server.host,
                port=server.port,
                client_keys=[stranger],
                known_hosts=None,
            )
    finally:
        await server.close()
