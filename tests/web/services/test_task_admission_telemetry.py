import pytest
from sqlalchemy import Column, Integer, String, event
from sqlalchemy.orm import declarative_base

from tests.web.services.test_task_execution_admission import engine as engine_fixture
from tests.web.services.test_task_execution_admission import (
    enqueue,
)
from tests.web.services.test_task_execution_admission import host as host_fixture
from xagent.web.services import task_admission_observation as observation
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_execution_admission as admission

engine = engine_fixture
host = host_fixture

_TelemetryBase = declarative_base()


class _MissingTelemetryTicket(_TelemetryBase):
    __tablename__ = "missing_admission_telemetry_ticket"

    command_id = Column(Integer, primary_key=True)
    bucket_key = Column(String(255), nullable=False)
    owner_attempt_id = Column(String(64), nullable=True)


@pytest.mark.parametrize(
    "governed",
    [
        pytest.param(True, marks=pytest.mark.postgresql, id="statement-failure"),
        pytest.param(False, id="ungoverned"),
    ],
)
async def test_optional_admission_telemetry_cannot_interrupt_dispatch(
    host, monkeypatch, governed
):
    if governed and host.engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL transaction-abort semantics are required")
    if not governed:
        admission.set_task_admission_hook(None)
    accepted = enqueue(host)
    statements = []

    def record_statement(_connection, _cursor, statement, *_args):
        statements.append(statement)

    event.listen(host.engine, "before_cursor_execute", record_statement)
    try:
        monkeypatch.setattr(observation, "TaskAdmissionTicket", _MissingTelemetryTicket)
        executed = []

        async def execute(command):
            executed.append(command.id)
            return {}

        assert await transport.dispatch_one_task_command(
            execute, command_db_id=accepted.command_id
        )
    finally:
        event.remove(host.engine, "before_cursor_execute", record_statement)

    assert executed == [accepted.command_id]
    telemetry_queried = any(
        "missing_admission_telemetry_ticket" in statement for statement in statements
    )
    assert telemetry_queried is governed
