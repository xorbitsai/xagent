"""Export committed V2 execution events from real E2E runs as payload templates.

Not a regression test: it is skipped unless ``XAGENT_EXPORT_EVENT_TEMPLATES``
names the JSON file to write. The read-budget measurements build synthetic
histories from that file instead of hand-written payload shapes::

    XAGENT_EXPORT_EVENT_TEMPLATES=tests/web/services/fixtures/execution_event_templates.json \\
        python -m pytest tests/e2e/test_execution_event_template_export.py --run-special

For every workflow the file holds the committed kind sequence, with identity
ordinals, the setup event watermark each recovery state carried, and payload
sizes. For the template workflows it also holds the first sanitized payload of
each kind (split by recovery-state label and by result shape). Temporary
paths, credentials and generated identities are replaced by placeholders of the
same length, so template sizes match the real payloads. The
model boundary is the deterministic E2E double, so payload text is test data.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e import test_shared_execution as workflows

OUTPUT = os.getenv("XAGENT_EXPORT_EVENT_TEMPLATES")
# Optional directory for the unsanitized rows, for local inspection only.
RAW_OUTPUT = os.getenv("XAGENT_EXPORT_EVENT_TEMPLATES_RAW")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.parametrize("shared_app", [2], indirect=True),
    pytest.mark.skipif(not OUTPUT, reason="XAGENT_EXPORT_EVENT_TEMPLATES is not set"),
]

WORKFLOWS = {
    # Two runs on one task: the second run's setup carries an event watermark.
    "sdk_two_runs": workflows.test_sdk_create_append_and_poll_execute_only_in_worker,
    # ReAct with read/write tools and a generated file output.
    "react_file_tools": workflows.test_sdk_file_input_tool_output_and_download,
    # WebSocket execution: final-answer streaming and a first-run watermark.
    "websocket_stream": workflows.test_websocket_execute_stream_and_reconnect,
    # Workforce manager delegating to a child agent.
    "workforce_child": lambda app: (
        workflows.test_workforce_manager_and_child_execute_in_worker(app, "owner")
    ),
}
# Payload templates are kept for these workflows; readers pick the first
# workflow, in this order, that has a template key.
TEMPLATE_WORKFLOWS = ("react_file_tools", "sdk_two_runs")

_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_HEX = re.compile(r"\b[0-9a-f]{32,64}\b", re.IGNORECASE)
_TEMP_PATH = re.compile(r"/(?:private/)?(?:var|tmp)/[^\s\"'\])]+")
_SECRET_KEY = re.compile(r"token|secret|password|api_key|authorization", re.I)


def _same_length(prefix: str, fill: str) -> Any:
    """Replace a match by a placeholder of equal length: sizes stay exact."""
    return lambda match: (prefix + fill * len(match.group(0)))[: len(match.group(0))]


def _sanitize(value: Any, root: str, key: str = "") -> Any:
    """Keep the payload structure and sizes; drop paths, secrets, identities."""
    if isinstance(value, dict):
        return {k: _sanitize(v, root, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v, root, key) for v in value]
    if isinstance(value, str):
        if value and _SECRET_KEY.search(key):
            return "x" * len(value)
        value = value.replace(root, ("/srv/xagent-e2e" + "/r" * len(root))[: len(root)])
        value = _TEMP_PATH.sub(_same_length("/srv/xagent-e2e/", "p"), value)
        value = _UUID.sub(_same_length("00000000-0000-4000-8000-", "0"), value)
        return _HEX.sub(_same_length("", "0"), value)
    return value


def _data(payload: Any) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, dict) else {}


def template_key(kind: str, payload: Any) -> str:
    """The kind, split where one kind has distinct payload shapes."""
    data = _data(payload)
    if kind == "recovery_state":
        return f"recovery_state:{data.get('label')}"
    if kind == "llm_call_end":
        response = data.get("response")
        calls = isinstance(response, dict) and response.get("tool_calls")
        return "llm_call_end:tool_call" if calls else "llm_call_end:answer"
    if kind == "tool_execution_end":
        result = data.get("result")
        is_file = isinstance(result, dict) and result.get("file_id")
        return "tool_execution_end:file" if is_file else "tool_execution_end:text"
    return kind


def _setup_watermark(kind: str, payload: Any) -> int | None:
    if kind != "recovery_state":
        return None
    context = (_data(payload).get("snapshot") or {}).get("context") or {}
    watermark = (context.get("metadata") or {}).get("model_context_watermark")
    return watermark.get("sequence") if isinstance(watermark, dict) else None


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_export_execution_event_templates(shared_app, workflow):
    # Each workflow gets its own application: they reuse fixed agent names.
    WORKFLOWS[workflow](shared_app)
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_execution_event import TaskExecutionEvent

    identities = ("run_id", "turn_id", "assistant_message_id", "tool_attempt_id")
    with get_session_local()() as db:
        raw = [
            {
                "task_id": row.task_id,
                "scope_id": row.scope_id,
                "sequence": row.sequence,
                "kind": row.kind,
                "payload_version": row.payload_version,
                **{column: getattr(row, column) for column in identities},
                "payload": row.payload,
            }
            for row in db.query(TaskExecutionEvent).order_by(
                TaskExecutionEvent.task_id,
                TaskExecutionEvent.scope_id,
                TaskExecutionEvent.sequence,
            )
        ]
    assert raw
    if RAW_OUTPUT:
        raw_dir = Path(RAW_OUTPUT)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.joinpath(f"{workflow}.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
    # Identity columns become ordinals: sequences keep which rows share a
    # run, turn, batch or attempt, not the generated values themselves.
    ordinals: dict[tuple[str, Any], int] = {}

    def ordinal(column: str, value: Any) -> int | None:
        if value is None:
            return None
        return ordinals.setdefault((column, value), len(ordinals) + 1)

    root = str(shared_app.root)
    sequence = []
    templates: dict[str, Any] = {}
    for event in raw:
        key = template_key(event["kind"], event["payload"])
        sequence.append(
            {
                "task": ordinal("task_id", event["task_id"]),
                "scope": "root"
                if event["scope_id"] == "root"
                else f"child-{ordinal('scope_id', event['scope_id'])}",
                "sequence": event["sequence"],
                "kind": event["kind"],
                "template": key,
                "payload_version": event["payload_version"],
                **{column: ordinal(column, event[column]) for column in identities},
                "setup_watermark_sequence": _setup_watermark(
                    event["kind"], event["payload"]
                ),
                "payload_bytes": len(
                    json.dumps(event["payload"], ensure_ascii=False).encode()
                ),
            }
        )
        if workflow in TEMPLATE_WORKFLOWS:
            templates.setdefault(key, _sanitize(event["payload"], root))

    output = Path(cast(str, OUTPUT))
    exported = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    exported["source"] = "tests/e2e/test_execution_event_template_export.py"
    exported["template_workflows"] = list(TEMPLATE_WORKFLOWS)
    exported.setdefault("sequences", {})[workflow] = sequence
    if templates:
        exported.setdefault("templates", {})[workflow] = templates
    output.write_text(
        json.dumps(exported, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
