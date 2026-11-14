from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, ReActPattern
from xagent.core.agent.tool_access import (
    TOOL_ACCESS_RULES,
    connector_tool_sources,
    tool_access_fact,
)
from xagent.core.tools.adapters.vibe.api_tool import APITool
from xagent.core.tools.adapters.vibe.api_tool_adapter import CustomApiTool
from xagent.core.tools.adapters.vibe.mcp_adapter import UnavailableMCPTool
from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)


def test_sources_use_metadata_not_names_or_configuration() -> None:
    connector = CustomApiTool(
        name="Inventory",
        description="Stock",
        env={"PRIVATE_KEY": "secret"},
        url="https://private.invalid/stock",
        headers={"Authorization": "secret"},
    )
    wrapped = OutputFilteredToolWrapper(
        connector, max_chars=1000, max_fields=10, max_recursion=3
    )
    sources = connector_tool_sources(
        [
            APITool(),
            wrapped,
            SimpleNamespace(name="mcp_unrelated", metadata=None),
            SimpleNamespace(
                name="legacy",
                metadata=SimpleNamespace(
                    name="actual_schema_name", source_server="selected_server"
                ),
            ),
        ]
    )
    assert sources == {
        "api_Inventory_call": "inventory",
        "actual_schema_name": "selected_server",
    }
    text = tool_access_fact(sources, ["api_Inventory_call", "api_call"])
    assert '"inventory": ["api_Inventory_call"]' in text
    assert "selected_server" not in text
    assert "private.invalid" not in text
    assert "PRIVATE_KEY" not in text
    assert "secret" not in text


def test_empty_inventory_is_a_fact_and_the_rules_are_static() -> None:
    text = tool_access_fact({}, ["api_call", "ask_user_question"])
    assert "{}" in text
    # The rules never depend on the run, so they can sit in the system message.
    assert "not all connections in the user's account" in TOOL_ACCESS_RULES
    assert "no preconfigured endpoint" in TOOL_ACCESS_RULES
    assert "use it to request the missing connection" in TOOL_ACCESS_RULES
    assert "uploaded/pasted data" in TOOL_ACCESS_RULES
    assert "documentation or tool results" in TOOL_ACCESS_RULES
    assert "otherwise report the missing source" in TOOL_ACCESS_RULES


def test_unavailable_connector_is_not_described_as_authorized() -> None:
    tool = UnavailableMCPTool(server_name="CRM", server_id=1)
    text = tool_access_fact(connector_tool_sources([tool]), [tool.name])
    assert '"crm"' in text
    assert "does not prove that authorization or a request will succeed" in (
        TOOL_ACCESS_RULES
    )


@pytest.mark.parametrize("names", [[], ["calculator"], ["final_answer"]])
def test_unrelated_or_narrowed_calls_do_not_receive_connector_guidance(
    names: list[str],
) -> None:
    assert tool_access_fact({"hidden_tool": "private_connection"}, names) == ""


def _prompt_text(call: dict[str, Any]) -> str:
    # The inventory is a per-call fact in the turn context, not in the system.
    return "\n".join(str(m.get("content")) for m in call["messages"])


class RecordingLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("restore_checkpoint", [False, True])
@pytest.mark.parametrize("remove_connector", [False, True])
async def test_react_refreshes_inventory_after_waiting(
    restore_checkpoint: bool,
    remove_connector: bool,
) -> None:
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Use the connection after I configure it.")
    llm = RecordingLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "question",
                        "function": {
                            "name": "ask_user_question",
                            "arguments": '{"message":"Configure the connection, then continue."}',
                        },
                    }
                ]
            }
        ]
    )
    connector = CustomApiTool(name="CRM", description="Customers", env={})
    before = [connector] if remove_connector else [APITool()]
    after = [APITool()] if remove_connector else [connector]
    first = await pattern.run(context=context, tools=before, llm=llm)
    assert first["status"] == "waiting_for_user"
    old_inventory = (
        '"crm": ["api_CRM_call"]'
        if remove_connector
        else "inventory (source -> exposed tool names): {}"
    )
    assert old_inventory in _prompt_text(llm.calls[0])
    if restore_checkpoint:
        resumed = ReActPattern(max_iterations=3)
        resumed.load_state(pattern.get_state())
        pattern = resumed
    context.add_user_message(
        "Continue.", metadata={"response_to_waiting_for_user": "legacy"}
    )
    final_llm = RecordingLLM([{"content": "Connection tools are now exposed."}])
    result = await pattern.run(context=context, tools=after, llm=final_llm)
    assert result["success"] is True
    prompt = _prompt_text(final_llm.calls[0])
    assert old_inventory not in prompt
    if remove_connector:
        assert "inventory (source -> exposed tool names): {}" in prompt
    else:
        assert '"crm": ["api_CRM_call"]' in prompt


def test_forced_answer_does_not_advertise_connector_tools() -> None:
    pattern = ReActPattern()
    pattern._connector_tool_sources = {"crm_read": "crm"}
    context = ExecutionContext()
    context.add_user_message("Summarize.")
    prompt = pattern._messages_for_llm(
        context, has_tools=True, force_final_answer=True, tool_names=["final_answer"]
    )[0]["content"]
    assert "connector-tool inventory" not in prompt
