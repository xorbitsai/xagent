from __future__ import annotations

from xagent.core.agent import Agent
from xagent.core.agent.runner import AgentRunner


def test_agent_exposes_runner_and_core_configuration() -> None:
    pattern = object()
    tool = object()

    agent = Agent(
        name="writer",
        patterns=[pattern],
        tools=[tool],
        llm="fake-llm",
        system_prompt="You are helpful.",
    )

    runner = agent.get_runner()

    assert isinstance(runner, AgentRunner)
    assert runner.agent is agent
    assert agent.patterns == [pattern]
    assert agent.tools == [tool]
    assert agent.llm == "fake-llm"
    assert agent.system_prompt == "You are helpful."
