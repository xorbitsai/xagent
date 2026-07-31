from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.task_runtime import (
    TaskRuntimeContribution,
    merge_task_runtime_contributions,
)
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec


def _tool(name: str) -> Any:
    return SimpleNamespace(
        name=name,
        metadata=SimpleNamespace(category=ToolCategory.OTHER),
    )


def _categorized_tool(name: str, category: ToolCategory) -> Any:
    return SimpleNamespace(
        name=name,
        metadata=SimpleNamespace(category=category),
    )


@pytest.mark.asyncio
async def test_runtime_tools_enter_normal_selection_pipeline(monkeypatch) -> None:
    base_tool = _tool("base_tool")
    runtime_tool = _tool("runtime_tool")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    config = ToolConfig({"allowed_tools": ["runtime_tool"]})

    tools = await ToolFactory.create_all_tools(
        config,
        additional_tools=(runtime_tool,),
    )

    assert tools == [runtime_tool]


@pytest.mark.asyncio
async def test_runtime_tools_filtered_by_policy_log_provider(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    base_tool = _tool("base_tool")
    runtime_tool = _tool("runtime_tool")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    config = ToolConfig({"allowed_tools": ["base_tool"]})
    config.get_task_runtime_contribution = lambda: TaskRuntimeContribution(
        tools=(runtime_tool,),
        tool_origins=(("runtime_tool", "browser_runtime"),),
    )

    with caplog.at_level("WARNING"):
        tools = await ToolFactory.create_all_tools(config)

    assert tools == [base_tool]
    assert "browser_runtime=[runtime_tool]" in caplog.text


@pytest.mark.asyncio
async def test_runtime_tools_are_restored_from_config_on_rebuild(monkeypatch) -> None:
    base_tool = _tool("base_tool")
    runtime_tool = _tool("runtime_tool")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    config = ToolConfig({})
    config.get_task_runtime_contribution = lambda: TaskRuntimeContribution(
        tools=(runtime_tool,)
    )

    first = await ToolFactory.create_all_tools(config)
    rebuilt = await ToolFactory.create_all_tools(config)

    assert [tool.name for tool in first] == ["base_tool", "runtime_tool"]
    assert [tool.name for tool in rebuilt] == ["base_tool", "runtime_tool"]


@pytest.mark.asyncio
async def test_runtime_tools_cannot_shadow_existing_tool(
    monkeypatch,
) -> None:
    async def create_registered_tools(config: Any) -> list[Any]:
        return [_tool("computer")]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )

    with pytest.raises(
        ValueError,
        match="desktop_runtime.*duplicate tool 'computer'",
    ):
        await ToolFactory.create_all_tools(
            ToolConfig({}),
            additional_tools=(_tool("computer"),),
            additional_tool_origins={"computer": "desktop_runtime"},
        )


@pytest.mark.asyncio
async def test_policy_filtered_runtime_collision_does_not_fail(monkeypatch) -> None:
    async def create_registered_tools(config: Any) -> list[Any]:
        return [_tool("computer")]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )

    tools = await ToolFactory.create_all_tools(
        ToolConfig({"allowed_tools": []}),
        additional_tools=(_tool("computer"),),
        additional_tool_origins={"computer": "desktop_runtime"},
    )

    assert tools == []


@pytest.mark.asyncio
async def test_structured_runtime_collision_drops_only_provider(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    core_tool = _tool("computer")
    runtime_tool = _tool("computer")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [core_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    contribution_holder = {
        "value": merge_task_runtime_contributions(
            {
                "desktop_runtime": TaskRuntimeContribution(
                    tools=(runtime_tool,),
                    environment="Control the desktop.",
                    preferred_input_modalities=("image",),
                )
            }
        )
    }
    config = ToolConfig({})
    config.get_task_runtime_contribution = lambda: contribution_holder["value"]
    config.set_task_runtime_contribution = lambda value: contribution_holder.update(
        value=value
    )

    with caplog.at_level("WARNING"):
        tools = await ToolFactory.create_all_tools(config)

    assert tools == [core_tool]
    assert contribution_holder["value"] == TaskRuntimeContribution()
    assert "Dropping task runtime extension 'desktop_runtime'" in caplog.text
    assert "computer" in caplog.text


@pytest.mark.asyncio
async def test_structured_runtime_collision_counts_shared_tool_occurrences(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    base_tool = _tool("base_tool")
    shared_tool = _tool("shared_tool")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    contribution_holder = {
        "value": merge_task_runtime_contributions(
            {
                "first_runtime": TaskRuntimeContribution(
                    tools=(shared_tool,),
                    environment="Use the first runtime.",
                ),
                "second_runtime": TaskRuntimeContribution(
                    tools=(shared_tool,),
                    environment="Use the second runtime.",
                ),
            }
        )
    }
    config = ToolConfig({})
    config.get_task_runtime_contribution = lambda: contribution_holder["value"]
    config.set_task_runtime_contribution = lambda value: contribution_holder.update(
        value=value
    )

    with caplog.at_level("WARNING"):
        tools = await ToolFactory.create_all_tools(config)

    assert tools == [base_tool, shared_tool]
    assert contribution_holder["value"].environment == "Use the first runtime."
    assert tuple(
        name
        for name, _contribution in contribution_holder["value"].provider_contributions
    ) == ("first_runtime",)
    assert "Dropping task runtime extension 'second_runtime'" in caplog.text


@pytest.mark.asyncio
async def test_explicit_runtime_tools_preserve_provider_origin(
    monkeypatch, caplog
) -> None:
    async def create_registered_tools(config: Any) -> list[Any]:
        return []

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    config = ToolConfig({"allowed_tools": []})

    with caplog.at_level("WARNING"):
        tools = await ToolFactory.create_all_tools(
            config,
            additional_tools=(_tool("runtime_tool"),),
            additional_tool_origins={"runtime_tool": "browser_runtime"},
        )

    assert tools == []
    assert "browser_runtime=[runtime_tool]" in caplog.text


@pytest.mark.asyncio
async def test_filtered_runtime_tools_remove_provider_environment(monkeypatch) -> None:
    base_tool = _tool("base_tool")
    runtime_tool = _tool("runtime_tool")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    contribution_holder = {
        "value": merge_task_runtime_contributions(
            {
                "browser_runtime": TaskRuntimeContribution(
                    tools=(runtime_tool,),
                    environment="Use the leased browser.",
                    preferred_input_modalities=("image",),
                )
            }
        )
    }
    config = ToolConfig({"allowed_tools": ["base_tool"]})
    config.get_task_runtime_contribution = lambda: contribution_holder["value"]
    config.set_task_runtime_contribution = lambda value: contribution_holder.update(
        value=value
    )

    tools = await ToolFactory.create_all_tools(config)

    assert tools == [base_tool]
    assert contribution_holder["value"].environment is None
    assert contribution_holder["value"].preferred_input_modalities == ()


@pytest.mark.asyncio
async def test_runtime_tools_require_a_non_empty_string_name(monkeypatch) -> None:
    async def create_registered_tools(config: Any) -> list[Any]:
        return []

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )

    with pytest.raises(TypeError, match="non-empty string 'name'"):
        await ToolFactory.create_all_tools(
            ToolConfig({}),
            additional_tools=(SimpleNamespace(),),
        )


@pytest.mark.asyncio
async def test_runtime_tool_without_metadata_category_is_dropped_not_fatal(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider tool lacking ``metadata.category`` must not kill the build.

    A plain LangChain ``@tool`` function has ``metadata = None``, so the
    category sort key would raise ``AttributeError`` and take down the whole
    task's tool initialization together with every core tool.
    """
    base_tool = _tool("base_tool")
    good_runtime_tool = _tool("good_runtime_tool")
    no_metadata_tool = SimpleNamespace(name="no_metadata_tool")
    none_metadata_tool = SimpleNamespace(name="none_metadata_tool", metadata=None)

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    config = ToolConfig({})

    with caplog.at_level("WARNING"):
        tools = await ToolFactory.create_all_tools(
            config,
            additional_tools=(
                good_runtime_tool,
                no_metadata_tool,
                none_metadata_tool,
            ),
            additional_tool_origins={
                "good_runtime_tool": "browser_runtime",
                "no_metadata_tool": "browser_runtime",
                "none_metadata_tool": "desktop_runtime",
            },
        )

    names = [tool.name for tool in tools]
    assert names == ["base_tool", "good_runtime_tool"]
    assert "no_metadata_tool" not in names
    assert "none_metadata_tool" not in names
    assert "no_metadata_tool" in caplog.text
    assert "none_metadata_tool" in caplog.text


@pytest.mark.asyncio
async def test_structured_collision_still_drops_only_provider_with_malformed_peer(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed peer contribution must not turn a collision into a crash.

    ``provider_a`` contributes one tool without ``metadata.category`` (dropped)
    plus one well-formed tool; ``provider_b`` independently collides with a
    core tool name. Dropping the malformed tool shrinks the extension list, so
    the reconciliation identity guard must still recognise the structured
    contribution instead of falling through to the fatal duplicate-name check.
    """
    core_tool = _tool("computer")
    malformed_tool = SimpleNamespace(name="malformed_tool", metadata=None)
    good_tool = _tool("a_good_tool")
    colliding_tool = _tool("computer")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [core_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    contribution_holder = {
        "value": merge_task_runtime_contributions(
            {
                "provider_a": TaskRuntimeContribution(
                    tools=(malformed_tool, good_tool),
                    environment="Use provider a.",
                ),
                "provider_b": TaskRuntimeContribution(
                    tools=(colliding_tool,),
                    environment="Use provider b.",
                ),
            }
        )
    }
    config = ToolConfig({})
    config.get_task_runtime_contribution = lambda: contribution_holder["value"]
    config.set_task_runtime_contribution = lambda value: contribution_holder.update(
        value=value
    )

    with caplog.at_level("WARNING"):
        tools = await ToolFactory.create_all_tools(config)

    assert tools == [core_tool, good_tool]
    assert "malformed_tool" in caplog.text
    assert "Dropping task runtime extension 'provider_b'" in caplog.text
    assert tuple(
        name
        for name, _contribution in contribution_holder["value"].provider_contributions
    ) == ("provider_a",)
    assert contribution_holder["value"].environment == "Use provider a."


@pytest.mark.asyncio
async def test_all_tools_malformed_drops_provider_prompt_context(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider whose every tool is malformed must lose its prompt context.

    When the malformed-tool filter removes the provider's only contribution,
    the stored contribution must be reconciled too. Otherwise the agent keeps
    the provider's ``environment`` text and ``preferred_input_modalities``
    routing preference for a capability that has no tool behind it.
    """
    base_tool = _tool("base_tool")
    malformed_tool = SimpleNamespace(name="malformed_tool", metadata=None)

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    contribution_holder = {
        "value": merge_task_runtime_contributions(
            {
                "browser_runtime": TaskRuntimeContribution(
                    tools=(malformed_tool,),
                    environment="Use the leased browser.",
                    preferred_input_modalities=("image",),
                )
            }
        )
    }
    config = ToolConfig({})
    config.get_task_runtime_contribution = lambda: contribution_holder["value"]
    config.set_task_runtime_contribution = lambda value: contribution_holder.update(
        value=value
    )

    with caplog.at_level("WARNING"):
        tools = await ToolFactory.create_all_tools(config)

    assert tools == [base_tool]
    assert "malformed_tool" in caplog.text
    assert contribution_holder["value"].environment is None
    assert contribution_holder["value"].preferred_input_modalities == ()
    assert contribution_holder["value"].provider_contributions == ()


@pytest.mark.asyncio
async def test_malformed_tool_name_does_not_evict_a_valid_peer_provider(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rejected tool name must not claim the name for its provider.

    ``provider_a`` contributes a malformed tool named ``shared_tool`` that the
    factory drops before reconciliation. ``provider_b`` contributes a *valid*
    tool with the same name. Name-only reconciliation would let ``provider_a``
    claim ``shared_tool`` with an object that never survived, flag
    ``provider_b`` as the colliding provider, and drop the only real tool.
    """
    base_tool = _tool("base_tool")
    malformed_tool = SimpleNamespace(name="shared_tool", metadata=None)
    valid_tool = _tool("shared_tool")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    contribution_holder = {
        "value": merge_task_runtime_contributions(
            {
                "provider_a": TaskRuntimeContribution(
                    tools=(malformed_tool,),
                    environment="Use provider a.",
                ),
                "provider_b": TaskRuntimeContribution(
                    tools=(valid_tool,),
                    environment="Use provider b.",
                ),
            }
        )
    }
    config = ToolConfig({})
    config.get_task_runtime_contribution = lambda: contribution_holder["value"]
    config.set_task_runtime_contribution = lambda value: contribution_holder.update(
        value=value
    )

    with caplog.at_level("WARNING"):
        tools = await ToolFactory.create_all_tools(config)

    # provider_b's valid tool must reach execution.
    assert tools == [base_tool, valid_tool]
    # The malformed-tool drop must be attributed to the provider that sent it.
    assert "'shared_tool' from 'provider_a'" in caplog.text
    assert "from 'provider_b'" not in caplog.text
    # provider_b survives reconciliation; provider_a is gone.
    assert tuple(
        name
        for name, _contribution in contribution_holder["value"].provider_contributions
    ) == ("provider_b",)
    assert contribution_holder["value"].environment == "Use provider b."


@pytest.mark.asyncio
async def test_runtime_tools_survive_category_scoped_agent_config(monkeypatch) -> None:
    """Category-scoped agents must still get task-runtime extension tools.

    An agent configured through the normal agent-builder flow carries a
    non-empty ``tool_categories``, i.e. a ``_SpecByCategories`` spec. Every
    task-runtime-contributed tool keeps ``ToolMetadata``'s default
    ``ToolCategory.OTHER``, and ``"other"`` is unconditionally stripped from a
    configured category set (``AGENT_CONFIG_UNASSIGNABLE_CATEGORIES``), so a
    pure default-deny category filter would silently drop every contributed
    tool for the majority of production agents.

    Task-runtime tools are requested explicitly at task-creation time and
    validated against the extension registry, which is a stronger, task-scoped
    opt-in than the agent-level category policy — so the spec must admit them
    regardless of category. The category filter itself must stay intact for
    everything else: a core tool whose category is outside the configured set
    is still excluded.
    """
    in_scope_core_tool = _categorized_tool("browser_use", ToolCategory.BROWSER)
    out_of_scope_core_tool = _categorized_tool("read_file", ToolCategory.FILE)
    runtime_tool = _tool("runtime_tool")  # default ToolCategory.OTHER

    async def create_registered_tools(config: Any) -> list[Any]:
        return [in_scope_core_tool, out_of_scope_core_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    config = ToolConfig({})
    config._tool_selection_spec = ToolSelectionSpec.from_raw(
        tool_categories=[ToolCategory.BROWSER.value],
    )
    config.get_task_runtime_contribution = lambda: TaskRuntimeContribution(
        tools=(runtime_tool,),
        tool_origins=(("runtime_tool", "browser_runtime"),),
    )

    tools = await ToolFactory.create_all_tools(config)

    names = {tool.name for tool in tools}
    # The headline capability: the contributed tool survives the category spec.
    assert "runtime_tool" in names
    # The configured category still admits its own tools ...
    assert "browser_use" in names
    # ... and the category filter is NOT weakened for non-contributed tools.
    assert "read_file" not in names


def test_selection_spec_extension_bypass_is_scoped_to_contributed_names() -> None:
    """The bypass admits only the names the caller declares as contributed."""
    contributed = _tool("runtime_tool")
    other_out_of_category = _categorized_tool("read_file", ToolCategory.FILE)
    in_category = _categorized_tool("browser_use", ToolCategory.BROWSER)

    spec = ToolSelectionSpec.from_raw(tool_categories=[ToolCategory.BROWSER.value])

    assert spec.compute_allowed_names(
        [contributed, other_out_of_category, in_category],
    ) == frozenset({"browser_use"})
    assert spec.compute_allowed_names(
        [contributed, other_out_of_category, in_category],
        extension_tool_names=frozenset({"runtime_tool"}),
    ) == frozenset({"browser_use", "runtime_tool"})


@pytest.mark.asyncio
async def test_extension_bypass_does_not_leak_out_of_category_core_name(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A contributed name colliding with a core tool must not widen the policy.

    The final selection filter is name-based, so admitting a contributed name
    into ``allowed_names`` admits *every* tool carrying that name — including a
    core tool whose real category the agent's ``tool_categories`` excludes.
    The colliding contribution is doomed anyway (reconciliation drops the
    offending provider), so it must never be granted the category bypass.
    """
    in_scope_core_tool = _categorized_tool("browser_use", ToolCategory.BROWSER)
    out_of_scope_core_tool = _categorized_tool("read_file", ToolCategory.FILE)
    colliding_runtime_tool = _tool("read_file")  # default ToolCategory.OTHER

    async def create_registered_tools(config: Any) -> list[Any]:
        return [in_scope_core_tool, out_of_scope_core_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    contribution_holder = {
        "value": merge_task_runtime_contributions(
            {
                "file_runtime": TaskRuntimeContribution(
                    tools=(colliding_runtime_tool,),
                    environment="Use the file runtime.",
                ),
            }
        )
    }
    config = ToolConfig({})
    config._tool_selection_spec = ToolSelectionSpec.from_raw(
        tool_categories=[ToolCategory.BROWSER.value],
    )
    config.get_task_runtime_contribution = lambda: contribution_holder["value"]
    config.set_task_runtime_contribution = lambda value: contribution_holder.update(
        value=value
    )

    with caplog.at_level("WARNING"):
        tools = await ToolFactory.create_all_tools(config)

    names = [tool.name for tool in tools]
    # The out-of-category core tool must stay excluded by the category policy.
    assert "read_file" not in names
    # The in-category core tool is unaffected.
    assert names == ["browser_use"]
    # And the colliding provider is still dropped from the contribution.
    assert contribution_holder["value"].provider_contributions == ()


@pytest.mark.asyncio
async def test_policy_narrowed_contribution_returns_when_policy_widens(
    monkeypatch,
) -> None:
    """A contribution filtered by a restrictive policy must come back later.

    Turn 1 narrows the stored contribution to nothing. Turn 2 relaxes the
    allowlist; the tool, its prompt environment, its modality preference and
    its ``provider_contributions`` entry must all be re-derived from the full
    contribution instead of staying permanently lost.
    """
    base_tool = _tool("base_tool")
    runtime_tool = _tool("runtime_tool")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    contribution_holder = {
        "value": merge_task_runtime_contributions(
            {
                "browser_runtime": TaskRuntimeContribution(
                    tools=(runtime_tool,),
                    environment="Use the leased browser.",
                    preferred_input_modalities=("image",),
                )
            }
        )
    }
    config = ToolConfig({"allowed_tools": ["base_tool"]})
    config.get_task_runtime_contribution = lambda: contribution_holder["value"]
    config.set_task_runtime_contribution = lambda value: contribution_holder.update(
        value=value
    )

    narrowed = await ToolFactory.create_all_tools(config)

    assert [tool.name for tool in narrowed] == ["base_tool"]
    assert contribution_holder["value"].environment is None
    assert contribution_holder["value"].provider_contributions == ()

    # Turn 2: the restrictive allowlist is relaxed back to normal.
    config.allowed_tools = ["base_tool", "runtime_tool"]

    widened = await ToolFactory.create_all_tools(config)

    assert [tool.name for tool in widened] == ["base_tool", "runtime_tool"]
    restored = contribution_holder["value"]
    assert restored.environment == "Use the leased browser."
    assert restored.preferred_input_modalities == ("image",)
    assert tuple(name for name, _c in restored.provider_contributions) == (
        "browser_runtime",
    )
    assert restored.tools == (runtime_tool,)


@pytest.mark.asyncio
async def test_whitespace_padded_runtime_tool_is_not_reported_as_filtered(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool whose name carries padding must not look policy-filtered.

    Policy matching compares the raw ``tool.name``, so tracking the contributed
    names in a stripped form makes a surviving padded name look dropped.
    """
    base_tool = _tool("base_tool")
    runtime_tool = _tool(" foo ")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    contribution_holder = {
        "value": merge_task_runtime_contributions(
            {
                "browser_runtime": TaskRuntimeContribution(
                    tools=(runtime_tool,),
                    environment="Use the padded tool.",
                )
            }
        )
    }
    config = ToolConfig({"allowed_tools": ["base_tool", " foo "]})
    config.get_task_runtime_contribution = lambda: contribution_holder["value"]
    config.set_task_runtime_contribution = lambda value: contribution_holder.update(
        value=value
    )

    with caplog.at_level("WARNING"):
        tools = await ToolFactory.create_all_tools(config)

    assert tools == [base_tool, runtime_tool]
    assert "filtered by task tool policy" not in caplog.text
    assert contribution_holder["value"].environment == "Use the padded tool."


@pytest.mark.asyncio
async def test_whitespace_padded_duplicate_tool_reports_its_provider(
    monkeypatch,
) -> None:
    """Origin lookup must find the provider of a padded, colliding tool name.

    ``tool_origins`` is keyed by the stripped name, so a raw-name lookup would
    attribute the duplicate to "unknown".
    """
    core_tool = _tool(" foo ")
    runtime_tool = _tool(" foo ")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [core_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    config = ToolConfig({})

    with pytest.raises(ValueError, match="browser_runtime"):
        await ToolFactory.create_all_tools(
            config,
            additional_tools=(runtime_tool,),
            additional_tool_origins={"foo": "browser_runtime"},
        )
