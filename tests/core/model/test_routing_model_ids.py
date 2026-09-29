"""Routing model ids: ids that select among or combine models instead of naming one.

Hosts that must validate one fixed model target (a capability probe that
has to hold for later execution) use ``is_routing_model`` to refuse both the
in-process virtual router and a provider's documented server-side
multi-model entry points (routers and Fusion), without refusing ordinary
model ids or stable single-model aliases.
"""

import pytest

from xagent.core.model.providers import is_auto_router_model, is_routing_model


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        # The in-process virtual router is a routing model too.
        ("openrouter", "auto"),
        ("openrouter", " AUTO "),
        ("router", "auto"),
        # OpenRouter's documented server-side routers.
        ("openrouter", "openrouter/auto"),
        ("openrouter", "OpenRouter/Auto"),
        ("openrouter", " openrouter/auto "),
        ("openrouter", "openrouter/auto-beta"),
        ("openrouter", "openrouter/free"),
        ("openrouter", "openrouter/pareto-code"),
        # Fusion: a panel of models plus an analyst whenever it deliberates.
        ("openrouter", "openrouter/fusion"),
        ("openrouter", "openrouter/fusion-flash"),
        ("openrouter", "OpenRouter/Fusion"),
        ("openrouter", " openrouter/FUSION-flash "),
        # A variant suffix still names the same entry point.
        ("openrouter", "openrouter/auto:online"),
        ("openrouter", "openrouter/fusion:free"),
        ("openrouter", "openrouter/fusion-flash:free"),
        ("openrouter", "openrouter/fusion:nitro"),
        ("OpenRouter", "openrouter/auto"),
        ("OpenRouter", "openrouter/fusion"),
    ],
)
def test_routers_are_recognized(provider, model_name):
    assert is_routing_model(provider, model_name) is True


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        # Ordinary ids and stable aliases each name one model.
        ("openrouter", "openai/gpt-4o"),
        ("openrouter", "anthropic/claude-sonnet-4.5"),
        ("openrouter", "openai/gpt-4o:nitro"),
        # Concrete models published under OpenRouter's own namespace.
        ("openrouter", "openrouter/optimus-alpha"),
        ("openrouter", "openrouter/automatic-model"),
        # Exact ids only: look-alikes are not Fusion.
        ("openrouter", "openrouter/fusion-pro"),
        ("openrouter", "openrouter/fusionx"),
        ("openrouter", "someone/fusion"),
        # The router ids are OpenRouter's; other providers do not inherit them.
        ("openai", "openrouter/auto"),
        ("openai", "openrouter/fusion"),
        ("deepseek", "deepseek-v4-flash"),
        ("openai", "auto"),
        ("openrouter", ""),
        ("openrouter", None),
    ],
)
def test_single_models_are_not_routers(provider, model_name):
    assert is_routing_model(provider, model_name) is False


def test_the_virtual_router_predicate_is_unchanged():
    # Construction still dispatches only the local "auto" to RouterLLM; the
    # server-side router ids are ordinary OpenRouter requests to XAgent.
    assert is_auto_router_model("openrouter", "auto") is True
    assert is_auto_router_model("openrouter", "openrouter/auto") is False
