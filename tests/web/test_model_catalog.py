"""Unit tests for ``xagent.web.services.model_catalog``.

Covers:
- YAML loading from the bundled real catalog
- Three-tier resolution (exact provider+model, wildcard provider+model,
  provider-only fallback)
- Case-insensitive matching, whitespace tolerance, empty inputs
- Graceful degradation when the YAML is missing or malformed
- ``reload_catalog`` picks up edits without restarting the process
- Regression tests for known-good rules and a recent vision-routing bug fix
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List

import pytest

from xagent.web.services import model_catalog
from xagent.web.services.model_catalog import (
    AbilitySuggestion,
    lookup,
    reload_catalog,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_real_catalog():
    """Each test starts with a fresh load of the real bundled catalog so
    that one test's monkeypatching can't leak into the next."""
    reload_catalog()
    yield
    reload_catalog()


@pytest.fixture
def temp_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return a callable that writes a tiny YAML and points the catalog
    module at it. Returns the new rule count after reload."""

    def _write(yaml_body: str) -> int:
        catalog_path = tmp_path / "abilities_catalog.yaml"
        catalog_path.write_text(textwrap.dedent(yaml_body), encoding="utf-8")
        monkeypatch.setattr(model_catalog, "_CATALOG_PATH", catalog_path)
        return reload_catalog()

    return _write


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestCatalogLoading:
    """Verify the bundled catalog file loads and basic shape is sane."""

    def test_real_catalog_loads_some_rules(self):
        n = reload_catalog()
        assert n > 50, f"expected > 50 rules in bundled catalog, got {n}"

    def test_lookup_returns_AbilitySuggestion(self):
        result = lookup("openai", "gpt-4o")
        assert isinstance(result, AbilitySuggestion)
        assert isinstance(result.abilities, list)
        assert result.source in {"exact", "wildcard_provider", "none"}


# ---------------------------------------------------------------------------
# Three-tier resolution
# ---------------------------------------------------------------------------


class TestResolutionTiers:
    """The three-tier fallback is the heart of the lookup logic — exercise
    every transition explicitly."""

    def test_tier1_exact_provider_and_model(self):
        r = lookup("openai", "gpt-4o-2024-08-06")
        assert r.source == "exact"
        assert r.matched_pattern == "openai/gpt-4o*"
        assert "vision" in r.abilities
        assert "tool_calling" in r.abilities

    def test_tier2_wildcard_provider_for_cross_provider_model(self):
        # DeepSeek through OpenAI-compatible endpoint should NOT be
        # absorbed by ``openai/*`` provider fallback.
        r = lookup("openai", "deepseek-reasoner-v2")
        assert r.source == "wildcard_provider"
        assert r.matched_pattern == "*/deepseek-reasoner*"
        assert r.abilities == ["chat", "thinking_mode"]

    def test_tier3_provider_only_fallback(self):
        # Unknown OpenAI model name should fall to ``openai/*`` rule.
        r = lookup("openai", "some-completely-unknown-2050-model")
        assert r.source == "exact"
        assert r.matched_pattern == "openai/*"

    def test_no_match_returns_none(self):
        r = lookup("totally-unknown-provider", "some-random-model")
        assert r.source == "none"
        assert r.abilities == []
        assert r.matched_pattern is None

    def test_specific_rule_wins_over_provider_fallback(self):
        # ``openai/o1*`` should win over ``openai/*``
        r = lookup("openai", "o1-preview")
        assert r.matched_pattern == "openai/o1*"
        assert "thinking_mode" in r.abilities

    def test_wildcard_provider_beats_provider_only_fallback(self):
        # ``*/deepseek-chat*`` (specific model, wildcard provider) must win
        # against ``openai/*`` (provider fallback). This is THE design
        # invariant that lets DeepSeek/Qwen via openai-compat be detected
        # without forcing every alias into the openai block.
        r = lookup("openai", "deepseek-chat-1.5")
        assert r.matched_pattern == "*/deepseek-chat*"
        assert "tool_calling" in r.abilities


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    """Inputs from the wild: weird casing, whitespace, empty strings, etc."""

    def test_case_insensitive_provider(self):
        a = lookup("OpenAI", "gpt-4o")
        b = lookup("openai", "gpt-4o")
        assert a.abilities == b.abilities
        assert a.source == b.source

    def test_case_insensitive_model_name(self):
        a = lookup("openai", "GPT-4O-2024-08-06")
        b = lookup("openai", "gpt-4o-2024-08-06")
        assert a.abilities == b.abilities

    def test_strips_whitespace(self):
        a = lookup("  openai  ", "  gpt-4o  ")
        b = lookup("openai", "gpt-4o")
        assert a.abilities == b.abilities

    def test_empty_model_name_returns_none(self):
        r = lookup("openai", "")
        assert r.source == "none"
        assert r.abilities == []

    def test_empty_provider_with_known_model_uses_wildcard(self):
        # Empty provider + a model that has a */xxx* rule should still hit
        # tier 2 (wildcard provider).
        r = lookup("", "deepseek-chat")
        assert r.source == "wildcard_provider"

    def test_lookup_never_raises_on_weird_input(self):
        # Catalog should be defensive about garbage inputs. None of these
        # should raise.
        for provider, model in [
            (None, None),  # type: ignore[arg-type]
            (None, "gpt-4o"),  # type: ignore[arg-type]
            ("openai", None),  # type: ignore[arg-type]
            ("", ""),
            ("openai", "a" * 10_000),  # extremely long model name
        ]:
            r = lookup(provider, model)  # type: ignore[arg-type]
            assert isinstance(r, AbilitySuggestion)


# ---------------------------------------------------------------------------
# Catalog file degradation paths
# ---------------------------------------------------------------------------


class TestCatalogDegradation:
    """If the YAML is missing or malformed the API stays online — lookups
    return ``source='none'`` instead of crashing the request."""

    def test_missing_catalog_file(self, tmp_path, monkeypatch):
        bogus = tmp_path / "does_not_exist.yaml"
        monkeypatch.setattr(model_catalog, "_CATALOG_PATH", bogus)
        n = reload_catalog()
        assert n == 0
        assert lookup("openai", "gpt-4o").source == "none"

    def test_malformed_yaml_does_not_crash(self, temp_catalog):
        # Intentionally broken YAML; loader logs and returns no rules.
        n = temp_catalog("rules: [unclosed list with: {nested broken")
        assert n == 0
        assert lookup("openai", "gpt-4o").source == "none"

    def test_unknown_ability_value_filtered_out(self, temp_catalog):
        n = temp_catalog(
            """
            rules:
              - pattern: "openai/test*"
                abilities: [chat, vision, made_up_ability]
            """
        )
        assert n == 1
        r = lookup("openai", "test-model")
        # 'made_up_ability' must not leak through — only the whitelisted
        # ones survive.
        assert r.abilities == ["chat", "vision"]
        assert "made_up_ability" not in r.abilities

    def test_rule_with_only_unknown_abilities_dropped(self, temp_catalog):
        n = temp_catalog(
            """
            rules:
              - pattern: "openai/test*"
                abilities: [foo_ability, bar_ability]
            """
        )
        # Rule had no valid abilities at all -> dropped entirely.
        assert n == 0
        assert lookup("openai", "test-model").source == "none"


# ---------------------------------------------------------------------------
# Reload semantics
# ---------------------------------------------------------------------------


class TestReload:
    def test_reload_picks_up_edits(self, temp_catalog):
        n1 = temp_catalog(
            """
            rules:
              - pattern: "demo/x*"
                abilities: [chat]
            """
        )
        assert n1 == 1
        assert lookup("demo", "x1").abilities == ["chat"]

        # Update the same file with a different rule set; reload should
        # see it without a process restart.
        n2 = temp_catalog(
            """
            rules:
              - pattern: "demo/x*"
                abilities: [chat, vision]
              - pattern: "demo/y*"
                abilities: [chat, tool_calling]
            """
        )
        assert n2 == 2
        assert lookup("demo", "x1").abilities == ["chat", "vision"]
        assert lookup("demo", "y2").abilities == ["chat", "tool_calling"]


# ---------------------------------------------------------------------------
# Real-catalog regression checks
# ---------------------------------------------------------------------------


class TestRealCatalogRegression:
    """Concrete model -> abilities expectations the team has agreed on.
    Update these when the catalog rules change intentionally."""

    @pytest.mark.parametrize(
        "provider,model_name,expected_abilities",
        [
            # OpenAI core
            ("openai", "gpt-4o-2024-08-06", ["chat", "vision", "tool_calling"]),
            ("openai", "gpt-3.5-turbo", ["chat", "tool_calling"]),
            ("openai", "o1-preview", ["chat", "vision", "thinking_mode"]),
            ("openai", "o3-mini", ["chat", "tool_calling", "thinking_mode"]),
            # Anthropic
            (
                "claude",
                "claude-3-5-sonnet-20241022",
                ["chat", "vision", "tool_calling"],
            ),
            (
                "claude",
                "claude-sonnet-4-5",
                ["chat", "vision", "tool_calling", "thinking_mode"],
            ),
            ("claude", "claude-3-5-haiku", ["chat", "tool_calling"]),
            # Gemini
            (
                "gemini",
                "gemini-2.5-pro",
                ["chat", "vision", "tool_calling", "thinking_mode"],
            ),
            ("gemini", "gemini-1.5-flash-002", ["chat", "vision", "tool_calling"]),
            # Zhipu
            ("zhipu", "glm-4.5-air", ["chat", "tool_calling", "thinking_mode"]),
            ("zhipu", "glm-4.5v", ["chat", "vision", "tool_calling", "thinking_mode"]),
            ("zhipu", "glm-4v-plus", ["chat", "vision"]),
            # Cross-provider via OpenAI-compatible endpoints
            ("openai", "deepseek-chat", ["chat", "tool_calling"]),
            ("openai", "deepseek-reasoner", ["chat", "thinking_mode"]),
            (
                "openai",
                "kimi-k2.5",
                ["chat", "vision", "tool_calling", "thinking_mode"],
            ),
        ],
    )
    def test_known_model(
        self,
        provider: str,
        model_name: str,
        expected_abilities: List[str],
    ):
        r = lookup(provider, model_name)
        assert r.source != "none", f"{provider}/{model_name} expected to be in catalog"
        assert r.abilities == expected_abilities, (
            f"{provider}/{model_name} matched {r.matched_pattern}: "
            f"expected {expected_abilities}, got {r.abilities}"
        )


# ---------------------------------------------------------------------------
# Bug regressions
# ---------------------------------------------------------------------------


class TestBugRegressions:
    """Each test here corresponds to a bug we fixed. Don't delete them
    when refactoring — they pin specific catalog behavior."""

    @pytest.mark.parametrize(
        "provider,model_name",
        [
            ("alibaba-coding-plan", "qwen3.5-vl-plus"),
            ("alibaba-coding-plan", "qwen3-vl-max"),
            ("alibaba-coding-plan", "qwen2.5-vl-72b"),
            ("alibaba-coding-plan-cn", "qwen3.5-vl-plus"),
            ("alibaba-coding-plan-cn", "qwen-vl-max"),
        ],
    )
    def test_alibaba_coding_plan_vl_variants_have_vision(
        self, provider: str, model_name: str
    ):
        """Bug: ``alibaba-coding-plan/qwen3.5*`` was catching vision variants
        before a more specific ``qwen*-vl*`` rule was added. This regression
        test pins the fix."""
        r = lookup(provider, model_name)
        assert "vision" in r.abilities, (
            f"{provider}/{model_name} must include 'vision'; "
            f"matched={r.matched_pattern}, abilities={r.abilities}"
        )
        assert "tool_calling" in r.abilities

    @pytest.mark.parametrize(
        "provider,model_name",
        [
            ("zai-coding-plan", "glm-4v-plus"),
            ("zai-coding-plan", "glm-4v-flash"),
            ("zai-coding-plan", "glm-4.5v"),
            ("zhipuai-coding-plan", "glm-4v"),
            ("zhipuai-coding-plan", "glm-4.5v"),
            ("zhipu", "glm-4.5v"),
        ],
    )
    def test_glm_coding_plan_4v_variants_have_vision(
        self, provider: str, model_name: str
    ):
        """Bug: ``glm-4*`` rule in the GLM coding-plan blocks was eating
        ``glm-4v*`` (vision variant). Fixed by inserting an explicit
        ``glm-4v*`` rule above ``glm-4*``."""
        r = lookup(provider, model_name)
        assert "vision" in r.abilities, (
            f"{provider}/{model_name} must include 'vision'; "
            f"matched={r.matched_pattern}"
        )
        assert "thinking_mode" in r.abilities or "glm-4v" in model_name

    @pytest.mark.parametrize(
        "provider,model_name",
        [
            ("kimi-for-coding", "kimi-k2.5"),
            ("kimi-for-coding", "kimi-k2.6"),
            ("openai", "kimi-k2.5"),
            ("openai", "kimi-k2.6"),
        ],
    )
    def test_kimi_k2_5_and_2_6_variants_keep_vision_and_thinking(
        self, provider: str, model_name: str
    ):
        """Bug: generic ``kimi-k2*`` was catching newer multimodal K2.5/K2.6
        models before more specific rules could add vision + thinking."""
        r = lookup(provider, model_name)
        assert "vision" in r.abilities, (
            f"{provider}/{model_name} must include 'vision'; "
            f"matched={r.matched_pattern}, abilities={r.abilities}"
        )
        assert "thinking_mode" in r.abilities

    @pytest.mark.parametrize(
        "provider,model_name",
        [
            # Pure text models that were regression-checked after adding the
            # -vl rules — these must NOT pick up vision.
            ("alibaba-coding-plan", "qwen3.5-plus"),
            ("alibaba-coding-plan", "qwen3-coder-plus"),
            ("alibaba-coding-plan", "qwen3-max-2026-01-23"),
            ("alibaba-coding-plan", "glm-5"),
            ("alibaba-coding-plan", "glm-4.7"),
            ("zai-coding-plan", "glm-4.5-air"),
            ("zai-coding-plan", "glm-4-plus"),
            ("zhipuai-coding-plan", "glm-4.5-flash"),
        ],
    )
    def test_text_only_variants_do_not_have_vision(
        self, provider: str, model_name: str
    ):
        """No false-positive vision after the -vl/-4v fix."""
        r = lookup(provider, model_name)
        assert "vision" not in r.abilities, (
            f"{provider}/{model_name} should be text-only; "
            f"matched={r.matched_pattern}, abilities={r.abilities}"
        )

    def test_provider_fallback_does_not_eat_cross_provider_model(self):
        """Regression for the resolution-tier ordering bug: previously
        ``openai/*`` was winning over ``*/deepseek-chat*``."""
        r = lookup("openai", "deepseek-chat")
        assert r.source == "wildcard_provider"
        assert r.matched_pattern == "*/deepseek-chat*"

    def test_deepseek_v4_remains_text_only(self):
        """Regression: DeepSeek V4 Preview is text-only and must not gain
        false-positive multimodal support from the catalog."""
        r = lookup("openai", "deepseek-v4")
        assert "vision" not in r.abilities
        assert "tool_calling" in r.abilities
        assert "thinking_mode" in r.abilities

    @pytest.mark.parametrize(
        "provider,model_name,must_include",
        [
            # Catalog rev2: GPT-5/5.5, Claude 4.5, Gemini 3, DeepSeek V4 etc.
            ("openai", "gpt-5.5", "thinking_mode"),
            ("openai", "gpt-5", "thinking_mode"),
            ("openai", "gpt-5-mini", "thinking_mode"),
            ("claude", "claude-opus-4.5", "thinking_mode"),
            ("gemini", "gemini-3-pro", "thinking_mode"),
            ("openai", "deepseek-v4", "thinking_mode"),
            ("openai", "deepseek-r2", "thinking_mode"),
        ],
    )
    def test_recent_models_picked_up(
        self, provider: str, model_name: str, must_include: str
    ):
        """Catalog rev2 added rules for newer models. Pin them so future
        edits don't accidentally remove."""
        r = lookup(provider, model_name)
        assert must_include in r.abilities, (
            f"{provider}/{model_name} expected to include {must_include}; "
            f"got {r.abilities} (matched={r.matched_pattern})"
        )
