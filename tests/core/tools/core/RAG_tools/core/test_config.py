"""Tests for RAG tool configuration defaults and their env overrides."""

from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy


def test_compaction_thresholds_read_the_environment(monkeypatch):
    """The thresholds govern an always-on inline path, so they need a stand-down
    switch that does not require a redeploy."""
    monkeypatch.setenv("XAGENT_KB_COMPACT_FRAGMENT_THRESHOLD", "5000")
    monkeypatch.setenv("XAGENT_KB_COMPACT_VERSION_THRESHOLD", "6000")
    monkeypatch.setenv("XAGENT_KB_VERSION_RETENTION_DAYS", "30")
    policy = IndexPolicy()
    assert policy.compact_fragment_threshold == 5000
    assert policy.compact_stale_version_threshold == 6000
    assert policy.version_retention_days == 30


def test_compaction_thresholds_ignore_unusable_environment_values(monkeypatch):
    """A typo must not construct a policy that compacts on every single write."""
    monkeypatch.setenv("XAGENT_KB_COMPACT_FRAGMENT_THRESHOLD", "nonsense")
    monkeypatch.setenv("XAGENT_KB_COMPACT_VERSION_THRESHOLD", "-1")
    monkeypatch.setenv("XAGENT_KB_VERSION_RETENTION_DAYS", "0")
    policy = IndexPolicy()
    assert policy.compact_fragment_threshold == 100
    assert policy.compact_stale_version_threshold == 100
    assert policy.version_retention_days == 7
