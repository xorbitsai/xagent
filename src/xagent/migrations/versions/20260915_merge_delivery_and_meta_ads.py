"""Merge channel delivery retry and Meta Ads connector migration branches."""

revision = "20260915_merge_delivery_meta"
down_revision = (
    "20260915_delivery_retry_budget",
    "20260909_seed_meta_ads_mcp_app",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
