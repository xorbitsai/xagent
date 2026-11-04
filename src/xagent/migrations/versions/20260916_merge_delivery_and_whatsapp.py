"""Merge shared delivery and WhatsApp connector migration branches."""

revision = "20260916_merge_delivery_whatsapp"
down_revision = (
    "20260915_merge_delivery_meta",
    "20260909_seed_whatsapp_mcp_app",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
