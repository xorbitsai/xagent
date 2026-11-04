"""Join routine history and Freshdesk visibility migration histories."""

revision = "20261030_merge_history_freshdesk"
down_revision = (
    "20261030_user_turn_uniqueness",
    "20261028_enable_freshdesk",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
