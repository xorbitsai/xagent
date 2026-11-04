"""Join email confirmation with the session/Planner integration head."""

revision = "20261016_email_session_merge"
down_revision = (
    "20261016_email_confirmation",
    "20261016_session_planner_merge",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
