"""Join session revocation and the published Planner scope migration."""

revision = "20261016_session_planner_merge"
down_revision = (
    "20261016_user_session_epoch",
    "20261015_planner_team_read_scope",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
