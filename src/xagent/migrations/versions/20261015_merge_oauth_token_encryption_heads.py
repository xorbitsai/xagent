"""Merge the OAuth token encryption and task command migration heads."""

from typing import Sequence, Union

revision: str = "20261015_merge_oauth_token_encryption_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20261014_encrypt_user_oauth_tokens",
    "20261014_task_cmd_claimable_kind",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
