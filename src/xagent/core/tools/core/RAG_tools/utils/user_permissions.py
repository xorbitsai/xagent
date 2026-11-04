"""User permissions and access control for RAG tools."""

from typing import Optional

from ..core.config import UNAUTHENTICATED_NO_ACCESS_FILTER
from .user_scope import resolve_user_scope


class UserPermissions:
    """Handle user permissions and data access control."""

    @staticmethod
    def get_no_access_filter() -> str:
        """Return a stable LanceDB filter expression that always matches no rows."""
        return UNAUTHENTICATED_NO_ACCESS_FILTER

    @staticmethod
    def is_no_access_filter(filter_expr: Optional[str]) -> bool:
        """Check whether a filter expression is the internal no-access marker."""
        return filter_expr == UNAUTHENTICATED_NO_ACCESS_FILTER

    @staticmethod
    def get_user_filter(
        user_id: Optional[int], is_admin: Optional[bool] = None
    ) -> Optional[str]:
        """
        Generate user filter expression for LanceDB queries.

        Args:
            user_id: Current user ID, None for unauthenticated
            is_admin: Whether user is admin (None to fallback to context)

        Returns:
            LanceDB filter expression string, or None for no filtering

        Note:
            Legacy data (user_id = NULL) is only accessible to admin users.
            Regular users can only see data they explicitly own (user_id == their ID).
        """
        scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)

        if scope.is_admin:
            # Admins can see all data (including NULL user_id legacy data)
            return None
        elif scope.user_id is not None:
            # Regular users can ONLY see their own data
            # Legacy data (NULL user_id) is NOT visible to regular users
            return f"user_id == {int(scope.user_id)}"
        else:
            # Unauthenticated users cannot see any data
            return UserPermissions.get_no_access_filter()

    @staticmethod
    def can_access_data(
        user_id: Optional[int],
        data_user_id: Optional[int],
        is_admin: Optional[bool] = None,
    ) -> bool:
        """
        Check if user can access specific data.

        Args:
            user_id: Current user ID
            data_user_id: Data owner's user ID
            is_admin: Whether current user is admin (None to fallback to context)

        Returns:
            True if access allowed

        Note:
            Legacy data (data_user_id = NULL) is only accessible to admin users.
            Regular users can only access data they explicitly own.
        """
        scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)

        if scope.is_admin:
            # Admins can access all data including legacy (NULL) data
            return True
        if scope.user_id is None:
            # Unauthenticated users cannot access any data
            return False
        # Users can ONLY access their own data
        # Legacy data (NULL data_user_id) is NOT accessible to regular users
        return data_user_id == scope.user_id

    @staticmethod
    def get_write_user_id(user_id: Optional[int]) -> Optional[int]:
        """
        Get user_id for writing new data.

        Args:
            user_id: Current user ID

        Returns:
            user_id to use for new data, None for legacy compatibility
        """
        return resolve_user_scope(user_id=user_id, is_admin=None).user_id
