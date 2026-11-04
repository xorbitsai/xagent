"""
Database timezone utilities for handling different database backends
"""

import logging
from datetime import datetime, timezone
from typing import Any, Union

logger = logging.getLogger(__name__)


def get_database_timezone_aware() -> bool:
    """Check if the current database supports timezone-aware datetime"""
    from ...core.storage import get_default_db_url

    database_url = get_default_db_url()

    if database_url.lower().startswith("postgresql"):
        return True
    elif database_url.lower().startswith("mysql"):
        return True
    elif database_url.lower().startswith("sqlite"):
        return False
    else:
        # Default to False for unknown databases
        logger.warning(
            f"Unknown database type: {database_url}, assuming no timezone support"
        )
        return False


def normalize_datetime_for_db(dt: datetime) -> datetime:
    """
    Normalize datetime for database storage
    - For timezone-aware databases: store as UTC
    - For SQLite: store as naive datetime (assumed UTC)
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to UTC
    dt_utc = dt.astimezone(timezone.utc)

    if get_database_timezone_aware():
        # Database supports timezone - keep timezone info
        return dt_utc
    else:
        # SQLite doesn't support timezone - store as naive datetime
        return dt_utc.replace(tzinfo=None)


def normalize_datetime_from_db(dt: datetime) -> datetime:
    """
    Normalize datetime from database for API response
    - For SQLite: naive datetime is assumed UTC
    - For other databases: preserve timezone info
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        # Naive datetime from database - assume UTC
        return dt.replace(tzinfo=timezone.utc)
    else:
        # Already has timezone - convert to UTC for consistency
        return dt.astimezone(timezone.utc)


def format_datetime_for_api(dt: Union[datetime, None, Any]) -> Union[str, None]:
    """
    Format datetime for API response - always return UTC ISO format
    Works with all database backends
    """
    if dt is None:
        return None

    # Normalize from database format
    normalized_dt = normalize_datetime_from_db(dt)

    # Return ISO format string
    return normalized_dt.isoformat()


def safe_timestamp_to_unix(timestamp: Any) -> float:
    """
    Safely convert timestamp to Unix timestamp, handling different input types
    """
    if timestamp is None:
        raise ValueError("Timestamp cannot be None")
    elif isinstance(timestamp, (int, float)):
        return float(timestamp)
    elif isinstance(timestamp, datetime):
        normalized_dt = normalize_datetime_from_db(timestamp)
        return normalized_dt.timestamp()
    else:
        raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")
