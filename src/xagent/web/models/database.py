from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from xagent.db import try_upgrade_db

from ...core.storage.manager import get_default_db_url

_SessionLocal: sessionmaker[Session] | None = None

_engine: Engine | None = None

# Create base model class
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    if _SessionLocal is None:
        raise RuntimeError("Session Local is not initialized. Call init_db() first.")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session_local() -> sessionmaker[Session]:
    if _SessionLocal is None:
        raise RuntimeError("Session Local is not initialized. Call init_db() first.")
    return _SessionLocal


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Engine is not initialized. Call init_db() first.")
    return _engine


def init_db(db_url: str | None = None) -> None:
    """Initialize database, create all tables and default users"""
    import logging
    import os

    # Import all models to ensure they are registered with Base.metadata
    from . import (  # noqa: F401
        MCPServer,
        Model,
        SystemSetting,
        Task,
        TemplateStats,
        ToolConfig,
        ToolUsage,
        User,
        UserDefaultModel,
        UserModel,
    )
    from .agent import Agent  # noqa: F401
    from .text2sql import Text2SQLDatabase  # noqa: F401

    global _SessionLocal
    global _engine

    # Database configuration
    if db_url is not None:
        database_url = db_url
    else:
        database_url = os.getenv("DATABASE_URL") or get_default_db_url()

    # Create database engine
    # For SQLite, use NullPool to prevent connection pool issues
    # For other databases, use QueuePool with timeout settings
    if "sqlite" in database_url:
        from sqlalchemy.pool import NullPool

        _engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,  # SQLite doesn't need connection pooling
        )
    else:
        from sqlalchemy.pool import QueuePool

        _engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,  # 30 seconds timeout for getting connection from pool
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_pre_ping=True,  # Verify connections before using
        )

    # Create session factory
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    # Try upgrade db to head first
    try_upgrade_db(_engine)

    # Create all tables
    Base.metadata.create_all(bind=_engine)

    logger = logging.getLogger(__name__)
    logger.info("Database initialized. Waiting for first admin setup.")
