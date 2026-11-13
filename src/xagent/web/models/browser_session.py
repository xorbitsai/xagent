from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String

from .database import Base


class BrowserSession(Base):  # type: ignore[no-any-unimported]
    """One interactive browser login and its idle-expiry state.

    Tokens issued for the login carry ``sid``. See ``browser_idle.py`` and
    docs/browser-idle-timeout.md; nothing else writes these rows.
    """

    __tablename__ = "browser_sessions"
    __table_args__ = (
        Index("ix_browser_sessions_user_id", "user_id"),
        Index("ix_browser_sessions_last_used_at", "last_used_at"),
    )

    sid = Column(String(32), primary_key=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(DateTime(timezone=True), nullable=False)
    # Moves only on login and explicit activity reports, never on refresh.
    last_activity_at = Column(DateTime(timezone=True), nullable=False)
    # Latest activity or refresh; drives retention only.
    last_used_at = Column(DateTime(timezone=True), nullable=False)
    # Once set, every credential carrying this sid is rejected, whether or not
    # the idle policy is currently enabled.
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_reason = Column(String(16), nullable=True)
