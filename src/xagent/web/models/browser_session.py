from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, String

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
    # The user's platform-admin role at the last login, activity or refresh.
    # With the policy off, expiry is judged by the stricter of this and the
    # current role, so a demotion after switch-off cannot revive a login.
    recorded_admin = Column(Boolean, nullable=False, default=False)


class BrowserIdleEnforcement(Base):  # type: ignore[no-any-unimported]
    """How long the idle policy is known to be enforced, per threshold set.

    Every replica with the policy enabled pushes the row for its own
    thresholds a lease ahead of now. With the policy disabled, a login that
    expired under any recorded threshold set before that set's
    ``enforced_until`` stays rejected, so switching the policy off cannot
    revive it, even after a rolling threshold change.
    """

    __tablename__ = "browser_idle_enforcement"

    ordinary_minutes = Column(Integer, primary_key=True)
    admin_minutes = Column(Integer, primary_key=True)
    enforced_until = Column(DateTime(timezone=True), nullable=False)
