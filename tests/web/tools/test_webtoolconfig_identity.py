"""Privilege-isolation pins for WebToolConfig identity.

When the runtime builds a tool config for a task OWNER while an admin is the
acting principal on the request, the config must reflect the owner -- not get
silently widened to admin scope by the request.
"""

from types import SimpleNamespace

from xagent.web.tools.config import WebToolConfig


def _admin_request(actor_user_id: int = 999):
    """A request whose authenticated user is an admin (the acting principal)."""
    return SimpleNamespace(user=SimpleNamespace(id=actor_user_id, is_admin=True))


def test_explicit_is_admin_false_wins_over_admin_request() -> None:
    """Tri-state: an explicit ``is_admin=False`` (the owner's status) is
    authoritative and is NOT OR-ed with the admin request. Otherwise an admin
    acting on another user's task would get admin-scoped tools/visibility."""
    cfg = WebToolConfig(
        db=None,
        request=_admin_request(actor_user_id=999),
        user_id=42,  # the task owner
        is_admin=False,  # owner is not an admin
    )
    assert cfg.get_user_id() == 42
    assert cfg.is_admin() is False


def test_unset_is_admin_falls_back_to_request() -> None:
    """When ``is_admin`` is unset (None), fall back to the request's user --
    preserves behavior for callers that don't pass an explicit value."""
    cfg = WebToolConfig(
        db=None,
        request=_admin_request(actor_user_id=7),
        user_id=7,
    )
    assert cfg.is_admin() is True


def test_minimal_request_without_user_is_not_admin() -> None:
    """A minimal request carrying only an id (no ``user``) must resolve to
    non-admin without raising / logging a spurious warning."""
    cfg = WebToolConfig(db=None, request=SimpleNamespace(), user_id=5)
    assert cfg.is_admin() is False


def test_skill_scope_context_does_not_retain_request_resources() -> None:
    from xagent.web.services.skill_runtime import build_detached_skill_scope

    db = object()
    request = _admin_request(actor_user_id=999)
    explicit_user = SimpleNamespace(id=42, is_admin=False)
    cfg = WebToolConfig(
        db=db,
        request=request,
        user_id=42,
        is_admin=False,
        user=explicit_user,
    )

    context = cfg.get_skill_scope_context()

    assert context == build_detached_skill_scope(user_id=42)
    assert not hasattr(context, "db")
    assert not hasattr(context, "request")
    assert not hasattr(context, "user")
