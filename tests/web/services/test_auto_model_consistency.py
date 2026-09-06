"""Auto binding, default resolution and lifecycle regressions against a real DB."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.model.chat.basic.router import RouterLLM, _ResolvedRouterLLM
from xagent.web import models
from xagent.web.api import admin_users, agents, workforces
from xagent.web.api import model as model_api
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models import database
from xagent.web.models.auto_model import AutoModelCandidate, AutoModelConfig
from xagent.web.models.model import Model
from xagent.web.models.user import User, UserDefaultModel, UserModel
from xagent.web.schemas.model import AutoModelConfigUpdate, ModelUpdate
from xagent.web.services import auto_model_service, builder_chat_runtime, model_service
from xagent.web.services.auto_model_service import (
    AutoModelConfigurationError,
    AutoModelService,
)
from xagent.web.services.client_error_messages import CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE
from xagent.web.services.llm_utils import (
    AutoModelUnavailableError,
    UserAwareModelStorage,
)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path))
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def foreign_keys(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")

    models.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = factory()

    def get_db():
        yield db

    monkeypatch.setattr(database, "get_db", get_db)
    monkeypatch.setattr(admin_users, "get_session_local", lambda: factory)
    monkeypatch.setattr(builder_chat_runtime, "get_session_local", lambda: factory)
    monkeypatch.setattr(model_service, "_visible_user_ids_hook", None)
    profiles = {
        "text-a": SimpleNamespace(input_modalities=("text",)),
        "text-b": SimpleNamespace(input_modalities=("text",)),
        "image-a": SimpleNamespace(input_modalities=("text", "image")),
    }
    catalog = SimpleNamespace(
        known_model_ids=lambda: tuple(profiles), get=profiles.__getitem__
    )
    monkeypatch.setattr(
        auto_model_service, "load_router_profile_catalog", lambda: catalog
    )
    owner = User(username="owner", password_hash="unused", is_admin=True)
    consumer = User(username="consumer", password_hash="unused", is_admin=False)
    db.add_all([owner, consumer])
    db.commit()
    yield SimpleNamespace(db=db, factory=factory, owner=owner, consumer=consumer)
    db.close()
    engine.dispose()


def target(env, owner, *, name="saved", abilities=None, shared=False):
    row = Model(
        model_id=name,
        category="llm",
        model_provider="openai",
        model_name="gpt-4",
        api_key="fake-test-key",
        base_url="https://example.invalid/v1",
        abilities=abilities or ["chat"],
        is_active=True,
    )
    env.db.add(row)
    env.db.flush()
    env.db.add(
        UserModel(
            user_id=owner.id,
            model_id=row.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=shared,
        )
    )
    env.db.commit()
    return row


def configure(env, user, bindings, *, default=False):
    return AutoModelService(env.db).upsert_config(
        user_id=user.id,
        request=AutoModelConfigUpdate(
            candidates=[
                {"routing_model_id": profile, "target_model_id": t.id}
                for profile, t in bindings
            ],
            fallback_model_id=bindings[0][1].id,
            set_as_default=default,
        ),
    )


def broken_default(env, slot="general"):
    t = target(env, env.consumer)
    cfg = configure(env, env.consumer, [("text-a", t)])
    env.db.add(
        UserDefaultModel(
            user_id=env.consumer.id, model_id=cfg.router_model_id, config_type=slot
        )
    )
    t.is_active = False
    env.db.commit()
    return cfg


@pytest.mark.parametrize(
    "profile,abilities", [("image-a", ["chat"]), ("text-a", ["chat", "vision"])]
)
def test_rejects_profile_capability_mismatch_without_saving(env, profile, abilities):
    t = target(env, env.consumer, abilities=abilities)
    with pytest.raises(AutoModelConfigurationError, match="input modalities"):
        configure(env, env.consumer, [(profile, t)])
    assert env.db.query(AutoModelConfig).count() == 0


def test_legacy_mismatched_binding_fails_before_provider_call(env):
    t = target(env, env.consumer)
    cfg = configure(env, env.consumer, [("text-a", t)])
    cfg.candidates[0].routing_model_id = "image-a"
    env.db.commit()
    with pytest.raises(AutoModelUnavailableError, match="input modalities"):
        UserAwareModelStorage(env.db).get_llm_by_id(
            cfg.router_model.model_id, env.consumer.id
        )


def test_configured_wrapper_never_adds_profile_only_abilities():
    downstream = SimpleNamespace(abilities=["chat"])
    router = RouterLLM(
        candidate_models=["image-a"], downstream_resolver=lambda _: downstream
    )
    resolved = _ResolvedRouterLLM(
        router=router,
        downstream=downstream,
        selected_model="image-a",
        context_window=None,
        input_modalities=("image",),
    )
    assert resolved.abilities == ["chat"]


@pytest.mark.parametrize(
    "changes",
    [
        {"model_name": "gpt-4.1"},
        {"model_provider": "deepseek", "model_name": "deepseek-v4-flash"},
        {"base_url": "https://another.invalid/v1"},
        {"abilities": ["chat", "vision"]},
    ],
)
def test_bound_candidate_identity_and_modalities_cannot_change(env, changes):
    t = target(env, env.consumer)
    configure(env, env.consumer, [("text-a", t)])
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            model_api.update_model(
                t.model_id, ModelUpdate(**changes), db=env.db, user=env.consumer
            )
        )
    assert error.value.status_code == 409
    env.db.rollback()
    assert t.model_name == "gpt-4" and t.abilities == ["chat"]


def test_owner_identity_change_prunes_external_binding(env):
    t = target(env, env.owner, shared=True)
    cfg = configure(env, env.consumer, [("text-a", t)])
    asyncio.run(
        model_api.update_model(
            t.model_id, ModelUpdate(model_name="gpt-4.1"), db=env.db, user=env.owner
        )
    )
    env.db.expire_all()
    assert env.db.query(AutoModelCandidate).filter_by(config_id=cfg.id).count() == 0
    assert env.db.get(AutoModelConfig, cfg.id).fallback_model_id is None


@pytest.mark.parametrize("configured_compact", [False, True])
def test_missing_compact_does_not_resolve_broken_unrelated_general(
    env, configured_compact
):
    broken_default(env)
    healthy = target(env, env.consumer, name="healthy")
    if configured_compact:
        env.db.add(
            UserDefaultModel(
                user_id=env.consumer.id, model_id=healthy.id, config_type="compact"
            )
        )
        env.db.commit()
    resolved = UserAwareModelStorage(env.db).resolve_llms_from_names(
        [healthy.model_id] * 3 + [None], env.consumer.id
    )
    assert all(llm.model_id == healthy.model_id for llm in resolved)
    builder = builder_chat_runtime._load_builder_chat_runtime_inputs_sync(
        user_id=env.consumer.id,
        requested_file_ids=[],
        model_name=healthy.model_id,
        compact_model_name=None,
    )
    assert builder.llm.model_id == builder.compact_llm.model_id == healthy.model_id


@pytest.mark.parametrize(
    "slot,getter",
    [
        ("general", "get_default_model"),
        ("small_fast", "get_fast_model"),
        ("visual", "get_default_vision_model"),
        ("compact", "get_compact_model"),
    ],
)
def test_default_getters_preserve_auto_configuration_error(env, slot, getter):
    broken_default(env, slot)
    with pytest.raises(AutoModelUnavailableError):
        UserAwareModelStorage(env.db).get_configured_defaults(env.consumer.id)
    kwargs = {"db": env.db} if slot == "visual" else {}
    with pytest.raises(AutoModelUnavailableError):
        getattr(model_service, getter)(env.consumer.id, **kwargs)


@pytest.mark.parametrize("surface", ["optimize", "workforce"])
def test_http_entrypoints_map_actual_broken_auto_to_safe_409(env, surface):
    broken_default(env)
    app = FastAPI()
    app.include_router(agents.router if surface == "optimize" else workforces.router)

    def db_override():
        yield env.db

    app.dependency_overrides[agents.get_db] = db_override
    app.dependency_overrides[workforces.get_db] = db_override
    app.dependency_overrides[get_current_user] = lambda: env.consumer
    with TestClient(app) as client:
        if surface == "optimize":
            response = client.post(
                "/api/agents/optimize-instructions",
                json={"instructions": "Write a clear summary."},
            )
        else:
            response = client.post(
                "/api/workforces/from-prompt", json={"prompt": "Write a clear summary."}
            )
    assert response.status_code == 409
    assert response.json()["detail"] == CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE


@pytest.mark.parametrize("retained_grant", [False, True])
def test_admin_delete_prunes_only_inaccessible_bindings(env, retained_grant):
    t = target(env, env.owner, shared=True, abilities=["chat", "tool_calling"])
    cfg = configure(env, env.consumer, [("text-a", t)])
    ids = env.owner.id, env.consumer.id, t.id, cfg.id, cfg.router_model_id
    if retained_grant:
        env.db.add(UserModel(user_id=env.consumer.id, model_id=t.id, is_owner=True))
        env.db.commit()
    env.db.close()
    assert admin_users._delete_user_rows_sync(user_id=ids[0]) is True
    with env.factory() as db:
        assert db.get(User, ids[0]) is None
        assert db.get(Model, ids[2]) is not None
        assert db.query(AutoModelCandidate).filter_by(config_id=ids[3]).count() == int(
            retained_grant
        )
        assert db.get(AutoModelConfig, ids[3]).fallback_model_id == (
            ids[2] if retained_grant else None
        )
        assert db.get(Model, ids[4]).abilities == (
            ["chat", "tool_calling"] if retained_grant else ["chat"]
        )


def test_compatible_ability_edit_refreshes_auto_without_locking_credentials(env):
    t = target(env, env.consumer, abilities=["chat", "tool_calling"])
    cfg = configure(env, env.consumer, [("text-a", t)])
    asyncio.run(
        model_api.update_model(
            t.model_id,
            ModelUpdate(
                abilities=["chat"], api_key="new-fake-key", base_url=t.base_url
            ),
            db=env.db,
            user=env.consumer,
        )
    )
    env.db.expire_all()
    assert env.db.get(Model, cfg.router_model_id).abilities == ["chat"]
    assert t.api_key == "new-fake-key"


def test_pruning_candidate_recalculates_remaining_abilities(env):
    shared = target(env, env.owner, name="shared", shared=True)
    own = target(env, env.consumer, name="own", abilities=["chat", "tool_calling"])
    cfg = configure(env, env.consumer, [("text-a", shared), ("text-b", own)])
    assert "tool_calling" not in cfg.router_model.abilities
    asyncio.run(
        model_api.update_model(
            shared.model_id,
            ModelUpdate(share_with_users=False),
            db=env.db,
            user=env.owner,
        )
    )
    env.db.expire_all()
    assert env.db.get(Model, cfg.router_model_id).abilities == ["chat", "tool_calling"]


def test_editing_legacy_unknown_profile_returns_conflict(env):
    t = target(env, env.consumer)
    cfg = configure(env, env.consumer, [("text-a", t)])
    cfg.candidates[0].routing_model_id = "removed-profile"
    env.db.commit()
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            model_api.update_model(
                t.model_id,
                ModelUpdate(abilities=["chat", "tool_calling"]),
                db=env.db,
                user=env.consumer,
            )
        )
    assert error.value.status_code == 409
    with pytest.raises(AutoModelUnavailableError, match="Unknown Auto profile"):
        UserAwareModelStorage(env.db).get_llm_by_id(
            cfg.router_model.model_id, env.consumer.id
        )
