from dataclasses import fields
from typing import Callable

import pytest

from xagent import skills
from xagent.skills.library import (
    CompositeSkillLibraryProvider,
    FilesystemSkillLibraryProvider,
    SkillRecord,
    SkillScopeContext,
    SkillWriteContext,
)
from xagent.skills.manager import SkillManager


def test_skill_scope_context_contains_only_detached_runtime_identity() -> None:
    context = SkillScopeContext(user_id=7, metadata={"team_id": 11})

    assert {field.name for field in fields(context)} == {"user_id", "metadata"}
    assert context.user_id == 7
    assert context.metadata == {"team_id": 11}


def test_skill_scope_context_rejects_request_owned_metadata() -> None:
    with pytest.raises(TypeError, match="detached scalar"):
        SkillScopeContext(user_id=7, metadata={"db": object()})


@pytest.mark.parametrize("context_type", [SkillScopeContext, SkillWriteContext])
def test_detached_contexts_reject_scalar_subclasses_with_attached_resources(
    context_type: Callable[..., object],
) -> None:
    class _AttachedStr(str):
        pass

    class _AttachedInt(int):
        pass

    attached_key = _AttachedStr("team_id")
    attached_key.request = object()
    attached_value = _AttachedInt(11)
    attached_value.session = object()

    with pytest.raises(TypeError, match="detached scalar"):
        context_type(user_id=7, metadata={attached_key: 11})
    with pytest.raises(TypeError, match="detached scalar"):
        context_type(user_id=7, metadata={"team_id": attached_value})


@pytest.mark.parametrize("context_type", [SkillScopeContext, SkillWriteContext])
@pytest.mark.parametrize("user_id", [True, object()])
def test_detached_contexts_reject_non_exact_integer_user_ids(
    context_type: Callable[..., object], user_id: object
) -> None:
    class _AttachedInt(int):
        pass

    subclass_user_id = _AttachedInt(7)
    subclass_user_id.request = object()

    with pytest.raises(TypeError, match="user_id"):
        context_type(user_id=user_id)
    with pytest.raises(TypeError, match="user_id"):
        context_type(user_id=subclass_user_id)


def test_skill_scope_context_copies_and_freezes_metadata() -> None:
    metadata = {"team_id": 11}
    context = SkillScopeContext(user_id=7, metadata=metadata)

    metadata["team_id"] = 12

    assert context.metadata == {"team_id": 11}
    with pytest.raises(TypeError):
        context.metadata["team_id"] = 12  # type: ignore[index]


def test_skill_write_context_contains_only_detached_identity() -> None:
    metadata = {"source_id": 11}

    context = SkillWriteContext(user_id=7, metadata=metadata)

    assert {field.name for field in fields(context)} == {"user_id", "metadata"}
    assert context.user_id == 7
    assert context.metadata == {"source_id": 11}
    metadata["source_id"] = 12
    assert context.metadata == {"source_id": 11}
    with pytest.raises(TypeError):
        context.metadata["source_id"] = 13  # type: ignore[index]


def test_skill_provider_failure_contract_is_exported() -> None:
    from xagent.skills.library import (
        SkillWriteProviderError,
        SkillWriteProviderErrorReason,
    )

    error = SkillWriteProviderError(
        SkillWriteProviderErrorReason.FORBIDDEN,
        "You cannot write this skill.",
    )

    assert error.reason is SkillWriteProviderErrorReason.FORBIDDEN
    assert error.public_detail == "You cannot write this skill."
    assert skills.SkillWriteProviderError is SkillWriteProviderError
    assert skills.SkillWriteProviderErrorReason is SkillWriteProviderErrorReason
    assert hasattr(skills, "SkillWriteProvider")
    assert hasattr(skills, "get_skill_write_provider")
    assert hasattr(skills, "set_skill_write_provider")


class StaticProvider:
    def __init__(self, source: str, records: list[SkillRecord]):
        self.source = source
        self.records = records

    async def list_records(self, context: SkillScopeContext) -> list[SkillRecord]:
        return self.records

    async def read_file(
        self, context: SkillScopeContext, record: SkillRecord, path: str
    ) -> bytes:
        return record.files[path]


def _record(name: str, source: str, description: str) -> SkillRecord:
    return SkillRecord(
        name=name,
        source=source,
        scope=source,
        files={
            "SKILL.md": (
                f"---\ndescription: {description!r}\n"
                f"when_to_use: Use {source}.\n---\n# {name}\n"
            ).encode()
        },
    )


@pytest.mark.asyncio
async def test_composite_provider_later_records_override_earlier_by_name():
    provider = CompositeSkillLibraryProvider(
        [
            StaticProvider("builtin", [_record("writer", "builtin", "builtin")]),
            StaticProvider("team", [_record("writer", "team", "team")]),
            StaticProvider("personal", [_record("writer", "personal", "personal")]),
        ]
    )

    manager = SkillManager(provider=provider, context=SkillScopeContext(user_id=7))
    await manager.initialize()

    skill = await manager.get_skill("writer")

    assert skill is not None
    assert skill["description"] == "personal"
    assert skill["source"] == "personal"
    assert skill["scope"] == "personal"


@pytest.mark.asyncio
async def test_composite_records_survive_unavailable_personal_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from xagent.skills.personal_db import XagentPersonalDbSkillProvider
    from xagent.web.models import database

    filesystem_skill = tmp_path / "filesystem-writer"
    filesystem_skill.mkdir()
    (filesystem_skill / "SKILL.md").write_bytes(
        b"---\ndescription: filesystem\n---\n# Filesystem writer\n"
    )

    monkeypatch.setattr(database, "get_optional_session_local", lambda: None)
    provider = CompositeSkillLibraryProvider(
        [
            FilesystemSkillLibraryProvider([tmp_path]),
            StaticProvider(
                "overlay", [_record("overlay-writer", "overlay", "overlay")]
            ),
            XagentPersonalDbSkillProvider(),
        ]
    )

    records = await provider.list_records(SkillScopeContext(user_id=7))

    assert [record.name for record in records] == [
        "filesystem-writer",
        "overlay-writer",
    ]


@pytest.mark.asyncio
async def test_composite_propagates_personal_database_failure_without_partial_records(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from xagent.skills.personal_db import XagentPersonalDbSkillProvider
    from xagent.web.models import database

    error = RuntimeError("Session Local is not initialized. Call init_db() first.")

    def _failing_session_factory():
        raise error

    monkeypatch.setattr(
        database,
        "get_optional_session_local",
        lambda: _failing_session_factory,
    )
    provider = CompositeSkillLibraryProvider(
        [
            StaticProvider("builtin", [_record("writer", "builtin", "builtin")]),
            XagentPersonalDbSkillProvider(),
        ]
    )

    with pytest.raises(RuntimeError) as caught:
        await provider.list_records(SkillScopeContext(user_id=7))

    assert caught.value is error
    assert (
        str(caught.value) == "Session Local is not initialized. Call init_db() first."
    )
    assert not any("unavailable" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_composite_provider_can_return_visible_records_with_shadowed_state():
    provider = CompositeSkillLibraryProvider(
        [
            StaticProvider("team", [_record("writer", "team", "team")]),
            StaticProvider("personal", [_record("writer", "personal", "personal")]),
        ]
    )

    records = await provider.list_visible_records(SkillScopeContext(user_id=7))

    assert [(r.scope, r.name, r.effective, r.shadowed_by) for r in records] == [
        ("team", "writer", False, "personal"),
        ("personal", "writer", True, None),
    ]
