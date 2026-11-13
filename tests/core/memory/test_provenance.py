"""Tests for host memory provenance validation."""

from types import MappingProxyType
from typing import Any

import pytest

from xagent.core.memory.provenance import validate_memory_provenance


def test_validate_memory_provenance_returns_frozen_copy() -> None:
    source = {"channel_kind": "dm", "channel_id": "D123ABC", "task_id": "42"}

    validated = validate_memory_provenance(source)

    assert isinstance(validated, MappingProxyType)
    assert dict(validated) == source
    # A later host-side mutation cannot reach notes stored from the copy.
    source["channel_id"] = "C999"
    assert validated["channel_id"] == "D123ABC"


@pytest.mark.parametrize(
    "identifier",
    [
        "C0123ABCD",  # Slack public channel
        "D0123ABCD",  # Slack DM
        "G0123ABCD",  # Slack group DM / private channel
        "19:abcDEF0123@thread.tacv2",  # Teams channel conversation
        "a:1Xyz-AbC_dEf+gh/ij=",  # Teams personal conversation
        "oc_5ad11d72b830411d72b836c20",  # Feishu chat
        "42",  # task id
    ],
)
def test_validate_memory_provenance_accepts_platform_identifiers(
    identifier: str,
) -> None:
    assert validate_memory_provenance({"channel_id": identifier})["channel_id"] == (
        identifier
    )


@pytest.mark.parametrize(
    "provenance",
    [
        # Conversation text can never pass the identifier grammar (#2753).
        {"channel_id": "please remember my salary is 100k"},
        {"note": "line\nbreak"},
        # Spaceless prose: non-ASCII scripts and prose punctuation fail too.
        {"channel_id": "我的银行卡密码是123456请帮我记住"},
        {"channel_id": "remember_this,it's_my_pin!"},
        {"channel_id": "café"},
        {"channel_id": ""},
        {"channel_id": "x" * 257},
        {"bad key": "C1"},
        {"": "C1"},
        {"channel:id": "C1"},
    ],
)
def test_validate_memory_provenance_rejects_text_and_bad_keys(
    provenance: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        validate_memory_provenance(provenance)


@pytest.mark.parametrize(
    "provenance",
    [{}, {"task_id": 42}, {"channel_id": None}, ["channel_id", "C1"]],
)
def test_validate_memory_provenance_rejects_wrong_shapes(provenance: Any) -> None:
    with pytest.raises(TypeError):
        validate_memory_provenance(provenance)
