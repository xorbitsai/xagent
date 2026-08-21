"""create_memory_store's model-hub engine points at the same database as
the shared web engine (web/models/database.py) -- the Model table it
reads/writes has an encrypted API key column, so a bind/commit failure
here must not surface it as a bound SQL parameter either.
"""

from __future__ import annotations

from unittest.mock import Mock

import sqlalchemy

from xagent.web import memory_utils
from xagent.web.user_isolated_memory import UserIsolatedMemoryStore


def test_create_memory_store_hides_bound_parameters(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'memory_hub.db'}"
    monkeypatch.setattr(
        "xagent.core.storage.manager.get_default_db_url", lambda: db_url
    )
    create_engine_mock = Mock(wraps=sqlalchemy.create_engine)
    monkeypatch.setattr(sqlalchemy, "create_engine", create_engine_mock)

    store = memory_utils.create_memory_store()

    assert isinstance(store, UserIsolatedMemoryStore)
    # Asserting call_count too, not just call_args (the LAST call) --
    # otherwise a future second create_engine() call added anywhere in
    # this no-embedding-model fallback path could go unnoticed by this
    # assertion even if it lacked hide_parameters.
    assert create_engine_mock.call_count == 1
    assert create_engine_mock.call_args.kwargs["hide_parameters"] is True
