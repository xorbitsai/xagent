"""Session transaction bookkeeping in models/database.py.

Covers release_db_connection_if_clean (issue #889) and the root-transaction-end
counter that the connector hook session boundary check reads.
"""

import pytest
from sqlalchemy import Column, Integer, String, create_engine, insert, text, update
from sqlalchemy.orm import declarative_base, sessionmaker

from xagent.web.models.database import (
    _ROOT_TXN_END_COUNT_KEY,
    release_db_connection_if_clean,
    root_transaction_end_count,
)

Base = declarative_base()


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)


def _make_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


def test_releases_read_only_transaction():
    db = _make_session()
    db.query(Item).all()
    assert db.in_transaction()

    assert release_db_connection_if_clean(db) is True
    assert not db.in_transaction()

    # Session stays usable and re-acquires a connection on the next query.
    assert db.query(Item).all() == []


def test_keeps_pending_writes():
    db = _make_session()
    db.add(Item(name="pending"))

    assert release_db_connection_if_clean(db) is False
    assert Item in {type(obj) for obj in db.new} or len(db.new) == 1

    db.commit()
    assert db.query(Item).count() == 1


def test_keeps_flushed_but_uncommitted_changes():
    """flush() empties new/dirty/deleted while the transaction still holds
    unpersisted DML; the helper must not roll that back."""
    db = _make_session()
    db.add(Item(name="one"))
    db.flush()
    assert not (db.new or db.dirty or db.deleted)

    assert release_db_connection_if_clean(db) is False

    db.commit()
    assert db.query(Item).count() == 1

    # After the commit the flush flag is cleared: a fresh read-only
    # transaction is releasable again.
    db.query(Item).all()
    assert release_db_connection_if_clean(db) is True


def test_keeps_core_dml_insert():
    """Core DML via Session.execute() never touches new/dirty/deleted and
    emits no after_flush; the do_orm_execute listener must catch it."""
    db = _make_session()
    db.execute(insert(Item).values(name="core"))
    assert not (db.new or db.dirty or db.deleted)

    assert release_db_connection_if_clean(db) is False

    db.commit()
    assert db.query(Item).count() == 1


def test_keeps_core_dml_update():
    db = _make_session()
    db.add(Item(name="one"))
    db.commit()

    db.execute(update(Item).values(name="core-updated"))
    assert release_db_connection_if_clean(db) is False

    db.commit()
    assert db.query(Item).first().name == "core-updated"


def test_keeps_textual_statements_conservatively():
    """text() statements can't be proven read-only; the helper must keep the
    connection even for a textual SELECT."""
    db = _make_session()
    db.execute(text("UPDATE items SET name = 'via-text'"))
    assert release_db_connection_if_clean(db) is False
    db.commit()

    db.execute(text("SELECT 1"))
    assert release_db_connection_if_clean(db) is False
    db.rollback()

    # A provable ORM SELECT after the rollback is releasable again.
    db.query(Item).all()
    assert release_db_connection_if_clean(db) is True


def test_savepoint_commit_preserves_outer_write_flag():
    """Savepoint completion must not clear the write flag: after_commit-style
    events fire for begin_nested() too while the outer transaction (and its
    flushed writes) is still open."""
    db = _make_session()
    db.add(Item(name="outer"))
    db.flush()

    nested = db.begin_nested()
    nested.commit()

    assert release_db_connection_if_clean(db) is False
    db.commit()
    assert db.query(Item).count() == 1


def test_savepoint_rollback_preserves_outer_write_flag():
    db = _make_session()
    db.add(Item(name="outer"))
    db.flush()

    nested = db.begin_nested()
    nested.rollback()

    assert release_db_connection_if_clean(db) is False
    db.commit()
    assert db.query(Item).count() == 1


def test_root_rollback_clears_write_flag():
    db = _make_session()
    db.add(Item(name="discarded"))
    db.flush()
    db.rollback()

    db.query(Item).all()
    assert release_db_connection_if_clean(db) is True


def test_keeps_flushed_dirty_changes():
    db = _make_session()
    db.add(Item(name="one"))
    db.commit()

    item = db.query(Item).first()
    item.name = "changed"

    assert release_db_connection_if_clean(db) is False
    db.commit()
    assert db.query(Item).first().name == "changed"


def test_none_session_is_noop():
    assert release_db_connection_if_clean(None) is False


def test_no_transaction_returns_true():
    db = _make_session()
    assert release_db_connection_if_clean(db) is True


def test_an_object_with_no_info_mapping_reports_no_count():
    """A duck-typed session with no ``.info`` cannot report a count at all."""

    class _NoInfo:
        pass

    assert root_transaction_end_count(_NoInfo()) is None


def test_info_present_but_not_a_mapping_raises():
    class _BadInfo:
        info = "not-a-mapping"

    with pytest.raises(AttributeError):
        root_transaction_end_count(_BadInfo())


def test_a_string_value_in_the_counter_key_raises():
    db = _make_session()
    db.info[_ROOT_TXN_END_COUNT_KEY] = "not-a-count"

    with pytest.raises(TypeError):
        root_transaction_end_count(db)


def test_a_boolean_value_in_the_counter_key_raises():
    """``isinstance(True, int)`` is ``True`` in Python; a hook writing a bare
    boolean into the counter key must not be read back as a count of 1."""
    db = _make_session()
    db.info[_ROOT_TXN_END_COUNT_KEY] = True

    with pytest.raises(TypeError):
        root_transaction_end_count(db)


def test_the_counter_and_the_write_flag_do_not_disturb_each_other():
    """The write-flag listener and the root-transaction-end counter listener
    both fire on ``after_transaction_end`` and must not step on each other."""
    db = _make_session()
    db.add(Item(name="outer"))
    db.flush()

    nested = db.begin_nested()
    nested.commit()

    # Savepoint completion clears neither the write flag nor the count.
    assert release_db_connection_if_clean(db) is False
    assert root_transaction_end_count(db) == 0

    db.commit()
    assert root_transaction_end_count(db) == 1

    db.query(Item).all()
    assert release_db_connection_if_clean(db) is True
    assert root_transaction_end_count(db) == 2


def test_a_non_count_value_does_not_raise_from_inside_the_listener():
    """A hook that leaves a non-count value in the counter key while the
    transaction is still open must not make ``commit()`` itself raise: the
    listener has to tolerate it and reset, leaving the refusal to
    ``root_transaction_end_count`` on the next read."""
    db = _make_session()
    db.add(Item(name="one"))
    db.info[_ROOT_TXN_END_COUNT_KEY] = "not-a-count"

    db.commit()

    assert db.query(Item).count() == 1
    assert root_transaction_end_count(db) == 1
