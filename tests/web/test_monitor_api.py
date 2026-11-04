"""
测试监控 API 的数据库兼容性
"""

import json
import re
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy import JSON, Column, Integer
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declarative_base

from xagent.web.api.monitor import (
    PG_ESCAPED_BACKSLASH,
    PG_ESCAPED_BACKSLASH_STANDIN,
    PG_SURROGATE_PAIR_PATTERN,
    PG_UNSAFE_ESCAPE_PATTERN,
    get_json_field_expression,
)

Base = declarative_base()


class MockTraceEvent(Base):
    """模拟的 TraceEvent 模型"""

    __tablename__ = "trace_events"

    event_id = Column(Integer, primary_key=True)
    data = Column(JSON)


class TestMonitorDatabaseCompatibility:
    """测试监控 API 的数据库兼容性"""

    def test_postgresql_json_extraction(self):
        """测试 PostgreSQL 的 JSON 字段提取"""
        # 创建模拟的数据库会话
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"
        mock_session.bind = mock_engine

        # 测试字段提取
        column = MockTraceEvent.data
        result = get_json_field_expression(column, "tool_name", mock_session)

        # 验证 PostgreSQL 语法 - 检查是否使用了 ->> 操作符
        result_str = str(result)
        assert "->>" in result_str
        assert "trace_events.data" in result_str

    def test_postgresql_guard_compiles_to_operators_postgresql_has(self):
        """The control-character guard must use operators that exist.

        ``~?`` is not a PostgreSQL operator for ``json``, ``jsonb`` or
        ``text``, so the guard failed every monitoring query on PostgreSQL
        while this SQLite-oriented suite stayed green (#1149). The real
        regex-match operator is ``~`` and it takes a text operand, hence the
        cast. Compiling for the PostgreSQL dialect is what makes this a guard
        rather than a restatement -- ``str(expr)`` renders the same either way.
        """
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"
        mock_session.bind = mock_engine

        column = MockTraceEvent.data
        compiled = get_json_field_expression(column, "tool_name", mock_session).compile(
            dialect=postgresql.dialect()
        )

        statement = str(compiled)
        assert "~?" not in statement
        assert " ~ " in statement
        assert "CAST(trace_events.data AS TEXT)" in statement
        assert "->>" in statement
        # The pattern travels as a bind parameter, not as inlined SQL.
        assert PG_UNSAFE_ESCAPE_PATTERN in compiled.params.values()

    def test_mysql_json_extraction(self):
        """测试 MySQL 的 JSON 字段提取"""
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "mysql"
        mock_session.bind = mock_engine

        column = MockTraceEvent.data
        result = get_json_field_expression(column, "tool_name", mock_session)

        # MySQL 应该使用 JSON_EXTRACT 函数
        assert "json_extract" in str(result).lower()

    def test_sqlite_json_extraction(self):
        """测试 SQLite 的 JSON 字段提取"""
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"
        mock_session.bind = mock_engine

        column = MockTraceEvent.data
        result = get_json_field_expression(column, "tool_name", mock_session)

        # SQLite 应该使用 json_extract 函数
        assert "json_extract" in str(result).lower()

    def test_field_path_formatting(self):
        """测试字段路径格式化"""
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"
        mock_session.bind = mock_engine

        column = MockTraceEvent.data

        # 测试带 '$.' 前缀的路径
        result1 = get_json_field_expression(column, "$.tool_name", mock_session)

        # 测试不带前缀的路径
        result2 = get_json_field_expression(column, "tool_name", mock_session)

        # 两种方式应该产生相同的结果
        assert str(result1) == str(result2)

    def test_none_bind_error(self):
        """测试 bind 为 None 时抛出错误"""
        mock_session = MagicMock()
        mock_session.bind = None

        column = MockTraceEvent.data

        with pytest.raises(ValueError, match="Database session bind is None"):
            get_json_field_expression(column, "tool_name", mock_session)


NUL = chr(0x0000)
LONE_HIGH = chr(0xD800)
LONE_LOW = chr(0xDC00)
NON_BMP = chr(0x1F600)  # json.dumps writes this as a valid surrogate pair
BACKSLASH = chr(92)


def matches_unsafe_escape(payload_text: str) -> bool:
    """Run the PostgreSQL guard's matching rules over a payload's text form.

    Mirrors what the SQL does, in the same order: neutralize escaped
    backslashes, delete valid surrogate pairs, then look for an escape
    PostgreSQL will not convert to text. Python's ``re`` and PostgreSQL's ARE
    agree on this pattern -- character classes, bounded repeats and plain
    alternation only, no lookaround -- so this is a faithful stand-in that runs
    without a database.
    """
    normalized = payload_text.replace(
        PG_ESCAPED_BACKSLASH, PG_ESCAPED_BACKSLASH_STANDIN
    )
    unpaired_only = re.sub(PG_SURROGATE_PAIR_PATTERN, "", normalized)
    return re.search(PG_UNSAFE_ESCAPE_PATTERN, unpaired_only) is not None


def guard_would_drop(value: object) -> bool:
    """Whether the guard drops a payload holding ``value``, as json.dumps writes it."""
    return matches_unsafe_escape(json.dumps({"a": value}))


class TestUnsafeEscapePattern:
    """Semantics of the PostgreSQL escape guard, without a database.

    The PostgreSQL-only suite pins the same behaviour end-to-end, but it skips
    unless ``XAGENT_TEST_POSTGRES_URL`` is set -- which is every ordinary local
    run and every leg of the main CI workflow, since those select
    ``-m "not postgresql"``. These run everywhere.
    """

    @pytest.mark.parametrize(
        "label,value",
        [
            ("nul escape", NUL),
            ("lone high surrogate", LONE_HIGH),
            ("lone low surrogate", LONE_LOW),
            ("lone surrogate after a pair", NON_BMP + LONE_LOW),
            ("lone surrogate before a pair", LONE_HIGH + NON_BMP),
            ("escape buried in text", "before" + NUL + "after"),
            # Text mimicking a high surrogate must not shield a real lone low.
            # Without the escaped-backslash normalization this reads as a valid
            # pair, slips past the guard and fails the whole request -- the
            # failure the guard exists to prevent, in its worst direction.
            ("real low behind a literal escape", BACKSLASH + "ud83d" + LONE_LOW),
        ],
    )
    def test_drops_payloads_postgresql_cannot_convert(self, label, value):
        """Each of these makes ``->>`` raise, so the row has to be dropped."""
        assert guard_would_drop(value) is True, label

    @pytest.mark.parametrize(
        "label,value",
        [
            ("plain text", "hello"),
            ("non-BMP character", NON_BMP),
            ("two non-BMP characters", NON_BMP + NON_BMP),
            ("non-BMP inside text", "before" + NON_BMP + "after"),
            # A doubled backslash in the JSON: text that merely looks like an
            # escape. ->> reads these fine, so dropping them would lose real
            # monitoring rows.
            ("literal backslash-u0000 text", BACKSLASH + "u0000"),
            ("literal backslash-ud83d text", BACKSLASH + "ud83d"),
        ],
    )
    def test_keeps_payloads_postgresql_can_convert(self, label, value):
        """Nothing here raises in PostgreSQL, so nothing here may be dropped."""
        assert guard_would_drop(value) is False, label

    @pytest.mark.parametrize(
        "label,payload_text,expected",
        [
            (
                "uppercase valid pair",
                '{"a": "' + BACKSLASH + "uD83D" + BACKSLASH + 'uDE00"}',
                False,
            ),
            ("uppercase lone high", '{"a": "' + BACKSLASH + 'uD800"}', True),
            ("uppercase lone low", '{"a": "' + BACKSLASH + 'uDC00"}', True),
        ],
    )
    def test_hex_case_is_tolerated(self, label, payload_text, expected):
        """Escapes from a non-Python writer may use uppercase hex.

        ``json.dumps`` only ever emits lowercase, so these payload texts are
        written out directly. The character classes in the pattern are built to
        accept either case; narrowing them would silently start dropping valid
        pairs and keeping fatal lone surrogates.
        """
        assert matches_unsafe_escape(payload_text) is expected, label


def test_admin_user_permissions():
    """测试管理员用户权限检查"""
    # 测试管理员用户
    admin_user = Mock()
    admin_user.is_admin = True

    from xagent.web.api.monitor import is_admin_user

    assert is_admin_user(admin_user) is True

    # 测试普通用户
    normal_user = Mock()
    normal_user.is_admin = False

    assert is_admin_user(normal_user) is False


class TestMonitorAPIUserIsolation:
    """测试监控 API 的用户隔离功能"""

    def test_user_query_filtering(self):
        """测试用户查询过滤逻辑"""
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"
        mock_session.bind = mock_engine

        # 模拟普通用户
        normal_user = Mock()
        normal_user.id = 123
        normal_user.is_admin = False

        # 模拟管理员用户
        admin_user = Mock()
        admin_user.id = 1
        admin_user.is_admin = True

        from xagent.web.api.monitor import is_admin_user

        # 测试权限检查
        assert is_admin_user(normal_user) is False
        assert is_admin_user(admin_user) is True


if __name__ == "__main__":
    pytest.main([__file__])
