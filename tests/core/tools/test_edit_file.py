"""
测试 edit_file 工具
"""

import os
import tempfile

import pytest

from xagent.core.tools.core.file_tool import (
    EditOperation,
    _delete_lines,
    _insert_lines,
    _replace_lines,
    edit_file,
    find_and_replace,
)


class TestEditFile:
    """测试 edit_file 工具"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        )
        self.test_content = """Line 1
Line 2
Line 3
Line 4
Line 5"""
        self.temp_file.write(self.test_content)
        self.temp_file.close()

    def teardown_method(self):
        """每个测试方法执行后的清理"""
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
        backup_file = self.temp_file.name + ".backup"
        if os.path.exists(backup_file):
            os.remove(backup_file)

    def test_edit_file_replace_by_line_number(self):
        """测试基于行号的替换操作"""
        operations = [
            {
                "operation_type": "replace",
                "line_number": 3,
                "content": "Modified Line Three",
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1
        assert "Modified Line Three" in result.preview

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Modified Line Three" in content
        assert "Line 3\n" not in content  # 原来的Line 3整行应该被完全替换
        assert content == "Line 1\nLine 2\nModified Line Three\nLine 4\nLine 5"

    def test_edit_file_replace_by_pattern(self):
        """测试基于模式匹配的替换操作"""
        operations = [
            {
                "operation_type": "replace",
                "pattern": r"Line 2",
                "content": "Replaced Line Two",
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Replaced Line Two" in content
        assert "Line 2\n" not in content  # 原来的Line 2整行应该被完全替换
        assert content == "Line 1\nReplaced Line Two\nLine 3\nLine 4\nLine 5"

    def test_edit_file_insert_by_line_number(self):
        """测试基于行号的插入操作"""
        operations = [
            {"operation_type": "insert", "line_number": 3, "content": "Inserted Line"}
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            lines = f.readlines()

        assert len(lines) == 6  # 原来是5行，插入1行
        assert "Inserted Line\n" in lines[2]  # 插在第3行位置

    def test_edit_file_insert_by_pattern(self):
        """测试基于模式匹配的插入操作"""
        operations = [
            {
                "operation_type": "insert",
                "pattern": r"Line 3",
                "content": "Inserted after Line 3",
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            lines = f.readlines()

        assert len(lines) == 6
        assert "Inserted after Line 3\n" in lines[3]  # 插在Line 3后面

    def test_edit_file_delete_by_line_number(self):
        """测试基于行号的删除操作"""
        operations = [{"operation_type": "delete", "line_number": 3}]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            lines = f.readlines()

        assert len(lines) == 4  # 原来是5行，删除1行
        assert "Line 3\n" not in lines

    def test_edit_file_delete_by_pattern(self):
        """测试基于模式匹配的删除操作"""
        operations = [{"operation_type": "delete", "pattern": r"Line 2"}]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Line 2" not in content

    def test_edit_file_multiple_operations(self):
        """测试多个操作"""
        operations = [
            {
                "operation_type": "replace",
                "line_number": 2,
                "content": "Modified Line 2",
            },
            {"operation_type": "insert", "line_number": 4, "content": "Inserted Line"},
            {
                "operation_type": "replace",
                "pattern": r"Line 5",
                "content": "Final Line",
            },
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 3

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Modified Line 2" in content
        assert "Inserted Line" in content
        assert "Final Line" in content

    def test_edit_file_with_backup(self):
        """测试创建备份文件"""
        operations = [
            {
                "operation_type": "replace",
                "line_number": 1,
                "content": "Modified first line",
            }
        ]

        result = edit_file(self.temp_file.name, operations, backup=True)

        assert result.success is True

        # 检查备份文件是否创建
        backup_file = self.temp_file.name + ".backup"
        assert os.path.exists(backup_file)

        with open(backup_file, "r") as f:
            backup_content = f.read()

        assert backup_content == self.test_content

        with open(self.temp_file.name, "r") as f:
            current_content = f.read()

        assert current_content != backup_content

    def test_edit_file_with_edit_operation_objects(self):
        """测试使用 EditOperation 对象"""
        operations = [
            EditOperation(
                operation_type="replace",
                line_number=2,
                content="Using EditOperation object",
            )
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert "Using EditOperation object" in result.preview

    def test_edit_file_file_not_found(self):
        """测试文件不存在的情况"""
        with pytest.raises(FileNotFoundError):
            edit_file("/nonexistent/file.txt", [])

    def test_edit_file_invalid_line_number(self):
        """测试无效行号"""
        operations = [
            {"operation_type": "replace", "line_number": 999, "content": "Content"}
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is False
        assert "out of range" in result.message

    def test_edit_file_no_matching_pattern(self):
        """测试没有匹配的模式"""
        operations = [
            {
                "operation_type": "replace",
                "pattern": r"NonExistentPattern",
                "content": "Content",
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is False
        assert "No lines found matching pattern" in result.message

    def test_edit_file_field_mapping_alternative_names(self):
        """测试使用替代字段名称的操作映射 (type->operation_type, target->pattern, replacement->content)"""
        operations = [
            {
                "type": "replace",
                "target": r"Line 2",
                "replacement": "Replaced using alternative field names",
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Replaced using alternative field names" in content
        assert "Line 2\n" not in content
        assert (
            content
            == "Line 1\nReplaced using alternative field names\nLine 3\nLine 4\nLine 5"
        )

    def test_edit_file_field_mapping_mixed_names(self):
        """测试混合使用标准字段名和替代字段名"""
        operations = [
            {
                "type": "replace",  # 替代字段名
                "line_number": 2,
                "content": "Mixed field names 1",
            },
            {
                "operation_type": "replace",  # 标准字段名
                "pattern": r"Line 4",
                "content": "Mixed field names 2",
            },
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 2

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Mixed field names 1" in content
        assert "Mixed field names 2" in content
        assert "Line 2\n" not in content
        assert "Line 4\n" not in content

    def test_edit_file_field_mapping_insert_with_alternative_names(self):
        """测试使用替代字段名的插入操作"""
        operations = [
            {
                "type": "insert",
                "target": r"Line 3",
                "replacement": "Inserted using alternative field names",
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            lines = f.readlines()

        assert len(lines) == 6  # 原来是5行，插入1行
        assert "Inserted using alternative field names\n" in lines[3]  # 插在Line 3后面

    def test_edit_file_field_mapping_delete_with_alternative_names(self):
        """测试使用替代字段名的删除操作"""
        operations = [{"type": "delete", "target": r"Line 2"}]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Line 2" not in content

    def test_edit_file_field_mapping_preserves_standard_names(self):
        """测试字段映射不会影响标准字段名的使用"""
        operations = [
            {
                "operation_type": "replace",
                "pattern": r"Line 3",
                "content": "Using standard field names",
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Using standard field names" in content
        assert "Line 3\n" not in content


class TestFindAndReplace:
    """测试 find_and_replace 便捷函数"""

    def setup_method(self):
        """设置"""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        )
        self.test_content = """Hello World
This is a test
Hello Universe
Goodbye World"""
        self.temp_file.write(self.test_content)
        self.temp_file.close()

    def teardown_method(self):
        """清理"""
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
        backup_file = self.temp_file.name + ".backup"
        if os.path.exists(backup_file):
            os.remove(backup_file)

    def test_find_and_replace_regex(self):
        """测试正则表达式替换"""
        result = find_and_replace(
            self.temp_file.name, r"Hello\s+\w+", "Hi Everyone", use_regex=True
        )

        assert result.success is True

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Hi Everyone" in content
        assert "Hello World" not in content
        assert "Hello Universe" not in content

    def test_find_and_replace_literal(self):
        """测试字面量替换"""
        result = find_and_replace(self.temp_file.name, "Hello", "Hi", use_regex=False)

        assert result.success is True

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Hi World" in content
        assert "Hi Universe" in content
        assert "Hello World" not in content
        assert "Hello Universe" not in content

    def test_find_and_replace_case_insensitive(self):
        """测试不区分大小写替换"""
        # 先创建包含大小写的内容
        with open(self.temp_file.name, "w") as f:
            f.write("hello HELLO Hello")

        result = find_and_replace(
            self.temp_file.name, "hello", "hi", case_sensitive=False
        )

        assert result.success is True

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert content == "hi hi hi"

    def test_find_and_replace_no_match(self):
        """测试没有匹配的情况"""
        result = find_and_replace(
            self.temp_file.name, "NonExistentPattern", "Replacement"
        )

        assert result.success is False
        assert "No matches found for pattern" in result.message


class TestHelperFunctions:
    """测试辅助函数"""

    def test_replace_lines_by_number(self):
        """测试 _replace_lines 按行号替换"""
        lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
        operation = EditOperation(
            operation_type="replace", line_number=2, content="New Line 2"
        )

        result = _replace_lines(lines, operation)

        assert result.success is True
        assert lines[1] == "New Line 2\n"

    def test_replace_lines_by_pattern(self):
        """测试 _replace_lines 按模式替换"""
        lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
        operation = EditOperation(
            operation_type="replace", pattern=r"Line 2", content="New Line 2"
        )

        result = _replace_lines(lines, operation)

        assert result.success is True
        assert lines[1] == "New Line 2\n"

    def test_insert_lines_by_number(self):
        """测试 _insert_lines 按行号插入"""
        lines = ["Line 1\n", "Line 3\n"]
        operation = EditOperation(
            operation_type="insert", line_number=2, content="Line 2"
        )

        result = _insert_lines(lines, operation)

        assert result.success is True
        assert len(lines) == 3
        assert lines[1] == "Line 2\n"

    def test_delete_lines_by_number(self):
        """测试 _delete_lines 按行号删除"""
        lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
        operation = EditOperation(operation_type="delete", line_number=2)

        result = _delete_lines(lines, operation)

        assert result.success is True
        assert len(lines) == 2
        assert lines == ["Line 1\n", "Line 3\n"]


class TestEditFileFieldMapping:
    """专门测试 edit_file 字段映射功能的测试类"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False
        )
        self.test_content = """<html>
<head>
    <title>Original Title</title>
</head>
<body>
    <h1>Hello World</h1>
</body>
</html>"""
        self.temp_file.write(self.test_content)
        self.temp_file.close()

    def teardown_method(self):
        """每个测试方法执行后的清理"""
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
        backup_file = self.temp_file.name + ".backup"
        if os.path.exists(backup_file):
            os.remove(backup_file)

    def test_field_mapping_html_style_replacement(self):
        """测试典型的HTML风格替换操作，模拟原始错误场景"""
        operations = [
            {
                "type": "replace",
                "target": r"<head>",
                "replacement": '<head>\n <meta charset="UTF-8">\n <meta name="viewport" content="width=device-width, initial-scale=1.0">\n <title>Updated Title</title>\n',
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert '<meta charset="UTF-8">' in content
        assert '<meta name="viewport"' in content
        assert "Updated Title" in content

    def test_field_mapping_all_alternative_names(self):
        """测试所有替代字段名的映射"""
        operations = [
            {
                "type": "replace",
                "target": r"Hello World",
                "replacement": "Replaced Hello World",
            },
            {
                "type": "insert",
                "target": r"Hello World",
                "replacement": "    <!-- Inserted comment -->\n",
            },
            {
                "type": "delete",
                "target": r"</html>",
            },
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 3

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Replaced Hello World" in content
        assert "<!-- Inserted comment -->" in content
        assert "</html>" not in content

    def test_field_mapping_preserves_original_fields(self):
        """测试原始字段名不受映射影响"""
        operations = [
            {
                "operation_type": "replace",
                "pattern": r"Hello World",
                "content": "Using original fields",
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Using original fields" in content

    def test_field_mapping_mixed_standard_and_alternative(self):
        """测试在同一个操作中混合使用标准字段名和替代字段名"""
        operations = [
            {
                "type": "replace",  # 替代字段名
                "pattern": r"Hello World",  # 标准字段名
                "replacement": "Mixed naming scheme",  # 替代字段名
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Mixed naming scheme" in content

    def test_field_mapping_with_line_number(self):
        """测试带行号的字段映射"""
        operations = [
            {
                "type": "replace",
                "line_number": 5,
                "replacement": "    <h1>Updated Heading</h1>",
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            lines = f.readlines()

        assert "Updated Heading" in lines[4]

    def test_field_mapping_error_handling(self):
        """测试字段映射的错误处理"""
        operations = [
            {
                "type": "replace",
                "target": r"NonExistentPattern",
                "replacement": "Some content",
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is False
        assert "No lines found matching pattern" in result.message

    def test_field_mapping_multiple_operations_different_naming(self):
        """测试多个操作使用不同命名约定"""
        operations = [
            {
                "type": "replace",
                "target": r"Original Title",
                "replacement": "New Title",
            },
            {
                "operation_type": "replace",
                "pattern": r"Hello World",
                "content": "Updated Greeting",
            },
            {
                "type": "insert",
                "line_number": 7,
                "replacement": "    <p>New paragraph</p>\n",
            },
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 3

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "New Title" in content
        assert "Updated Greeting" in content
        assert "<p>New paragraph</p>" in content

    def test_field_mapping_empty_operations(self):
        """测试空操作列表的字段映射"""
        operations = []

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 0

    def test_field_mapping_unknown_fields_preserved(self):
        """测试未知字段会被保留"""
        operations = [
            {
                "type": "replace",
                "target": r"Hello World",
                "replacement": "Updated Hello",
                "unknown_field": "should_be_preserved",
                "another_unknown": 123,
            }
        ]

        result = edit_file(self.temp_file.name, operations)

        assert result.success is True
        assert result.lines_changed == 1

        with open(self.temp_file.name, "r") as f:
            content = f.read()

        assert "Updated Hello" in content
