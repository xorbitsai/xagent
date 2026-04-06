import os
import tempfile

import pytest

from xagent.core.tools.adapters.vibe.file_tool import (
    FILE_TOOLS,
    append_file,
    create_directory,
    delete_file,
    file_exists,
    get_file_info,
    list_files,
    read_csv_file,
    read_file,
    read_json_file,
    write_csv_file,
    write_file,
    write_json_file,
)


def test_basic_file_operations():
    """测试基本文件操作"""
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        temp_file = f.name
        f.write("Hello, World!")

    try:
        # 测试读取文件
        content = read_file(temp_file)
        assert content == "Hello, World!"

        # 测试文件存在检查
        assert file_exists(temp_file)

        # 测试获取文件信息
        info = get_file_info(temp_file)
        assert info.name.endswith(".txt")
        assert info.is_file
        assert not info.is_dir

        # 测试写入文件
        write_file(temp_file, "New content")
        content = read_file(temp_file)
        assert content == "New content"

        # 测试追加文件
        append_file(temp_file, " Appended content")
        content = read_file(temp_file)
        assert content == "New content Appended content"

    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_directory_operations():
    """测试目录操作"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 测试创建目录
        new_dir = os.path.join(temp_dir, "test_dir", "sub_dir")
        create_directory(new_dir)
        assert os.path.exists(new_dir)

        # 测试列出文件
        files = list_files(temp_dir)
        assert files.total_count > 0
        assert any(f.name == "test_dir" for f in files.files)

        # 测试递归列出文件
        recursive_files = list_files(temp_dir, recursive=True)
        assert recursive_files.total_count >= 2  # 至少包含 test_dir 和 sub_dir


def test_json_operations():
    """测试JSON文件操作"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_file = f.name

    try:
        # 测试写入JSON文件
        test_data = {"name": "Test", "age": 25, "hobbies": ["reading", "coding"]}
        write_json_file(temp_file, test_data)

        # 测试读取JSON文件
        loaded_data = read_json_file(temp_file)
        assert loaded_data == test_data

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_csv_operations():
    """测试CSV文件操作"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        temp_file = f.name

    try:
        # 测试写入CSV文件
        test_data = [
            {"name": "Alice", "age": "25", "city": "New York"},
            {"name": "Bob", "age": "30", "city": "San Francisco"},
        ]
        write_csv_file(temp_file, test_data)

        # 测试读取CSV文件
        loaded_data = read_csv_file(temp_file)
        assert len(loaded_data) == 2
        assert loaded_data[0]["name"] == "Alice"
        assert loaded_data[1]["name"] == "Bob"

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_error_handling():
    """测试错误处理"""
    # 测试读取不存在的文件
    with pytest.raises(FileNotFoundError):
        read_file("/non/existent/file.txt")

    # 测试检查不存在的文件
    assert not file_exists("/non/existent/file.txt")

    # 测试获取不存在文件的信息
    with pytest.raises(FileNotFoundError):
        get_file_info("/non/existent/file.txt")


def test_file_tools_integration():
    """测试FileTool集成"""
    # 验证所有工具都能正确导入和实例化（包含新增的edit_file和find_and_replace）
    assert len(FILE_TOOLS) == 14

    # 验证每个工具都有正确的属性
    for tool in FILE_TOOLS:
        assert hasattr(tool, "metadata")
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert callable(tool.run_json_sync)
        assert callable(tool.run_json_async)

    # 验证工具名称唯一性
    tool_names = [tool.name for tool in FILE_TOOLS]
    assert len(tool_names) == len(set(tool_names)), "工具名称应该唯一"

    # 验证新添加的工具存在
    assert "edit_file" in tool_names
    assert "find_and_replace" in tool_names


def test_specific_tool_functionality():
    """测试特定工具的功能"""
    # 测试read_file_tool
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        temp_file = f.name
        f.write("Test content")

    try:
        # 使用工具实例测试
        read_tool = next(t for t in FILE_TOOLS if t.name == "read_file")
        result = read_tool.run_json_sync({"file_path": temp_file})
        assert "Test content" in str(result)

        # 测试write_file_tool
        write_tool = next(t for t in FILE_TOOLS if t.name == "write_file")
        write_tool.run_json_sync({"file_path": temp_file, "content": "New content"})

        # 验证写入成功
        result = read_tool.run_json_sync({"file_path": temp_file})
        assert "New content" in str(result)

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_image_file_info():
    """Test image file information retrieval"""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not installed")

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as f:
        temp_file = f.name

    try:
        # Create a test image (800x600, RGB)
        test_image = Image.new("RGB", (800, 600), color="red")
        test_image.save(temp_file)

        # Get file information
        info = get_file_info(temp_file)

        # Verify basic information
        assert info.name.endswith(".png")
        assert info.is_file
        assert not info.is_dir
        assert info.size > 0

        # Verify image metadata
        assert info.image_width == 800, f"Expected width 800, got {info.image_width}"
        assert info.image_height == 600, f"Expected height 600, got {info.image_height}"
        assert info.image_format == "PNG", (
            f"Expected format PNG, got {info.image_format}"
        )
        assert info.image_mode == "RGB", f"Expected mode RGB, got {info.image_mode}"

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_non_image_file_info():
    """Test non-image file information retrieval (should not include image metadata)"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        temp_file = f.name
        f.write("Test content")

    try:
        # Get file information
        info = get_file_info(temp_file)

        # Verify basic information
        assert info.name.endswith(".txt")
        assert info.is_file
        assert not info.is_dir

        # Verify image metadata is None
        assert info.image_width is None
        assert info.image_height is None
        assert info.image_format is None
        assert info.image_mode is None

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_image_file_info_without_pil():
    """Test that get_file_info handles PIL unavailability gracefully."""
    from unittest.mock import patch

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as f:
        temp_file = f.name
        f.write(b"fake png data")

    try:
        with patch("xagent.core.tools.core.file_tool.PIL_AVAILABLE", False):
            info = get_file_info(temp_file)

            assert info.is_file
            # When PIL is not available, image metadata should all be None
            assert info.image_width is None
            assert info.image_height is None
            assert info.image_format is None
            assert info.image_mode is None

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)
