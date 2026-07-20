"""
Unit tests for logo_overlay tool functionality.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image

from src.xagent.core.tools.adapters.vibe.logo_overlay import (
    LogoOverlayArgs,
    LogoOverlayResult,
    LogoOverlayTool,
    create_logo_overlay_tool,
    get_logo_overlay_tool,
)
from src.xagent.core.workspace import TaskWorkspace
from xagent.core.tools.core.logo_overlay import LogoOverlayCore


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test"""
    # Simple cleanup - remove any temp directories that might exist
    temp_dirs = [
        Path("uploads/temp"),
        Path("workspaces"),
        Path("temp"),
    ]

    yield

    # Clean up after test
    for temp_dir in temp_dirs:
        try:
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
        except (OSError, PermissionError):
            pass


class TestLogoOverlayArgs:
    """Test cases for LogoOverlayArgs validation."""

    def test_valid_args(self) -> None:
        """Test valid logo overlay arguments."""
        args = LogoOverlayArgs(
            base_image_uri="https://example.com/base.jpg",
            logo_image_uri="https://example.com/logo.png",
            position="bottom-right",
            size_ratio=0.2,
            opacity=1.0,
            padding=20,
        )
        assert args.base_image_uri == "https://example.com/base.jpg"
        assert args.logo_image_uri == "https://example.com/logo.png"
        assert args.position == "bottom-right"
        assert args.size_ratio == 0.2
        assert args.opacity == 1.0
        assert args.padding == 20

    def test_valid_positions(self) -> None:
        """Test all valid position values."""
        positions = ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
        for position in positions:
            args = LogoOverlayArgs(
                base_image_uri="base.jpg",
                logo_image_uri="logo.png",
                position=position,
            )
            assert args.position == position

    def test_size_ratio_validation(self) -> None:
        """Test size_ratio validation constraints."""
        # Valid values
        LogoOverlayArgs(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
            size_ratio=0.1,
        )
        LogoOverlayArgs(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
            size_ratio=0.5,
        )

        # Invalid values
        with pytest.raises(ValueError):
            LogoOverlayArgs(
                base_image_uri="base.jpg",
                logo_image_uri="logo.png",
                size_ratio=0.05,  # Too small
            )
        with pytest.raises(ValueError):
            LogoOverlayArgs(
                base_image_uri="base.jpg",
                logo_image_uri="logo.png",
                size_ratio=0.6,  # Too large
            )

    def test_opacity_validation(self) -> None:
        """Test opacity validation constraints."""
        # Valid values
        LogoOverlayArgs(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
            opacity=0.0,
        )
        LogoOverlayArgs(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
            opacity=1.0,
        )

        # Invalid values
        with pytest.raises(ValueError):
            LogoOverlayArgs(
                base_image_uri="base.jpg",
                logo_image_uri="logo.png",
                opacity=-0.1,  # Too small
            )
        with pytest.raises(ValueError):
            LogoOverlayArgs(
                base_image_uri="base.jpg",
                logo_image_uri="logo.png",
                opacity=1.1,  # Too large
            )

    def test_padding_validation(self) -> None:
        """Test padding validation constraints."""
        # Valid values
        LogoOverlayArgs(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
            padding=0,
        )
        LogoOverlayArgs(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
            padding=100,
        )

        # Invalid values
        with pytest.raises(ValueError):
            LogoOverlayArgs(
                base_image_uri="base.jpg",
                logo_image_uri="logo.png",
                padding=-1,  # Too small
            )
        with pytest.raises(ValueError):
            LogoOverlayArgs(
                base_image_uri="base.jpg",
                logo_image_uri="logo.png",
                padding=101,  # Too large
            )

    def test_workspace_id_optional(self) -> None:
        """Test workspace_id is optional."""
        args = LogoOverlayArgs(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
            workspace_id=None,
        )
        assert args.workspace_id is None

        args = LogoOverlayArgs(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
            workspace_id="test_workspace",
        )
        assert args.workspace_id == "test_workspace"


class TestLogoOverlayTool:
    """Test cases for LogoOverlayTool functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        from contextlib import contextmanager

        self.mock_workspace = Mock(spec=TaskWorkspace)
        self.mock_workspace.temp_dir = Path("/tmp/workspace/temp")
        self.mock_workspace.output_dir = Path("/tmp/workspace/output")

        # Mock auto_register_files to return a proper context manager
        @contextmanager
        def auto_register_files():
            yield self.mock_workspace

        self.mock_workspace.auto_register_files = auto_register_files
        # Mock get_file_id_from_path to return a valid file_id
        self.mock_workspace.get_file_id_from_path = Mock(return_value="test-file-id")
        # Mock resolve_path_with_search
        self.mock_workspace.resolve_path_with_search = Mock(
            return_value=Path("/tmp/test.jpg")
        )

        self.tool = LogoOverlayTool(workspace=self.mock_workspace)
        self.core = LogoOverlayCore()

    def test_tool_properties(self) -> None:
        """Test tool basic properties."""
        assert self.tool.name == "logo_overlay"
        assert "logo" in self.tool.description.lower()
        assert "overlay" in self.tool.description.lower()
        assert "exact logo asset" in self.tool.description.lower()
        assert "instead of asking an image model" in self.tool.description.lower()
        assert "image" in self.tool.tags
        assert "logo" in self.tool.tags
        assert self.tool.args_type() == LogoOverlayArgs
        assert self.tool.return_type() == LogoOverlayResult

    def test_run_json_sync_not_implemented(self) -> None:
        """Test sync execution is not implemented."""
        with pytest.raises(NotImplementedError):
            self.tool.run_json_sync({})

    @pytest.mark.asyncio
    async def test_run_json_async_success(self) -> None:
        """Test successful async execution."""
        # Mock the _overlay_logo method
        mock_result = {
            "success": True,
            "output_path": "/tmp/workspace/temp/output.png",
            "message": "Logo overlay completed successfully",
            "error": None,
        }
        # self.tool._overlay_logo = AsyncMock(return_value=mock_result)

        args = {
            "base_image_uri": "base.jpg",
            "logo_image_uri": "logo.png",
            "position": "bottom-right",
            "size_ratio": 0.2,
            "opacity": 1.0,
            "padding": 20,
        }

        with patch(
            "src.xagent.core.tools.adapters.vibe.logo_overlay.LogoOverlayCore.overlay_logo",
            return_value=mock_result,
        ):
            result = await self.tool.run_json_async(args)

        # Check that result is a dict with expected structure
        assert isinstance(result, dict)
        assert "success" in result
        assert "output_path" in result
        assert "message" in result
        assert "error" in result
        assert result["success"] is True
        assert result["output_path"] == "/tmp/workspace/temp/output.png"
        assert result["message"] == "Logo overlay completed successfully"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_overlay_logo_success(self) -> None:
        """Test successful logo overlay process."""
        # Mock image loading
        mock_base_image = Mock(spec=Image.Image)
        mock_base_image.size = (800, 600)
        mock_logo_image = Mock(spec=Image.Image)
        mock_logo_image.size = (200, 100)
        mock_logo_image.width = 200
        mock_logo_image.height = 100

        # Mock image processing
        mock_result_image = Mock(spec=Image.Image)
        mock_result_image.mode = "RGB"

        core = LogoOverlayCore(output_directory="/tmp/workspace/temp")
        core._load_image = AsyncMock(side_effect=[mock_base_image, mock_logo_image])
        core._process_logo_overlay = Mock(return_value=mock_result_image)
        core._save_result_image = Mock(return_value="/tmp/workspace/temp/output.jpg")
        result = await core.overlay_logo(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
            position="bottom-right",
            size_ratio=0.2,
            opacity=1.0,
            padding=20,
        )

        assert result["success"] is True
        assert result["output_path"] == "/tmp/workspace/temp/output.jpg"
        assert result["message"] == "Logo overlay completed successfully"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_overlay_logo_base_image_load_failure(self) -> None:
        """Test logo overlay with base image load failure."""

        self.core._load_image = AsyncMock(return_value=None)

        result = await self.core.overlay_logo(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
        )

        assert result["success"] is False
        assert result["output_path"] == ""
        assert result["message"] == "Failed to load base image"
        assert result["error"] == "Base image could not be loaded"

    @pytest.mark.asyncio
    async def test_overlay_logo_logo_image_load_failure(self) -> None:
        """Test logo overlay with logo image load failure."""
        mock_base_image = Mock(spec=Image.Image)
        core = LogoOverlayCore()
        core._load_image = AsyncMock(side_effect=[mock_base_image, None])

        result = await core.overlay_logo(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
        )

        assert result["success"] is False
        assert result["output_path"] == ""
        assert result["message"] == "Failed to load logo image"
        assert result["error"] == "Logo image could not be loaded"

    @pytest.mark.asyncio
    async def test_overlay_logo_exception_handling(self) -> None:
        """Test logo overlay with general exception."""
        self.core._load_image = AsyncMock(side_effect=Exception("Test error"))

        result = await self.core.overlay_logo(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
        )

        assert result["success"] is False
        assert result["output_path"] == ""
        assert result["message"] == "Logo overlay failed"
        assert result["error"] == "Test error"

    @pytest.mark.asyncio
    async def test_load_image_remote_url(self) -> None:
        """Test loading image from remote URL."""
        # Create a mock image
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (800, 600)
        mock_image.mode = "RGB"

        # Mock httpx client
        mock_response = Mock()
        mock_response.content = b"fake_image_data"
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            with patch("PIL.Image.open") as mock_open:
                mock_image_instance = Mock()
                mock_image_instance.mode = "RGB"
                mock_open.return_value = mock_image_instance

                result = await self.core._load_image(
                    "https://example.com/image.jpg", "test image"
                )

                assert result is not None
                mock_client.get.assert_called_once_with("https://example.com/image.jpg")

    @pytest.mark.asyncio
    async def test_load_image_local_path(self) -> None:
        """Test loading image from local path."""
        with (
            patch("os.path.exists", return_value=True),
            patch("PIL.Image.open") as mock_open,
        ):
            mock_image_instance = Mock()
            mock_image_instance.mode = "RGB"
            mock_open.return_value = mock_image_instance

            # Mock the workspace path resolution
            self.mock_workspace.resolve_path_with_search.return_value = (
                "/local/path/image.jpg"
            )

            result = await self.core._load_image("/local/path/image.jpg", "test image")

            assert result is not None
            mock_open.assert_called_once_with("/local/path/image.jpg")

    @pytest.mark.asyncio
    async def test_load_image_local_path_not_found(self) -> None:
        """Test loading image from non-existent local path."""
        with patch("os.path.exists", return_value=False):
            result = await self.core._load_image("/nonexistent/path.jpg", "test image")

            assert result is None

    @pytest.mark.asyncio
    async def test_load_image_exception_handling(self) -> None:
        """Test image loading with exception."""
        with (
            patch("os.path.exists", return_value=True),
            patch("PIL.Image.open", side_effect=Exception("Load error")),
        ):
            result = await self.core._load_image("/local/path/image.jpg", "test image")

            assert result is None

    @pytest.mark.asyncio
    async def test_process_logo_overlay(self) -> None:
        """Test logo overlay processing."""
        # Create mock images
        mock_base_image = Mock(spec=Image.Image)
        mock_base_image.size = (800, 600)
        mock_base_image.copy.return_value = Mock(spec=Image.Image)

        mock_logo_image = Mock(spec=Image.Image)
        mock_logo_image.width = 200
        mock_logo_image.height = 100

        result = self.core._process_logo_overlay(
            mock_base_image, mock_logo_image, "bottom-right", 0.2, 1.0, 20
        )

        assert result is not None
        # Verify logo was resized and positioned correctly
        mock_base_image.copy.assert_called_once()

    def test_apply_opacity(self) -> None:
        """Test opacity application."""
        # Create a mock RGBA image
        mock_image = Mock(spec=Image.Image)
        mock_image.mode = "RGBA"
        mock_r = Mock()
        mock_g = Mock()
        mock_b = Mock()
        mock_a = Mock()
        mock_image.split.return_value = (mock_r, mock_g, mock_b, mock_a)

        # Mock alpha channel transformation
        mock_a.point.return_value = mock_a

        with patch("PIL.Image.merge") as mock_merge:
            mock_merge.return_value = mock_image

            result = self.core._apply_opacity(mock_image, 0.5)

            assert result is not None
            mock_a.point.assert_called_once()
            mock_merge.assert_called_once_with("RGBA", (mock_r, mock_g, mock_b, mock_a))

    def test_apply_opacity_non_rgba(self) -> None:
        """Test opacity application on non-RGBA image."""
        # Create a mock non-RGBA image
        mock_image = Mock(spec=Image.Image)
        mock_image.mode = "RGB"
        mock_image.convert.return_value = mock_image

        mock_image.split.return_value = (Mock(), Mock(), Mock(), Mock())

        with patch("PIL.Image.merge") as mock_merge:
            mock_merge.return_value = mock_image

            result = self.core._apply_opacity(mock_image, 0.5)

            assert result is not None
            mock_image.convert.assert_called_once_with("RGBA")

    def test_calculate_position(self) -> None:
        """Test position calculation for all positions."""
        base_width, base_height = 800, 600
        logo_width, logo_height = 160, 80
        padding = 20

        # Test all positions
        positions = [
            ("top-left", 20, 20),
            ("top-right", 620, 20),
            ("bottom-left", 20, 500),
            ("bottom-right", 620, 500),
            ("center", 320, 260),
        ]

        for position, expected_x, expected_y in positions:
            x, y = self.core._calculate_position(
                base_width, base_height, logo_width, logo_height, position, padding
            )
            assert x == expected_x
            assert y == expected_y

    def test_calculate_position_invalid(self) -> None:
        """Test position calculation with invalid position (defaults to bottom-right)."""
        base_width, base_height = 800, 600
        logo_width, logo_height = 160, 80
        padding = 20

        x, y = self.core._calculate_position(
            base_width, base_height, logo_width, logo_height, "invalid", padding
        )

        # Should default to bottom-right
        expected_x = base_width - logo_width - padding
        expected_y = base_height - logo_height - padding
        assert x == expected_x
        assert y == expected_y

    @pytest.mark.asyncio
    async def test_save_result_image_with_workspace(self) -> None:
        """Test saving result image with workspace."""
        mock_image = Mock(spec=Image.Image)
        mock_image.mode = "RGB"

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "test1234"
            with patch.object(self.tool, "_workspace") as mock_workspace:
                mock_workspace.output_dir = Path("/tmp/workspace/output")

                result = self.core._save_result_image(mock_image)

                assert result.endswith("logo_overlay_test1234.jpg")
                mock_image.convert.assert_called_once_with("RGB")

    @pytest.mark.asyncio
    async def test_save_result_image_with_workspace_id(self) -> None:
        """Test saving result image with workspace_id."""
        mock_image = Mock(spec=Image.Image)
        mock_image.mode = "RGBA"

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "test5678"
            result = self.core._save_result_image(mock_image)

            assert result.endswith("logo_overlay_test5678.png")

    @pytest.mark.asyncio
    async def test_save_result_image_custom_filename(self) -> None:
        """Test saving result image with custom filename."""
        mock_image = Mock(spec=Image.Image)
        mock_image.mode = "RGB"

        # Mock the Path construction to return a proper path
        with patch(
            "pathlib.Path.__truediv__",
            return_value=Path("/tmp/workspace/output/my_custom_logo.jpg"),
        ):
            result = self.core._save_result_image(
                mock_image, output_filename="my_custom_logo"
            )

            assert result.endswith("my_custom_logo.jpg")

    @pytest.mark.asyncio
    async def test_save_result_image_filename_sanitization(self) -> None:
        """Test filename sanitization for custom filenames."""
        mock_image = Mock(spec=Image.Image)
        mock_image.mode = "RGB"

        # Test with problematic characters
        with patch(
            "pathlib.Path.__truediv__",
            return_value=Path("/tmp/workspace/output/My_Logo.jpg"),
        ):
            result = self.core._save_result_image(
                mock_image, output_filename="My Logo!!! @#$%"
            )

            assert "My_Logo" in result
            assert result.endswith(".jpg")


class TestLogoOverlayToolIntegration:
    """Integration tests for logo overlay tool."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_workspace = Mock(spec=TaskWorkspace)
        self.mock_workspace.temp_dir = Path("/tmp/workspace/temp")
        self.mock_workspace.output_dir = Path("/tmp/workspace/output")

    def test_get_logo_overlay_tool_without_workspace(self) -> None:
        """Test factory function without workspace."""
        tool = get_logo_overlay_tool()
        assert isinstance(tool, LogoOverlayTool)
        assert tool._workspace is None

    def test_get_logo_overlay_tool_with_workspace(self) -> None:
        """Test factory function with workspace."""
        tool = get_logo_overlay_tool(workspace=self.mock_workspace)
        assert isinstance(tool, LogoOverlayTool)
        assert tool._workspace == self.mock_workspace

    def test_create_logo_overlay_tool(self) -> None:
        """Test create_logo_overlay_tool function."""
        tool = create_logo_overlay_tool(self.mock_workspace)
        assert isinstance(tool, LogoOverlayTool)
        assert tool._workspace == self.mock_workspace

    @pytest.mark.asyncio
    async def test_full_workflow_with_real_images(self) -> None:
        """Test full workflow with mock real images."""
        # Create mock images
        mock_base_image = Mock(spec=Image.Image)
        mock_base_image.size = (800, 600)
        mock_base_image.mode = "RGB"
        mock_base_image.copy.return_value = Mock(spec=Image.Image)

        mock_logo_image = Mock(spec=Image.Image)
        mock_logo_image.width = 100
        mock_logo_image.height = 50
        mock_logo_image.resize.return_value = mock_logo_image

        core = LogoOverlayCore()

        # Mock all the methods
        core._load_image = AsyncMock(side_effect=[mock_base_image, mock_logo_image])
        core._process_logo_overlay = AsyncMock(return_value=mock_base_image)
        core._save_result_image = Mock(return_value="/tmp/workspace/temp/output.jpg")

        result = await core.overlay_logo(
            base_image_uri="base.jpg",
            logo_image_uri="logo.png",
            position="bottom-right",
            size_ratio=0.2,
            opacity=1.0,
            padding=20,
        )

        assert result["success"] is True
        assert result["output_path"] == "/tmp/workspace/temp/output.jpg"

    @pytest.mark.asyncio
    async def test_proxy_configuration(self) -> None:
        """Test proxy configuration for remote image loading."""
        core = LogoOverlayCore()

        mock_image = Mock(spec=Image.Image)
        mock_image.mode = "RGB"

        with (
            patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy.example.com:8080"}),
            patch("httpx.AsyncClient") as mock_client_class,
            patch("PIL.Image.open") as mock_open,
        ):
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response = Mock()
            mock_response.content = b"fake_image_data"
            mock_response.raise_for_status = Mock()
            mock_client.get.return_value = mock_response

            mock_image_instance = Mock()
            mock_image_instance.mode = "RGB"
            mock_open.return_value = mock_image_instance

            result = await core._load_image(
                "https://example.com/image.jpg", "test image"
            )

            assert result is not None
            # Verify proxy was configured
            call_kwargs = mock_client_class.call_args[1]
            assert "proxy" in call_kwargs
            assert call_kwargs["proxy"] == "http://proxy.example.com:8080"
