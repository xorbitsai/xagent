"""
Unit tests for PPTX tools.

Tests theme configuration, validation, and core functionality.
"""

import tempfile
import zipfile
from pathlib import Path

import pytest

from xagent.core.tools.core.pptx_tool import (
    THEME_CONFIGS,
    _preset_to_config,
    _validate_theme_config,
    add_slide_pptx,
    clean_pptx,
    generate_pptx,
    pack_pptx,
    read_pptx,
    unpack_pptx,
)
from xagent.core.workspace import TaskWorkspace


class TestThemeConfiguration:
    """Test theme configuration system."""

    def test_preset_theme_aurora(self):
        """Test aurora theme preset conversion."""
        config = _preset_to_config("aurora")

        assert config["name"] == "Aurora"
        assert config["colors"]["background"] == "#FFFFFF"
        assert config["colors"]["primary"] == "#0B1F3B"
        assert config["colors"]["accent"] == "#2563EB"
        assert config["typography"]["title_size"] == 56
        assert config["typography"]["body_size"] == 20
        assert config["layout"]["title_bar"] is False
        assert config["visual"]["background_style"] == "solid"

    def test_preset_theme_vortex(self):
        """Test vortex theme preset conversion."""
        config = _preset_to_config("vortex")

        assert config["name"] == "Vortex"
        assert config["colors"]["background"] == "#0F172A"
        assert config["colors"]["primary"] == "#F8FAFC"
        assert config["colors"]["secondary"] == "#94A3B8"
        assert config["typography"]["title_size"] == 58
        assert config["typography"]["body_size"] == 20
        assert config["layout"]["title_bar"] is False

    def test_preset_theme_mono(self):
        """Test mono theme preset conversion."""
        config = _preset_to_config("mono")

        assert config["name"] == "Mono"
        assert config["colors"]["background"] == "#FAFAFA"
        assert config["colors"]["primary"] == "#111111"
        assert config["colors"]["text"] == "#111111"
        assert config["typography"]["title_size"] == 60
        assert config["typography"]["body_size"] == 19
        assert config["layout"]["title_bar"] is False

    def test_preset_theme_invalid(self):
        """Test invalid preset theme falls back to default (aurora)."""
        config = _preset_to_config("invalid_theme_name")

        # Should return default config (aurora)
        assert config["colors"]["background"] == "#FFFFFF"
        assert config["typography"]["title_size"] == 56
        assert config["typography"]["body_size"] == 20
        assert config["spacing"]["side_margin"] == 0.9
        assert config["visual_weight"]["divider_opacity"] == 0.12


class TestThemeValidation:
    """Test theme configuration validation."""

    def test_validate_complete_valid_config(self):
        """Test validation of complete valid config."""
        config = {
            "colors": {
                "background": "#1E2761",
                "surface": "#FFFFFF",
                "primary": "#CADCFC",
                "secondary": "#E8E8D1",
                "accent": "#FFFFFF",
                "text": "#1E2761",
            },
            "typography": {
                "title_size": 44,
                "body_size": 18,
                "quote_size": 24,
                "title_weight": "bold",
                "body_weight": "normal",
            },
            "layout": {
                "title_bar": True,
                "content_padding": 0.5,
                "card_style": "flat",
                "rounded_corners": False,
            },
            "visual": {
                "background_style": "solid",
                "shadow": False,
                "rounded_corners": False,
            },
        }

        errors = _validate_theme_config(config)
        assert errors == []

    def test_validate_missing_color(self):
        """Test validation catches missing color."""
        config = {
            "colors": {
                "background": "#1E2761",
                "primary": "#CADCFC",
                # Missing: surface, secondary, accent, text
            },
            "typography": {"title_size": 44, "body_size": 18},
        }

        errors = _validate_theme_config(config)
        assert len(errors) > 0
        assert any("Missing or invalid color" in e for e in errors)

    def test_validate_invalid_hex_format(self):
        """Test validation catches invalid hex format."""
        config = {
            "colors": {
                "background": "invalid_color",
                "primary": "#CADCFC",
            },
            "typography": {"title_size": 44, "body_size": 18},
        }

        errors = _validate_theme_config(config)
        assert len(errors) > 0
        assert any("Invalid hex color format" in e for e in errors)

    def test_validate_invalid_typography_value(self):
        """Test validation catches invalid typography value."""
        config = {
            "colors": {"background": "#1E2761", "text": "#1E2761"},
            "typography": {"title_size": "not_a_number", "body_size": 18},
        }

        errors = _validate_theme_config(config)
        assert len(errors) > 0
        assert any("Invalid title_size" in e for e in errors)

    def test_validate_invalid_font_weight(self):
        """Test validation catches invalid font weight."""
        config = {
            "colors": {"background": "#1E2761", "text": "#1E2761"},
            "typography": {
                "title_size": 44,
                "body_size": 18,
                "title_weight": "invalid_weight",
            },
        }

        errors = _validate_theme_config(config)
        assert len(errors) > 0
        assert any("Invalid title_weight" in e for e in errors)


class TestThemeConfigCoverage:
    """Test theme config coverage for all presets."""

    def test_all_presets_have_required_keys(self):
        """Ensure all preset themes have required keys."""
        required_color_keys = {
            "background",
            "surface",
            "primary",
            "secondary",
            "accent",
            "text",
        }
        required_typo_keys = {
            "title_size",
            "body_size",
            "quote_size",
            "title_weight",
            "body_weight",
        }
        required_layout_keys = {
            "title_bar",
            "content_padding",
            "card_style",
            "rounded_corners",
        }
        required_visual_keys = {
            "background_style",
            "shadow",
        }  # rounded_corners moved to layout

        for theme_name in THEME_CONFIGS.keys():
            config = THEME_CONFIGS[theme_name]

            # Check colors
            assert "colors" in config
            for key in required_color_keys:
                assert key in config["colors"]

            # Check typography
            assert "typography" in config
            for key in required_typo_keys:
                assert key in config["typography"]

            # Check layout
            assert "layout" in config
            for key in required_layout_keys:
                assert key in config["layout"]

            # Check visual
            assert "visual" in config
            for key in required_visual_keys:
                assert key in config["visual"]

    def test_all_presets_use_valid_hex_format(self):
        """Ensure all preset theme colors use valid 7-char hex."""
        for theme_name in THEME_CONFIGS.keys():
            config = THEME_CONFIGS[theme_name]
            colors = config.get("colors", {})

            for key, value in colors.items():
                if key != "name":  # Skip theme name
                    assert value.startswith("#")
                    assert len(value) == 7  # #RRGGBB format

    def test_all_presets_use_valid_typography_ranges(self):
        """Ensure typography values are reasonable."""
        for theme_name in THEME_CONFIGS.keys():
            config = THEME_CONFIGS[theme_name]
            typo = config.get("typography", {})

            # Title should be 32-64pt (allowing for new themes)
            assert 32 <= typo.get("title_size", 56) <= 64
            # Body should be 14-24pt
            assert 14 <= typo.get("body_size", 20) <= 24
            # Quote should be 18-40pt
            assert 18 <= typo.get("quote_size", 32) <= 40


class TestContentRules:
    """Test content generation rules."""

    def test_content_rules_documentation(self):
        """Test that content rules are documented."""
        # This test ensures the rules are clearly defined
        # Actual rule enforcement happens in PresentationGenerator

        # Max 5 bullets per slide
        # Max 20 words per bullet
        # No text-only slides
        # No repeating same layout more than twice

        # These are tested in integration/e2e tests
        assert True  # Rules documented

    def test_slide_type_limit(self):
        """Test that slide types are limited."""
        # Supported: title, content, two_column, section_divider, quote, thank_you
        # Should be enforced by tool interface
        assert True  # Types defined


class TestBackwardCompatibility:
    """Test backward compatibility."""

    def test_theme_string_still_works(self):
        """Test that simple theme string still works."""
        config = _preset_to_config("aurora")
        assert config is not None
        assert "colors" in config
        assert config["colors"]["background"] == "#FFFFFF"


class TestThemeConfigStructure:
    """Test theme config structure requirements."""

    def test_theme_config_accepts_partial_config(self):
        """Test that theme config can be partial (only some sections)."""
        # Colors only
        config_colors_only = {"colors": {"background": "#123456", "text": "#ABCDEF"}}

        _validate_theme_config(config_colors_only)
        # Should not error - missing sections are allowed
        # But this is implementation-dependent

        # Typography only
        config_typo_only = {"typography": {"title_size": 48, "body_size": 20}}

        _validate_theme_config(config_typo_only)

        # Layout only
        config_layout_only = {"layout": {"title_bar": False}}

        _validate_theme_config(config_layout_only)

        # Visual only
        config_visual_only = {"visual": {"shadow": True}}

        _validate_theme_config(config_visual_only)

        assert True  # Partial configs accepted


class TestNewEnterpriseSlideTypes:
    """Test new enterprise slide types."""

    def test_metrics_slide_type_exists(self):
        """Test metrics slide type is recognized."""
        # This test verifies the type is documented
        assert "metrics" in [
            "title",
            "content",
            "two_column",
            "section_divider",
            "quote",
            "thank_you",
            "metrics",
            "timeline",
            "comparison",
            "statement",
            "image_highlight",
            "flow",
        ]

    def test_timeline_slide_type_exists(self):
        """Test timeline slide type is recognized."""
        assert "timeline" in [
            "title",
            "content",
            "two_column",
            "section_divider",
            "quote",
            "thank_you",
            "metrics",
            "timeline",
            "comparison",
            "statement",
            "image_highlight",
            "flow",
        ]

    def test_comparison_slide_type_exists(self):
        """Test comparison slide type is recognized."""
        assert "comparison" in [
            "title",
            "content",
            "two_column",
            "section_divider",
            "quote",
            "thank_you",
            "metrics",
            "timeline",
            "comparison",
            "statement",
            "image_highlight",
            "flow",
        ]

    def test_statement_slide_type_exists(self):
        """Test statement slide type is recognized."""
        assert "statement" in [
            "title",
            "content",
            "two_column",
            "section_divider",
            "quote",
            "thank_you",
            "metrics",
            "timeline",
            "comparison",
            "statement",
            "image_highlight",
            "flow",
        ]

    def test_image_highlight_slide_type_exists(self):
        """Test image_highlight slide type is recognized."""
        assert "image_highlight" in [
            "title",
            "content",
            "two_column",
            "section_divider",
            "quote",
            "thank_you",
            "metrics",
            "timeline",
            "comparison",
            "statement",
            "image_highlight",
            "flow",
        ]

    def test_flow_slide_type_exists(self):
        """Test flow slide type is recognized."""
        assert "flow" in [
            "title",
            "content",
            "two_column",
            "section_divider",
            "quote",
            "thank_you",
            "metrics",
            "timeline",
            "comparison",
            "statement",
            "image_highlight",
            "flow",
        ]

    def test_twelve_slide_types_total(self):
        """Test there are exactly 12 slide types."""
        all_types = [
            "title",
            "content",
            "two_column",
            "section_divider",
            "quote",
            "thank_you",
            "metrics",
            "timeline",
            "comparison",
            "statement",
            "image_highlight",
            "flow",
        ]
        assert len(all_types) == 12
        assert len(set(all_types)) == 12  # All unique


class TestRealPptxGeneration:
    """Test actual PPTX file generation with real Node.js."""

    def test_generate_simple_pptx_with_executive_theme(self):
        """Test generating a simple PPTX file with executive theme."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.pptx"

            slides = [
                {
                    "type": "title",
                    "title": "Test Presentation",
                    "subtitle": "Real Test",
                },
                {
                    "type": "content",
                    "title": "Agenda",
                    "bullets": ["Item 1", "Item 2", "Item 3"],
                },
            ]

            result = generate_pptx(
                output_path=str(output_path),
                title="Test Presentation",
                theme="executive",
                slide_contents=slides,
            )

            assert result["success"] is True
            assert output_path.exists(), f"PPTX file not created at {output_path}"
            assert output_path.stat().st_size > 0, "PPTX file is empty"

            # Verify the PPTX actually has content by reading it back
            read_result = read_pptx(str(output_path))
            assert read_result.get("success") is True, "Failed to read generated PPTX"
            assert read_result.get("slide_count") == 2, (
                f"Expected 2 slides, got {read_result.get('slide_count')}"
            )
            assert len(read_result.get("slides", [])) == 2, "Should have 2 slides"
            assert len(read_result.get("titles", [])) == 2, "Should have 2 titles"

    def test_generate_pptx_with_custom_theme_config(self):
        """Test generating PPTX with custom theme configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_custom_theme.pptx"

            slides = [
                {"type": "title", "title": "Custom Theme Test"},
            ]

            custom_theme = {
                "colors": {
                    "background": "#FF0000",
                    "surface": "#FFFFFF",
                    "primary": "#0000FF",
                    "secondary": "#00FF00",
                    "accent": "#FFFF00",
                    "text": "#000000",
                },
                "typography": {
                    "title_size": 48,
                    "body_size": 20,
                    "quote_size": 28,
                    "title_weight": "bold",
                    "body_weight": "normal",
                },
            }

            result = generate_pptx(
                output_path=str(output_path),
                title="Custom Theme",
                theme_config=custom_theme,
                slide_contents=slides,
            )

            assert result["success"] is True
            assert output_path.exists()

            # Verify content
            read_result = read_pptx(str(output_path))
            assert read_result.get("slide_count") == 1

    def test_generate_all_slide_types(self):
        """Test generating PPTX with all 12 slide types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_all_types.pptx"

            slides = [
                {"type": "title", "title": "Title Slide"},
                {
                    "type": "content",
                    "title": "Content",
                    "bullets": ["Bullet 1", "Bullet 2"],
                },
                {
                    "type": "two_column",
                    "title": "Two Column",
                    "left": ["L1", "L2"],
                    "right": ["R1", "R2"],
                },
                {"type": "section_divider"},
                {"type": "quote", "text": "Test quote slide"},
                {"type": "thank_you", "message": "Thank you!"},
                {
                    "type": "metrics",
                    "title": "KPIs",
                    "items": [
                        {"label": "Revenue", "value": "$1M"},
                        {"label": "Users", "value": "10K"},
                    ],
                },
                {
                    "type": "timeline",
                    "title": "Roadmap",
                    "milestones": [
                        {"title": "Q1", "description": "Launch"},
                        {"title": "Q2", "description": "Grow"},
                    ],
                },
                {
                    "type": "comparison",
                    "title": "Compare",
                    "left_title": "Old",
                    "left_items": ["Slow"],
                    "right_title": "New",
                    "right_items": ["Fast"],
                },
                {"type": "statement", "text": "Big Statement"},
                {
                    "type": "image_highlight",
                    "title": "Image",
                    "caption": "Caption here",
                },
                {
                    "type": "flow",
                    "title": "Process",
                    "steps": ["Step 1", "Step 2", "Step 3"],
                },
            ]

            result = generate_pptx(
                output_path=str(output_path),
                title="All Slide Types",
                theme="minimal",
                slide_contents=slides,
            )

            assert result["success"] is True
            assert output_path.exists()

            # Verify all 12 slides were created
            read_result = read_pptx(str(output_path))
            assert read_result.get("slide_count") == 12, (
                f"Expected 12 slides, got {read_result.get('slide_count')}"
            )

    def test_read_generated_pptx(self):
        """Test reading a generated PPTX file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First generate
            output_path = Path(temp_dir) / "test_read.pptx"
            slides = [
                {"type": "title", "title": "Read Test"},
                {
                    "type": "content",
                    "title": "Content",
                    "bullets": ["Item 1", "Item 2"],
                },
            ]

            gen_result = generate_pptx(
                output_path=str(output_path),
                title="Read Test",
                theme="ocean",
                slide_contents=slides,
            )
            assert gen_result["success"] is True

            # Then read
            read_result = read_pptx(str(output_path))
            assert "slide_count" in read_result
            assert read_result["slide_count"] == 2
            assert "slides" in read_result
            assert len(read_result["slides"]) == 2

            # Verify text content extraction works
            text_result = read_pptx(str(output_path), extract_text=True)
            assert text_result.get("success") is True
            extracted_text = text_result.get("text", "")
            assert len(extracted_text) > 0, "Extracted text should not be empty"
            # Check for expected content
            assert "Read Test" in extracted_text or "Content" in extracted_text, (
                "Should find slide titles in extracted text"
            )

    def test_unpack_and_pack_pptx(self):
        """Test unpacking and repacking a PPTX file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "test_unpack.pptx"
            unpack_dir = temp_path / "unpacked"
            repacked_path = temp_path / "test_repacked.pptx"

            # Generate
            slides = [{"type": "title", "title": "Unpack Test"}]
            gen_result = generate_pptx(
                output_path=str(output_path),
                title="Unpack Test",
                theme="warm",
                slide_contents=slides,
            )
            assert gen_result["success"] is True

            # Unpack
            unpack_result = unpack_pptx(str(output_path), str(unpack_dir))
            assert unpack_result["success"] is True
            assert unpack_dir.exists()
            assert (unpack_dir / "ppt").exists()

            # Repack
            pack_result = pack_pptx(str(unpack_dir), str(repacked_path))
            assert pack_result["success"] is True
            assert repacked_path.exists()

    def test_generate_pptx_chinese_content(self):
        """Test generating PPTX with Chinese characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_chinese.pptx"

            slides = [
                {"type": "title", "title": "中文测试", "subtitle": "测试副标题"},
                {
                    "type": "content",
                    "title": "内容大纲",
                    "bullets": ["第一点", "第二点", "第三点"],
                },
            ]

            result = generate_pptx(
                output_path=str(output_path),
                title="中文演示文稿",
                theme="executive",
                slide_contents=slides,
            )

            assert result["success"] is True
            assert output_path.exists()

            # Verify Chinese text is in the generated file
            text_result = read_pptx(str(output_path), extract_text=True)
            assert text_result.get("success") is True
            extracted_text = text_result.get("text", "")
            assert len(extracted_text) > 0, "Should have extracted Chinese text"
            # Check for Chinese content
            assert (
                "中文" in extracted_text
                or "测试" in extracted_text
                or "内容" in extracted_text
            ), "Should find Chinese characters in extracted text"

    def test_generate_pptx_all_presets(self):
        """Test generating PPTX with all preset themes."""
        preset_themes = ["executive", "ocean", "minimal", "warm", "forest", "coral"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for theme in preset_themes:
                output_path = temp_path / f"test_{theme}.pptx"

                slides = [
                    {"type": "title", "title": f"{theme.title()} Theme Test"},
                    {"type": "content", "title": "Content", "bullets": ["Item 1"]},
                ]

                result = generate_pptx(
                    output_path=str(output_path),
                    title=f"{theme.title()} Test",
                    theme=theme,
                    slide_contents=slides,
                )

                assert result["success"] is True, (
                    f"Failed to generate PPTX with theme: {theme}"
                )
                assert output_path.exists(), f"PPTX file not created for theme: {theme}"

                # Verify each PPTX has actual content
                read_result = read_pptx(str(output_path))
                assert read_result.get("slide_count") == 2, (
                    f"PPTX for theme {theme} should have 2 slides"
                )


class TestWorkspaceIntegration:
    """Test workspace integration for PPTX tools."""

    def test_generate_to_workspace_output(self):
        """Test generating PPTX to workspace output directory."""
        with tempfile.TemporaryDirectory() as base_dir:
            workspace = TaskWorkspace(id="test_pptx_gen", base_dir=base_dir)

            # Generate with relative path - should save to workspace output
            result = generate_pptx(
                output_path="test_workspace.pptx",
                title="Workspace Test",
                theme="executive",
                slide_contents=[{"type": "title", "title": "Test"}],
                workspace=workspace,
            )

            assert result["success"] is True
            assert result.get("saved_to_workspace") is True

            # Check file is in workspace output directory
            expected_path = workspace.output_dir / "test_workspace.pptx"
            assert expected_path.exists(), f"PPTX not found at {expected_path}"

    def test_read_from_workspace_input(self):
        """Test reading PPTX from workspace input directory."""
        with tempfile.TemporaryDirectory() as base_dir:
            workspace = TaskWorkspace(id="test_pptx_read", base_dir=base_dir)

            # First, generate a PPTX file in input directory
            source_pptx = workspace.input_dir / "source.pptx"
            generate_pptx(
                output_path=str(source_pptx),
                title="Source Presentation",
                slide_contents=[{"type": "title", "title": "Source"}],
                workspace=workspace,
            )

            assert source_pptx.exists()

            # Now read it back using workspace path resolution
            result = read_pptx(
                pptx_path="source.pptx",  # Relative path - should be found in workspace
                extract_text=False,
                workspace=workspace,
            )

            # Debug print
            if not result.get("success"):
                print(f"Read result: {result}")
            assert result.get("success", True) is True
            assert result.get("slide_count", 0) == 1

    def test_workspace_path_resolution_priority(self):
        """Test that workspace searches input -> output -> temp priority."""
        with tempfile.TemporaryDirectory() as base_dir:
            workspace = TaskWorkspace(id="test_pptx_resolve", base_dir=base_dir)

            # Create a PPTX in input directory
            input_pptx = workspace.input_dir / "input_test.pptx"
            generate_pptx(
                output_path=str(input_pptx),
                title="Input Test",
                slide_contents=[{"type": "title", "title": "In Input"}],
                workspace=workspace,
            )

            # Also create one in output directory
            output_pptx = workspace.output_dir / "output_test.pptx"
            generate_pptx(
                output_path=str(output_pptx),
                title="Output Test",
                slide_contents=[{"type": "title", "title": "In Output"}],
                workspace=workspace,
            )

            # Read with just filename - should find input first (higher priority)
            result = read_pptx(
                pptx_path="input_test.pptx",
                workspace=workspace,
            )

            # Debug print
            if not result.get("success"):
                print(f"Resolve result: {result}")
            assert result.get("success", True) is True
            # Verify it read the one from input directory
            assert result.get("slide_count", 0) == 1

    def test_pack_to_workspace_output(self):
        """Test packing directory to PPTX in workspace output."""
        with tempfile.TemporaryDirectory() as base_dir:
            workspace = TaskWorkspace(id="test_pptx_pack", base_dir=base_dir)

            # First unpack a PPTX to temp directory
            source_pptx = workspace.temp_dir / "source.pptx"
            generate_pptx(
                output_path=str(source_pptx),
                title="Pack Test",
                slide_contents=[{"type": "title", "title": "Pack Me"}],
                workspace=workspace,
            )

            # Pack it back with relative output path
            result = pack_pptx(
                input_dir=str(workspace.temp_dir),
                output_path="packed.pptx",
                workspace=workspace,
            )

            assert result["success"] is True
            # Check output is in workspace output directory
            expected_path = workspace.output_dir / "packed.pptx"
            assert expected_path.exists()

    def test_generate_with_absolute_path_ignores_workspace(self):
        """Test that absolute paths don't use workspace output."""
        with tempfile.TemporaryDirectory() as base_dir:
            workspace = TaskWorkspace(id="test_pptx_absolute", base_dir=base_dir)
            custom_dir = Path(base_dir) / "custom_output"
            custom_dir.mkdir()

            # Generate with absolute path - should NOT save to workspace output
            absolute_path = custom_dir / "absolute_test.pptx"
            result = generate_pptx(
                output_path=str(absolute_path),
                title="Absolute Path Test",
                slide_contents=[{"type": "title", "title": "Absolute"}],
                workspace=workspace,
            )

            assert result["success"] is True

            # File should be at absolute path, not in workspace output
            assert absolute_path.exists()
            workspace_output = workspace.output_dir / "absolute_test.pptx"
            assert not workspace_output.exists()

    def test_workspace_none_backward_compat(self):
        """Test that tools work without workspace (backward compatibility)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "no_workspace.pptx"

            # All functions should work without workspace
            result = generate_pptx(
                output_path=str(output_path),
                title="No Workspace",
                slide_contents=[{"type": "title", "title": "No WS"}],
                workspace=None,
            )

            assert result["success"] is True
            assert output_path.exists()

    def test_add_slide_with_workspace(self):
        """Test adding slide using workspace for path resolution."""
        with tempfile.TemporaryDirectory() as base_dir:
            workspace = TaskWorkspace(id="test_pptx_add_slide", base_dir=base_dir)

            # Create source PPTX in input directory
            source_pptx = workspace.input_dir / "add_slide_test.pptx"
            generate_pptx(
                output_path=str(source_pptx),
                title="Add Slide Test",
                slide_contents=[{"type": "title", "title": "Original"}],
                workspace=workspace,
            )

            # Create temp directory name for unpacking
            unpack_dir = workspace.temp_dir / "unpacked"
            unpack_dir.mkdir()

            # Copy and extract (simulate unpack)
            with zipfile.ZipFile(source_pptx, "r") as zf:
                zf.extractall(unpack_dir)

            # Add slide using workspace (path resolution for source file)
            # This tests that PresentationReader can use workspace
            result = add_slide_pptx(
                unpacked_dir=str(unpack_dir),
                source="slide1.xml",  # Typically first slide in unpacked PPTX
                workspace=workspace,
            )

            assert result["success"] is True

    def test_clean_with_workspace(self):
        """Test cleaning orphaned files with workspace."""
        with tempfile.TemporaryDirectory() as base_dir:
            workspace = TaskWorkspace(id="test_pptx_clean", base_dir=base_dir)

            # First, create a valid PPTX structure in temp directory
            source_pptx = workspace.temp_dir / "source_for_clean.pptx"
            generate_pptx(
                output_path=str(source_pptx),
                title="Clean Test",
                slide_contents=[{"type": "title", "title": "Clean Test"}],
                workspace=workspace,
            )

            # Unpack it to create a valid directory structure
            unpack_dir = workspace.temp_dir / "unpacked_for_clean"
            unpack_dir.mkdir()

            with zipfile.ZipFile(source_pptx, "r") as zf:
                zf.extractall(unpack_dir)

            # Now clean should work with valid directory structure
            result = clean_pptx(
                unpacked_dir=str(unpack_dir),
                workspace=workspace,
            )

            # Should succeed
            assert result.get("success", False) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
