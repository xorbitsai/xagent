"""
PowerPoint Presentation Tool for xagent with structured theme system.

Provides core PPTX functionality:
- Read PPTX files and extract content
- Generate PPTX files using pptxgenjs (JavaScript PPTX library)
- Advanced editing: unpack, pack, add slide, clean

Structured Theme Configuration:
- Preset themes (executive, ocean, minimal, etc.) with predefined colors/typography/layout
- Custom themes via theme_config parameter for fine-grained control
"""

import json
import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from xml.etree import ElementTree as ET

from ...workspace import TaskWorkspace

# PPTX namespaces
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

logger = logging.getLogger(__name__)


# ============================================================================
# THEME CONFIGURATION SYSTEM
# ============================================================================

# Preset theme configurations
THEME_CONFIGS = {
    "aurora": {
        "name": "Aurora",
        "colors": {
            "background": "#FFFFFF",
            "surface": "#FFFFFF",
            "primary": "#0B1F3B",
            "primary_light": "#4A5B7C",
            "primary_dark": "#081629",
            "secondary": "#5C6B80",
            "accent": "#2563EB",
            "text": "#0B1F3B",
            "success": "#10B981",
            "warning": "#F59E0B",
            "error": "#EF4444",
            "info": "#3B82F6",
            "border": "#E5E7EB",
            "subtle": "#F3F4F6",
        },
        "typography": {
            "title_font": "Arial",
            "body_font": "Calibri",
            "mono_font": "Courier New",
            "font_fallback": "sans-serif",
            "title_size": 56,
            "subtitle_size": 28,
            "body_size": 20,
            "quote_size": 32,
            "small_size": 14,
            "line_height": 1.45,
            "max_text_width_ratio": 0.62,
            "letter_spacing": 0.5,
            "title_weight": "bold",
            "body_weight": "normal",
            "title_align": "left",
            "body_align": "left",
        },
        "layout": {
            "side_margin": 0.9,
            "top_margin": 0.9,
            "column_gap": 0.7,
            "section_gap": 0.6,
            "bullet_spacing": 0.35,
            "content_padding": 0.8,
            "title_bar": False,
            "card_style": "minimal",
            "rounded_corners": False,
        },
        "visual": {
            "background_style": "solid",
            "background_gradient_type": None,
            "background_gradient_direction": None,
            "background_gradient_stops": None,
            "divider_opacity": 0.12,
            "shadow": False,
            "card_radius": 6,
            "background_ratio": 0.0,
            "accent_thickness": 2,
        },
        "proportion": {
            "title_area_ratio": 0.28,
            "content_area_ratio": 0.62,
            "footer_area_ratio": 0.10,
        },
        "shapes": {
            "bullet_style": "circle",
            "bullet_color": "primary",
            "bullet_size": 0.3,
            "divider_style": "thin",
            "divider_color": "border",
            "divider_length": "full",
            "corner_style": "rounded",
            "corner_radius": 4,
        },
        "media": {
            "image_border": False,
            "image_shadow": False,
            "image_corner_radius": 8,
            "placeholder_color": "#F3F4F6",
            "placeholder_text": "Insert image",
        },
    },
    "vortex": {
        "name": "Vortex",
        "colors": {
            "background": "#0F172A",
            "surface": "#111827",
            "primary": "#F8FAFC",
            "primary_light": "#CBD5E1",
            "primary_dark": "#64748B",
            "secondary": "#94A3B8",
            "accent": "#22D3EE",
            "text": "#F8FAFC",
            "success": "#34D399",
            "warning": "#FBBF24",
            "error": "#F87171",
            "info": "#60A5FA",
            "border": "#374151",
            "subtle": "#1F2937",
        },
        "typography": {
            "title_font": "Segoe UI",
            "body_font": "Segoe UI",
            "mono_font": "Consolas",
            "font_fallback": "sans-serif",
            "title_size": 58,
            "subtitle_size": 28,
            "body_size": 20,
            "quote_size": 34,
            "small_size": 14,
            "line_height": 1.45,
            "max_text_width_ratio": 0.60,
            "letter_spacing": 0.5,
            "title_weight": "bold",
            "body_weight": "normal",
            "title_align": "left",
            "body_align": "left",
        },
        "layout": {
            "side_margin": 1.0,
            "top_margin": 1.0,
            "column_gap": 0.8,
            "section_gap": 0.6,
            "bullet_spacing": 0.35,
            "content_padding": 0.9,
            "title_bar": False,
            "card_style": "minimal",
            "rounded_corners": False,
        },
        "visual": {
            "background_style": "solid",
            "background_gradient_type": None,
            "background_gradient_direction": None,
            "background_gradient_stops": None,
            "divider_opacity": 0.15,
            "shadow": True,
            "card_radius": 8,
            "background_ratio": 0.0,
            "accent_thickness": 2,
        },
        "proportion": {
            "title_area_ratio": 0.30,
            "content_area_ratio": 0.60,
            "footer_area_ratio": 0.10,
        },
        "shapes": {
            "bullet_style": "square",
            "bullet_color": "accent",
            "bullet_size": 0.25,
            "divider_style": "thick",
            "divider_color": "primary",
            "divider_length": "full",
            "corner_style": "sharp",
            "corner_radius": 0,
        },
        "media": {
            "image_border": True,
            "image_shadow": True,
            "image_corner_radius": 4,
            "placeholder_color": "#1F2937",
            "placeholder_text": "Insert image",
        },
    },
    "mono": {
        "name": "Mono",
        "colors": {
            "background": "#FAFAFA",
            "surface": "#FFFFFF",
            "primary": "#111111",
            "primary_light": "#666666",
            "primary_dark": "#000000",
            "secondary": "#666666",
            "accent": "#000000",
            "text": "#111111",
            "success": "#059669",
            "warning": "#D97706",
            "error": "#DC2626",
            "info": "#2563EB",
            "border": "#E5E5E5",
            "subtle": "#F5F5F5",
        },
        "typography": {
            "title_font": "Helvetica",
            "body_font": "Helvetica",
            "mono_font": "Monaco",
            "font_fallback": "sans-serif",
            "title_size": 60,
            "subtitle_size": 26,
            "body_size": 19,
            "quote_size": 36,
            "small_size": 14,
            "line_height": 1.5,
            "max_text_width_ratio": 0.58,
            "letter_spacing": 0.6,
            "title_weight": "bold",
            "body_weight": "normal",
            "title_align": "center",
            "body_align": "left",
        },
        "layout": {
            "side_margin": 1.1,
            "top_margin": 1.0,
            "column_gap": 0.8,
            "section_gap": 0.7,
            "bullet_spacing": 0.4,
            "content_padding": 0.9,
            "title_bar": False,
            "card_style": "none",
            "rounded_corners": False,
        },
        "visual": {
            "background_style": "solid",
            "background_gradient_type": None,
            "background_gradient_direction": None,
            "background_gradient_stops": None,
            "divider_opacity": 0.08,
            "shadow": False,
            "card_radius": 0,
            "background_ratio": 0.0,
            "accent_thickness": 1,
        },
        "proportion": {
            "title_area_ratio": 0.30,
            "content_area_ratio": 0.60,
            "footer_area_ratio": 0.10,
        },
        "shapes": {
            "bullet_style": "dash",
            "bullet_color": "primary",
            "bullet_size": 0.2,
            "divider_style": "dotted",
            "divider_color": "border",
            "divider_length": "partial",
            "corner_style": "sharp",
            "corner_radius": 0,
        },
        "media": {
            "image_border": False,
            "image_shadow": False,
            "image_corner_radius": 0,
            "placeholder_color": "#F5F5F5",
            "placeholder_text": "Insert image",
        },
    },
}


def _preset_to_config(theme: str) -> Dict[str, Any]:
    """Convert preset theme name to structured theme configuration.

    Args:
        theme: Preset theme name (aurora, vortex, mono)

    Returns:
        Structured theme configuration dictionary with all fields populated
    """
    theme_lower: str = theme.lower()

    # Default fallback configuration (Aurora)
    default_config: Dict[str, Any] = {
        "name": "Aurora",
        "colors": {
            "background": "#FFFFFF",
            "surface": "#FFFFFF",
            "primary": "#0B1F3B",
            "primary_light": "#4A5B7C",
            "primary_dark": "#081629",
            "secondary": "#5C6B80",
            "accent": "#2563EB",
            "text": "#0B1F3B",
            "success": "#10B981",
            "warning": "#F59E0B",
            "error": "#EF4444",
            "info": "#3B82F6",
            "border": "#E5E7EB",
            "subtle": "#F3F4F6",
        },
        "typography": {
            "title_font": "Arial",
            "body_font": "Calibri",
            "mono_font": "Courier New",
            "font_fallback": "sans-serif",
            "title_size": 56,
            "subtitle_size": 28,
            "body_size": 20,
            "quote_size": 32,
            "small_size": 14,
            "line_height": 1.45,
            "max_text_width_ratio": 0.62,
            "letter_spacing": 0.5,
            "title_weight": "bold",
            "body_weight": "normal",
            "title_align": "left",
            "body_align": "left",
        },
        "layout": {
            "title_bar": False,
            "content_padding": 0.8,
            "card_style": "minimal",
            "rounded_corners": False,
        },
        "spacing": {
            "side_margin": 0.9,
            "top_margin": 0.9,
            "column_gap": 0.7,
            "section_gap": 0.6,
            "bullet_spacing": 0.35,
        },
        "visual": {
            "background_style": "solid",
            "background_gradient_type": None,
            "background_gradient_direction": None,
            "background_gradient_stops": None,
            "shadow": False,
            "rounded_corners": False,
        },
        "visual_weight": {
            "divider_opacity": 0.12,
            "accent_thickness": 2,
            "card_radius": 6,
            "background_ratio": 0.0,
        },
        "proportion": {
            "title_area_ratio": 0.28,
            "content_area_ratio": 0.62,
            "footer_area_ratio": 0.10,
        },
        "shapes": {
            "bullet_style": "circle",
            "bullet_color": "primary",
            "bullet_size": 0.3,
            "divider_style": "thin",
            "divider_color": "border",
            "divider_length": "full",
            "corner_style": "rounded",
            "corner_radius": 4,
        },
        "media": {
            "image_border": False,
            "image_shadow": False,
            "image_corner_radius": 8,
            "placeholder_color": "#F3F4F6",
            "placeholder_text": "Insert image",
        },
    }

    # Get preset config or use default
    config = THEME_CONFIGS.get(theme_lower, default_config)

    # Normalize structure - extract nested fields to proper sections
    # New themes have spacing/visual_weight/proportion nested in layout or visual

    # Extract spacing from layout if not separate
    if "spacing" not in config and "layout" in config:
        layout = config["layout"]
        spacing_fields = [
            "side_margin",
            "top_margin",
            "column_gap",
            "section_gap",
            "bullet_spacing",
        ]
        if any(field in layout for field in spacing_fields):
            config["spacing"] = {}
            for field in spacing_fields:
                if field in layout:
                    config["spacing"][field] = layout[field]
            for field in spacing_fields:
                layout.pop(field, None)

    # Extract visual_weight from visual if not separate
    if "visual_weight" not in config and "visual" in config:
        visual = config["visual"]
        vw_fields = [
            "divider_opacity",
            "accent_thickness",
            "card_radius",
            "background_ratio",
        ]
        if any(field in visual for field in vw_fields):
            config["visual_weight"] = {}
            for field in vw_fields:
                if field in visual:
                    config["visual_weight"][field] = visual[field]
            for field in vw_fields:
                visual.pop(field, None)

    # Extract proportion from layout if not separate
    if "proportion" not in config and "layout" in config:
        layout = config["layout"]
        prop_fields = ["title_area_ratio", "content_area_ratio", "footer_area_ratio"]
        if any(field in layout for field in prop_fields):
            config["proportion"] = {}
            for field in prop_fields:
                if field in layout:
                    config["proportion"][field] = layout[field]
            for field in prop_fields:
                layout.pop(field, None)

    return config


def _validate_theme_config(theme_config: Dict[str, Any]) -> List[str]:
    """Validate theme configuration and return list of errors.

    Args:
        theme_config: Structured theme configuration

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: List[str] = []

    # Validate colors
    if "colors" in theme_config:
        colors = theme_config["colors"]
        # P0: Required core colors
        required_colors = [
            "background",
            "surface",
            "primary",
            "secondary",
            "accent",
            "text",
        ]
        # P0: Extended colors
        optional_colors = [
            "primary_light",
            "primary_dark",
            "success",
            "warning",
            "error",
            "info",
            "border",
            "subtle",
        ]
        all_colors = required_colors + optional_colors

        for key in all_colors:
            if key in colors and colors[key]:
                # Validate hex format
                if not colors[key].startswith("#") or len(colors[key]) != 7:
                    errors.append(f"Invalid hex color format for {key}: {colors[key]}")

        # Check required colors
        for key in required_colors:
            if key not in colors or not colors[key]:
                errors.append(f"Missing or invalid color: {key}")

    # Validate typography
    if "typography" in theme_config:
        typo = theme_config["typography"]

        # P0: Font family validation (optional but recommended)
        font_fields = ["title_font", "body_font", "mono_font", "font_fallback"]
        for field in font_fields:
            if field in typo and not isinstance(typo[field], str):
                errors.append(f"Invalid {field}: must be string")

        # Existing validation
        numeric_fields = ["title_size", "body_size", "quote_size"]
        for field in numeric_fields:
            if field in typo and not isinstance(typo[field], (int, float)):
                errors.append(f"Invalid {field}: must be number")

        # P1: Alignment validation
        align_fields = ["title_align", "body_align"]
        valid_aligns = ["left", "center", "right"]
        for field in align_fields:
            if field in typo and typo[field] not in valid_aligns:
                errors.append(
                    f"Invalid {field}: {typo[field]} (must be one of {valid_aligns})"
                )

        # Validate weight values
        valid_weights = ["normal", "bold", "semibold", "light"]
        if "title_weight" in typo and typo["title_weight"] not in valid_weights:
            errors.append(f"Invalid title_weight: {typo['title_weight']}")
        if "body_weight" in typo and typo["body_weight"] not in valid_weights:
            errors.append(f"Invalid body_weight: {typo['body_weight']}")

    # Validate layout
    if "layout" in theme_config:
        layout = theme_config["layout"]
        for key in layout:
            if not isinstance(layout[key], (bool, int, float, str)):
                errors.append(f"Invalid layout value type for {key}: {layout[key]}")

    # Validate visual (P1: gradients)
    if "visual" in theme_config:
        visual = theme_config["visual"]
        for key, value in visual.items():
            if key == "background_gradient_type" and value is not None:
                if value not in ["linear", "radial", None]:
                    errors.append(f"Invalid background_gradient_type: {value}")
            elif key == "background_gradient_direction" and value is not None:
                if not isinstance(value, (int, float)) or not (0 <= value <= 360):
                    errors.append(f"Invalid background_gradient_direction: {value}")
            elif key == "background_gradient_stops" and value is not None:
                if not isinstance(value, list):
                    errors.append("background_gradient_stops must be a list")
            elif key in [
                "divider_opacity",
                "card_radius",
                "background_ratio",
                "accent_thickness",
            ]:
                # These can be numbers
                if not isinstance(value, (int, float, type(None))):
                    errors.append(
                        f"Invalid visual value type for {key}: must be number"
                    )
            elif not isinstance(value, (bool, str, type(None))):
                errors.append(f"Invalid visual value type for {key}: {value}")

    # P2: Validate shapes
    if "shapes" in theme_config:
        shapes = theme_config["shapes"]

        if "bullet_style" in shapes and shapes["bullet_style"] not in [
            "circle",
            "square",
            "dash",
            "custom",
        ]:
            errors.append(f"Invalid bullet_style: {shapes['bullet_style']}")

        if "divider_style" in shapes and shapes["divider_style"] not in [
            "thin",
            "thick",
            "double",
            "dotted",
        ]:
            errors.append(f"Invalid divider_style: {shapes['divider_style']}")

        if "corner_style" in shapes and shapes["corner_style"] not in [
            "rounded",
            "sharp",
            "custom",
        ]:
            errors.append(f"Invalid corner_style: {shapes['corner_style']}")

        if "bullet_size" in shapes and not isinstance(
            shapes["bullet_size"], (int, float)
        ):
            errors.append("Invalid bullet_size: must be number")

        if "corner_radius" in shapes and not isinstance(
            shapes["corner_radius"], (int, float)
        ):
            errors.append("Invalid corner_radius: must be number")

    # P2: Validate media
    if "media" in theme_config:
        media = theme_config["media"]

        bool_fields = ["image_border", "image_shadow"]
        for field in bool_fields:
            if field in media and not isinstance(media[field], bool):
                errors.append(f"Invalid {field}: must be boolean")

        if "image_corner_radius" in media and not isinstance(
            media["image_corner_radius"], (int, float)
        ):
            errors.append("Invalid image_corner_radius: must be number")

    return errors


def register_namespaces() -> None:
    """Register XML namespaces for parsing."""
    ET.register_namespace("p", NS_P)
    ET.register_namespace("a", NS_A)
    ET.register_namespace("r", NS_R)


# ============================================================================
# PPTX READING
# ============================================================================


class PresentationReader:
    """Reads and extracts information from PPTX files."""

    def __init__(self, pptx_path: str, workspace: Optional[TaskWorkspace] = None):
        """
        Initialize PresentationReader with workspace support.

        Args:
            pptx_path: Path to PPTX file
            workspace: Optional workspace for resolving file paths
        """
        self._workspace = workspace
        self.pptx_path = self._resolve_path(pptx_path)
        self.temp_dir: Optional[Path] = None

    def _resolve_path(self, file_path: str) -> Path:
        """
        Resolve file path using workspace if available.

        Args:
            file_path: File path to resolve

        Returns:
            Resolved absolute path
        """
        path = Path(file_path)

        # If workspace is available and path is relative, use workspace to resolve
        if self._workspace and not path.is_absolute():
            try:
                resolved = self._workspace.resolve_path_with_search(file_path)
                logger.info(
                    f"Resolved PPTX path using workspace: {file_path} -> {resolved}"
                )
                return resolved
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Cannot resolve PPTX path in workspace: {e}")
                # Fall back to simple path resolution
            except Exception as e:
                logger.warning(f"Error using workspace path resolution: {e}")
                # Fall back to simple path resolution

        # Fallback: simple path resolution (for when workspace is not available)
        if not path.is_absolute():
            path = Path.cwd() / path

        return path.resolve()

    def __enter__(self) -> "PresentationReader":
        self.temp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(self.pptx_path, "r") as zf:
            zf.extractall(self.temp_dir)
        return self

    def __exit__(self, *args: Any) -> None:
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def read_presentation(self) -> Dict[str, Any]:
        """Extract presentation information."""
        register_namespaces()

        if self.temp_dir is None:
            return {
                "error": "temp_dir not initialized",
                "slide_count": 0,
                "slides": [],
                "titles": [],
            }

        pres_xml_path = self.temp_dir / "ppt" / "presentation.xml"
        if not pres_xml_path.exists():
            return {
                "error": "presentation.xml not found",
                "slide_count": 0,
                "slides": [],
                "titles": [],
            }

        tree = ET.parse(str(pres_xml_path))
        root = tree.getroot()

        # Get slide references
        slide_refs: Dict[str, bool] = {}
        for sld_id in root.findall(".//p:sldId", namespaces={"p": NS_P}):
            r_id = sld_id.get(f"{{{NS_R}}}id")
            if r_id is None:
                continue
            target_attr = sld_id.get("show")
            is_hidden = target_attr == "0"
            slide_refs[r_id] = is_hidden

        # Get slide info from relationships
        rels_xml_path = self.temp_dir / "ppt" / "_rels" / "presentation.xml.rels"
        if not rels_xml_path.exists():
            return {
                "error": "relationships.xml not found",
                "slide_count": 0,
                "slides": [],
                "titles": [],
            }

        rels_tree = ET.parse(str(rels_xml_path))
        slide_files: Dict[str, str] = {}

        for rel in rels_tree.getroot():
            r_id = rel.get("Id")
            target = rel.get("Target")
            if r_id is None or target is None:
                continue
            if "slide" in target:
                slide_num = r_id.replace("rId", "")
                slide_files[slide_num] = target.replace("slides/", "")

        slides: List[Dict[str, Any]] = []
        for idx, (num, is_hidden) in enumerate(sorted(slide_refs.items())):
            filename = slide_files.get(num, f"slide{idx + 1}.xml")
            slides.append({"index": idx, "filename": filename, "hidden": is_hidden})

        titles = []
        for slide in slides:
            slide_path = self.temp_dir / "ppt" / "slides" / str(slide["filename"])
            if slide_path.exists():
                try:
                    tree = ET.parse(str(slide_path))
                    root = tree.getroot()

                    texts = []
                    for t in root.findall(".//a:t", namespaces={"a": NS_A}):
                        if t.text and t.text.strip():
                            texts.append(t.text.strip())

                    titles.append(texts[0] if texts else "")
                except Exception as e:
                    logger.debug(f"Failed to parse slide title: {e}")
                    titles.append("")

        return {"slide_count": len(slides), "slides": slides, "titles": titles}

    def extract_text(self) -> str:
        """Extract all text content from presentation."""
        register_namespaces()

        if self.temp_dir is None:
            return "Error: temp_dir not initialized"

        result = []
        info = self.read_presentation()

        if "error" in info:
            return f"Error: {info.get('error')}"

        for slide in info.get("slides", []):
            if not slide["hidden"]:
                slide_path = self.temp_dir / "ppt" / "slides" / slide["filename"]
                if slide_path.exists():
                    try:
                        tree = ET.parse(str(slide_path))
                        root = tree.getroot()

                        for t in root.findall(".//a:t", namespaces={"a": NS_A}):
                            if t.text and t.text.strip():
                                result.append(f"- {t.text.strip()}\n")
                    except Exception as e:
                        logger.debug(f"Failed to extract text from slide: {e}")

                result.append("\n")

        return "".join(result)

    def unpack(self, output_dir: str) -> Dict[str, Any]:
        """Unpack PPTX files to directory for manual editing.

        Args:
            output_dir: Directory path to extract files to

        Returns:
            Dictionary with success status, output directory, and file count
        """
        if self.temp_dir is None:
            return {"error": "No PPTX file loaded", "success": False}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy all files to output directory
        import shutil

        file_count = 0
        for item in self.temp_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(self.temp_dir)
                dest_file = output_path / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_file)
                file_count += 1

        # Pretty print XML files
        self._prettify_xml_files(output_path)

        return {
            "success": True,
            "output_dir": str(output_path),
            "file_count": file_count,
            "message": f"Extracted {file_count} files to {output_dir}",
        }

    def pack(self, output_path: str, validate: bool = True) -> Dict[str, Any]:
        """Pack directory into PPTX file.

        Args:
            output_path: Output .pptx file path
            validate: Whether to validate structure (default: True)

        Returns:
            Dictionary with success status and output file path
        """
        if self.temp_dir is None:
            return {"error": "No directory loaded", "success": False}

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Create ZIP file
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.temp_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.temp_dir)
                    zf.write(file_path, arcname)

        return {
            "success": True,
            "output_path": str(output_path),
            "message": f"Created PPTX file: {output_path}",
        }

    def add_slide(self, source: str) -> Dict[str, Any]:
        """Add slide to unpacked PPTX directory.

        Args:
            source: Source slide file (e.g., "slide2.xml") or layout (e.g., "slideLayout1.xml")

        Returns:
            Dictionary with success status and slide information
        """
        if self.temp_dir is None:
            return {"error": "No directory loaded", "success": False}

        slides_dir = self.temp_dir / "ppt" / "slides"
        if not slides_dir.exists():
            return {"error": "slides directory not found", "success": False}

        # Find existing slides
        existing_slides = sorted(slides_dir.glob("slide*.xml"))
        if not existing_slides:
            return {"error": "No existing slides found", "success": False}

        # Determine source
        source_path = None
        if source.startswith("slide") and source.endswith(".xml"):
            source_path = slides_dir / source
        elif source.startswith("slideLayout") and source.endswith(".xml"):
            layout_path = self.temp_dir / "ppt" / "slideLayouts" / source
            if layout_path.exists():
                source_path = layout_path
            else:
                return {"error": f"Layout not found: {source}", "success": False}
        else:
            return {"error": f"Invalid source: {source}", "success": False}

        if not source_path.exists():
            return {"error": f"Source not found: {source}", "success": False}

        # Get next slide number
        last_slide_num = int(existing_slides[-1].stem.replace("slide", ""))
        new_slide_num = last_slide_num + 1
        new_slide_name = f"slide{new_slide_num}.xml"
        new_slide_path = slides_dir / new_slide_name

        # Copy source to new slide
        import shutil

        shutil.copy2(source_path, new_slide_path)

        return {
            "success": True,
            "slide_name": new_slide_name,
            "slide_number": new_slide_num,
            "message": f"Added {new_slide_name}",
        }

    def clean(self) -> Dict[str, Any]:
        """Clean orphaned files from unpacked PPTX directory.

        Removes slides that are not in presentation.xml slide list,
        unreferenced media files, and orphaned relationship files.

        Returns:
            Dictionary with success status and count of removed files
        """
        if self.temp_dir is None:
            return {"error": "No directory loaded", "success": False}

        removed_count = 0

        # Get valid slide IDs from presentation.xml
        pres_xml_path = self.temp_dir / "ppt" / "presentation.xml"
        if not pres_xml_path.exists():
            return {"error": "presentation.xml not found", "success": False}

        register_namespaces()
        tree = ET.parse(str(pres_xml_path))
        root = tree.getroot()

        valid_slide_ids = set()
        for sld_id in root.findall(".//p:sldId", namespaces={"p": NS_P}):
            r_id = sld_id.get(f"{{{NS_R}}}id")
            if r_id:
                valid_slide_ids.add(r_id)

        # Get valid relationships
        rels_xml_path = self.temp_dir / "ppt" / "_rels" / "presentation.xml.rels"
        if rels_xml_path.exists():
            rels_tree = ET.parse(str(rels_xml_path))
            valid_targets = set()
            for rel in rels_tree.getroot():
                target = rel.get("Target", "")
                if target.startswith("slides/"):
                    valid_targets.add(target.replace("slides/", ""))

            # Remove orphaned slides
            slides_dir = self.temp_dir / "ppt" / "slides"
            if slides_dir.exists():
                for slide_file in slides_dir.glob("slide*.xml"):
                    if slide_file.name not in valid_targets:
                        slide_file.unlink()
                        removed_count += 1

        return {
            "success": True,
            "removed_count": removed_count,
            "message": f"Removed {removed_count} orphaned files",
        }

    def _prettify_xml_files(self, directory: Path) -> None:
        """Pretty print XML files for easier manual editing."""
        for xml_file in directory.rglob("*.xml"):
            try:
                tree = ET.parse(str(xml_file))
                ET.indent(tree, space="  ")
                tree.write(str(xml_file), encoding="UTF-8", xml_declaration=True)
            except Exception as e:
                logger.debug(f"Failed to format XML file {xml_file}: {e}")
                pass  # Skip files that can't be parsed


# ============================================================================
# PPTX GENERATION (via pptxgenjs npm package)
# ============================================================================


class PresentationGenerator:
    """Generate PowerPoint presentations using structured theme configuration.

    Supports:
    - Preset themes (executive, ocean, minimal, warm, forest, coral)
    - Custom themes via theme_config for fine-grained control
    """

    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        """
        Initialize PresentationGenerator with workspace support.

        Args:
            workspace: Optional workspace for saving generated files
        """
        self._workspace = workspace
        self.temp_dir: Optional[Path] = None
        self.slides: List[Dict[str, Any]] = []
        self.title = "Presentation"
        self.theme_config: Optional[Dict[str, Any]] = None

    def create(self, title: str = "Presentation") -> None:
        """Initialize a new presentation.

        Args:
            title: Presentation title
        """
        self.slides = []
        self.title = title
        self.theme_config = None

    def add_slide(self, slide_type: str, **content: Any) -> None:
        """Add a slide to presentation.

        Args:
            slide_type: One of 12 slide types:
                       - 'title': Title slide with subtitle
                       - 'content': Bullet points slide
                       - 'two_column': Two column comparison
                       - 'section_divider': Section separator
                       - 'quote': Quote/reference slide
                       - 'thank_you': Closing/thank you slide
                       - 'metrics': KPI/metrics display (2-4 items)
                       - 'timeline': Timeline/milestones (max 5)
                       - 'comparison': Side-by-side comparison
                       - 'statement': Large centered text
                       - 'image_highlight': Image with caption
                       - 'flow': Process flow diagram (max 6 steps)
            **content: Slide-specific content (text, bullets, items, etc.)
        """
        self.slides.append({"type": slide_type, **content})

    def generate(
        self,
        output_path: str,
        title: str = "Presentation",
        theme: str = "executive",
        theme_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate PPTX file using structured theme configuration.

        Args:
            output_path: Output .pptx file path (or filename if workspace provided)
            title: Presentation title
            theme: Preset theme name (executive, ocean, minimal, warm, forest, coral)
            theme_config: Structured theme configuration (overrides preset theme)

        Returns:
            Dictionary with success/error status
        """
        # Determine final output path
        output_file = Path(output_path)
        if self._workspace:
            # If workspace is available and path is relative, save to output directory
            if not output_file.is_absolute():
                output_file = self._workspace.output_dir / output_file
                logger.info(f"Saving generated PPTX to workspace output: {output_file}")
            else:
                logger.info(f"Saving generated PPTX to absolute path: {output_file}")
        else:
            # No workspace, use provided path as-is or relative to cwd
            if not output_file.is_absolute():
                output_file = Path.cwd() / output_file
                logger.info(f"Saving generated PPTX to cwd: {output_file}")

        # Ensure parent directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Validate theme_config if provided
        if theme_config is not None:
            errors = _validate_theme_config(theme_config)
            if errors:
                return {"error": "; ".join(errors), "success": False}

        # Process theme configuration
        if theme_config:
            # Use structured configuration
            self.theme_config = theme_config
        else:
            # Convert preset theme to config
            self.theme_config = _preset_to_config(theme)

        # Find node executable and set up environment
        import os
        import shutil

        node_path = shutil.which("node")
        if not node_path:
            raise RuntimeError(
                f"Node.js not found in PATH. "
                f"Current PATH: {os.environ.get('PATH', 'empty')}. "
                f"Please install Node.js or add it to PATH."
            )

        # Find global node_modules path for NODE_PATH
        npm_root = subprocess.run(
            ["npm", "root", "-g"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        logger.debug(f"Using Node.js at: {node_path}, NODE_PATH: {npm_root}")

        # Prepare environment with NODE_PATH
        env = os.environ.copy()
        env["NODE_PATH"] = npm_root

        # Build JavaScript with theme configuration
        js_script = self._build_js_script(str(output_file), self.theme_config)

        # Write JS script to temp file and execute
        # IMPORTANT: Keep temp_dir alive until subprocess completes
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            js_file = temp_path / "generate.js"
            js_file.write_text(js_script, encoding="utf-8")

            # Call node to execute the script
            try:
                result = subprocess.run(
                    [node_path, str(js_file)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=temp_path,
                    env=env,
                )

                # Check for errors
                if result.returncode != 0:
                    error_msg = (
                        f"Node.js execution failed (exit code {result.returncode})\n"
                    )
                    error_msg += f"stderr: {result.stderr}\n"
                    error_msg += f"stdout: {result.stdout}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Find the generated PPTX file
                pptx_files = list(temp_path.glob("*.pptx"))

                if not pptx_files:
                    raise RuntimeError(
                        f"PPTX file not generated. Node output: {result.stdout}"
                    )

                pptx_temp_path = pptx_files[0]

                # Copy to final destination
                shutil.copy2(pptx_temp_path, output_file)

                logger.info(f"Generated PPTX: {output_file}")

                return {
                    "success": True,
                    "output": str(output_file),
                    "saved_to_workspace": self._workspace is not None,
                }

            except subprocess.TimeoutExpired:
                raise RuntimeError("Node.js execution timed out after 60 seconds")
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Node.js not found or failed to execute at {node_path}. "
                    f"Current PATH: {os.environ.get('PATH', 'empty')}. "
                    f"Error: {e}"
                )

    def _get_color(self, colors: Dict[str, str], key: str, default: str) -> str:
        """Get color from theme config with fallback."""
        return colors.get(key, default)

    def _get_size(self, typography: Dict[str, Any], key: str, default: Any) -> Any:
        """Get size from theme config with fallback."""
        return typography.get(key, default)

    def _build_js_script(self, output_path: str, theme_config: Dict[str, Any]) -> str:
        """Build JavaScript code that calls pptxgenjs with theme configuration.

        Args:
            output_path: Output file path (for filename generation)
            theme_config: Structured theme configuration

        Returns:
            JavaScript code as string
        """
        # Extract theme configuration
        colors = theme_config.get("colors", {})
        typography = theme_config.get("typography", {})
        layout = theme_config.get("layout", {})
        visual = theme_config.get("visual", {})

        # Build complete JS script with theme-based styling
        output_name = Path(output_path).name

        # Start with basic setup
        js_code = f"""const PptxGenJS = require('pptxgenjs');

const pres = new PptxGenJS();

pres.author = 'Generated by xagent';
pres.title = '{self.title}';

// Dynamic font size estimation based on text length and character type
const estimateFontSize = (text, baseSize, minSize = 12) => {{
  // Calculate effective length considering Chinese characters are wider
  let effectiveLength = 0;
  for (const char of text) {{
    // Chinese characters weight 1.8, others weight 1.0
    effectiveLength += /[\u4e00-\u9fa5]/.test(char) ? 1.8 : 1.0;
  }}

  if (effectiveLength <= 15) return baseSize;
  if (effectiveLength <= 30) return Math.max(baseSize * 0.7, minSize);
  if (effectiveLength <= 50) return Math.max(baseSize * 0.55, minSize);
  return Math.max(baseSize * 0.4, minSize);
}};

// Theme-based styling helper functions
const addBackground = (slide) => {{
  slide.background = {{ color: '{colors.get("background", "#FFFFFF")}' }};
}};

const addTitleBar = (slide, title) => {{
  const surfaceColor = '{colors.get("surface", colors.get("primary", "#F5F5F5"))}';
  const textColor = '{colors.get("text", "#000000")}';
  const baseTitleSize = {typography.get("title_size", 36)};
  const titleSize = estimateFontSize(title, baseTitleSize);

  slide.addShape('rect', {{
    x: 0, y: 0, w: '100%', h: 1.2,
    fill: {{ color: surfaceColor }},
    line: {{ type: 'none' }}
  }});
  slide.addText(title, {{
    x: 0.5, y: 0.35, w: 9, h: 0.7,
    fontSize: titleSize, bold: true,
    color: textColor
  }});
}};

const addAccentLine = (slide, y) => {{
  const primaryColor = '{colors.get("primary", "#000000")}';

  slide.addShape('rect', {{
    x: 0, y: y, w: '100%', h: 0.08,
    fill: {{ color: primaryColor }},
    line: {{ type: 'none' }}
  }});
}};

const addNumberedBullet = (slide, num, text, x, y) => {{
  const secondaryColor = '{colors.get("secondary", "#CCCCCC")}';
  const textColor = '{colors.get("text", "#000000")}';
  const accentColor = '{colors.get("accent", "#FFFFFF")}';
  const bodySize = {typography.get("body_size", 18)};

  // Number badge
  slide.addShape('rect', {{
    x: x, y: y + 0.05, w: 0.4, h: 0.4,
    fill: {{ color: secondaryColor }},
    line: {{ type: 'none' }}
  }});
  slide.addText(num.toString(), {{
    x: x, y: y, w: 0.4, h: 0.5,
    fontSize: bodySize - 4, bold: true,
    color: textColor,
    align: 'center', valign: 'middle'
  }});

  // Text
  slide.addText(text, {{
    x: x + 0.6, y: y + 0.05, w: 8.4, h: 0.4,
    fontSize: bodySize - 2,
    color: accentColor
  }});
}};

"""

        # Add each slide
        for slide in self.slides:
            slide_type = slide["type"]
            slide_data = self._format_slide_data(
                slide_type, slide, colors, typography, layout, visual
            )
            js_code += slide_data["master"]

        # Close with file write
        js_code += f"""
// Export to file
pres.writeFile({{ fileName: '{output_name}' }});

console.log('Presentation generated successfully!');
"""

        return js_code

    def _format_slide_data(
        self,
        slide_type: str,
        slide: Dict[str, Any],
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format slide data for JavaScript generation.

        Returns:
            Formatted slide data dictionary with 'master' field containing JS code
        """
        formatted = {"type": slide_type}

        # Route to appropriate slide builder
        if slide_type == "title":
            formatted["title"] = slide.get("title", "Title")
            if "subtitle" in slide:
                formatted["subtitle"] = slide["subtitle"]
            formatted["master"] = self._build_title_slide_js(
                colors, typography, layout, visual, slide
            )

        elif slide_type == "content":
            formatted["title"] = slide.get("title", "Content")
            if "bullets" in slide:
                formatted["bullets"] = slide["bullets"][:5]
            formatted["master"] = self._build_content_slide_js(
                colors, typography, layout, visual, slide
            )

        elif slide_type == "two_column":
            formatted["title"] = slide.get("title", "Two Column")
            if "left" in slide:
                formatted["left"] = slide["left"][:5]
            if "right" in slide:
                formatted["right"] = slide["right"][:5]
            formatted["master"] = self._build_two_column_slide_js(
                colors, typography, layout, visual, slide
            )

        elif slide_type == "section_divider":
            formatted["master"] = self._build_section_divider_js(colors, layout)

        elif slide_type == "quote":
            formatted["text"] = slide.get("text", "Quote")
            formatted["master"] = self._build_quote_slide_js(
                colors, typography, layout, slide
            )

        elif slide_type == "thank_you":
            formatted["text"] = slide.get("message", "Thank You")
            formatted["master"] = self._build_thank_you_slide_js(
                colors, typography, layout, visual, slide
            )

        # New enterprise slide types
        elif slide_type == "metrics":
            formatted["title"] = slide.get("title", "")
            if "items" in slide:
                formatted["items"] = slide["items"][:4]
            formatted["master"] = self._build_metrics_slide_js(
                colors, typography, layout, visual, slide
            )

        elif slide_type == "timeline":
            formatted["title"] = slide.get("title", "Timeline")
            if "milestones" in slide:
                formatted["milestones"] = slide["milestones"][:5]
            formatted["master"] = self._build_timeline_slide_js(
                colors, typography, layout, visual, slide
            )

        elif slide_type == "comparison":
            formatted["title"] = slide.get("title", "Comparison")
            formatted["left_title"] = slide.get("left_title", "Left")
            formatted["right_title"] = slide.get("right_title", "Right")
            if "left_items" in slide:
                formatted["left_items"] = slide["left_items"][:5]
            if "right_items" in slide:
                formatted["right_items"] = slide["right_items"][:5]
            formatted["master"] = self._build_comparison_slide_js(
                colors, typography, layout, visual, slide
            )

        elif slide_type == "statement":
            formatted["text"] = slide.get("text", "")
            formatted["master"] = self._build_statement_slide_js(
                colors, typography, layout, visual, slide
            )

        elif slide_type == "image_highlight":
            formatted["title"] = slide.get("title", "")
            formatted["caption"] = slide.get("caption", "")
            if "image_path" in slide:
                formatted["image_path"] = slide["image_path"]
            formatted["master"] = self._build_image_highlight_slide_js(
                colors, typography, layout, visual, slide
            )

        elif slide_type == "flow":
            formatted["title"] = slide.get("title", "Process")
            if "steps" in slide:
                formatted["steps"] = slide["steps"][:6]
            formatted["master"] = self._build_flow_slide_js(
                colors, typography, layout, visual, slide
            )

        else:
            # Default to content slide
            formatted["master"] = self._build_content_slide_js(
                colors, typography, layout, visual, slide
            )

        return formatted

    def _build_title_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for title slide using theme-based styling."""
        title = slide.get("title", "Title")
        subtitle = slide.get("subtitle", "")

        # Style from theme config
        bg_color = colors.get("background", "#FFFFFF")
        primary_color = colors.get("primary", "#000000")
        accent_color = colors.get("accent", "#FFFFFF")
        text_color = colors.get("text", "#000000")
        title_size = typography.get("title_size", 44)
        body_size = typography.get("body_size", 18)

        js_lines = [
            "var _slide = pres.addSlide();",
            f"_slide.background = {{ color: '{bg_color}' }};",
            f"_slide.addShape('rect', {{ x: 0, y: 0, w: '100%', h: 3.5, fill: {{ color: '{primary_color}' }}, line: {{ type: 'none' }} }});",
            # Use dynamic font size for title
            f"const titleSize = estimateFontSize({json.dumps(title, ensure_ascii=False)}, {title_size});",
            f"_slide.addText({json.dumps(title, ensure_ascii=False)}, {{ x: 0.5, y: 1.5, w: 9, h: 1.5, fontSize: titleSize, bold: true, color: '{text_color}', align: 'center', valign: 'middle' }});",
        ]

        if subtitle:
            js_lines.append(
                f"_slide.addText({json.dumps(subtitle, ensure_ascii=False)}, {{ x: 0.5, y: 3.2, w: 9, h: 0.8, fontSize: {body_size + 2}, color: '{accent_color}', align: 'center' }});"
            )

        # Decorative bottom bar
        secondary_color = colors.get("secondary", "#CCCCCC")
        js_lines.append(
            f"_slide.addShape('rect', {{ x: 0, y: 4.2, w: '100%', h: 0.1, fill: {{ color: '{secondary_color}' }}, line: {{ type: 'none' }} }});"
        )

        js_lines.append("")
        return "\n".join(js_lines) + "\n"

    def _build_content_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for content slide using theme-based styling."""
        title = slide.get("title", "Content")
        bullets = slide.get("bullets", [])[:5]

        # Generate slide using helper functions (all styles from theme config)
        js_lines = [
            "var _slide = pres.addSlide();",
            "addBackground(_slide);",
            f"addTitleBar(_slide, {json.dumps(title, ensure_ascii=False)});",
            "addAccentLine(_slide, 1.2);",
        ]

        # Add bullets
        for i, bullet in enumerate(bullets):
            y_pos = 1.6 + i * 0.6
            js_lines.append(
                f"addNumberedBullet(_slide, {i + 1}, {json.dumps(bullet, ensure_ascii=False)}, 0.5, {y_pos});"
            )

        js_lines.append("")
        return "\n".join(js_lines) + "\n"

    def _build_two_column_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for two column slide using theme-based styling."""
        title = slide.get("title", "Two Column")
        left_items = slide.get("left", [])[:5]
        right_items = slide.get("right", [])[:5]

        # Generate using helper functions
        js_lines = [
            "var _slide = pres.addSlide();",
            "addBackground(_slide);",
            f"addTitleBar(_slide, {json.dumps(title, ensure_ascii=False)});",
            "addAccentLine(_slide, 1.2);",
        ]

        # Add column divider
        secondary_color = colors.get("secondary", "#CCCCCC")
        js_lines.append(
            f"_slide.addShape('rect', {{ x: 4.95, y: 1.5, w: 0.1, h: 5, fill: {{ color: '{secondary_color}' }}, line: {{ type: 'none' }} }});"
        )

        # Left column items
        for i, item in enumerate(left_items):
            y_pos = 1.6 + i * 0.6
            js_lines.append(
                f"addNumberedBullet(_slide, {i + 1}, {json.dumps(item, ensure_ascii=False)}, 0.5, {y_pos});"
            )

        # Right column items
        for i, item in enumerate(right_items):
            y_pos = 1.6 + i * 0.6
            js_lines.append(
                f"addNumberedBullet(_slide, {i + 1}, {json.dumps(item, ensure_ascii=False)}, 5.5, {y_pos});"
            )

        js_lines.append("")
        return "\n".join(js_lines) + "\n"

    def _build_section_divider_js(
        self, colors: Dict[str, str], layout: Dict[str, Any]
    ) -> str:
        """Build JavaScript for section divider slide."""
        return "let slide = pres.addSlide();\n"

    def _build_quote_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for quote slide."""
        text = slide.get("text", "Quote")
        quote_size = typography.get("quote_size", 24)

        js_lines = []
        js_lines.append("var _slide = pres.addSlide();")
        # Use dynamic font size for quote text
        js_lines.append(
            f"const quoteSize = estimateFontSize({json.dumps(text, ensure_ascii=False)}, {quote_size}, 18);"
        )
        js_lines.append(
            f"_slide.addText({json.dumps(text, ensure_ascii=False)}, {{ x: 1, y: 2.5, w: 8, h: 2, fontSize: quoteSize, bold: true, color: {json.dumps(colors.get('text', '#1E2761'), ensure_ascii=False)}, align: 'center' }});"
        )
        js_lines.append("")
        return "\n".join(js_lines) + "\n"

    def _build_thank_you_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for thank you slide."""
        text = slide.get("message", "Thank You!")
        title_size = typography.get("title_size", 44)

        js_lines = []
        js_lines.append("var _slide = pres.addSlide();")
        # Use dynamic font size for thank you message
        js_lines.append(
            f"const tySize = estimateFontSize({json.dumps(text, ensure_ascii=False)}, {title_size}, 24);"
        )
        js_lines.append(
            f"_slide.addText({json.dumps(text, ensure_ascii=False)}, {{ x: 1, y: 2.5, w: 8, h: 2, fontSize: tySize, bold: true, color: {json.dumps(colors.get('text', '#1E2761'), ensure_ascii=False)}, align: 'center' }});"
        )
        js_lines.append("")
        return "\n".join(js_lines) + "\n"

    def _build_metrics_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for metrics/KPI slide."""
        title = slide.get("title", "")
        items = slide.get("items", [])[:4]

        js_lines = []
        text_color = colors.get("text", "#1E2761")

        js_lines.append("var _slide_metrics = pres.addSlide();")

        if title:
            js_lines.append(
                f"_slide_metrics.addText({json.dumps(title, ensure_ascii=False)}, {{ x: 0.5, y: 0.5, w: 9, h: 0.5, fontSize: 18, bold: true, color: {json.dumps(text_color, ensure_ascii=False)} }});"
            )

        for i, item in enumerate(items):
            label = item.get("label", f"Metric {i + 1}")
            value = item.get("value", "N/A")
            x_pos = 0.5 + (i % 2) * 4.5
            y_val = 1.5 + (i // 2) * 2.5
            y_label = 2.8 + (i // 2) * 2.5

            js_lines.append(
                f"_slide_metrics.addText({json.dumps(value, ensure_ascii=False)}, {{ x: {x_pos}, y: {y_val}, w: 4, h: 1.5, fontSize: 32, bold: true, color: {json.dumps(text_color, ensure_ascii=False)}, align: 'center' }});"
            )
            # Use dynamic font size for metric labels
            js_lines.append(
                f"const metric{i}LabelSize = estimateFontSize({json.dumps(label, ensure_ascii=False)}, 14, 10);"
            )
            js_lines.append(
                f"_slide_metrics.addText({json.dumps(label, ensure_ascii=False)}, {{ x: {x_pos}, y: {y_label}, w: 4, h: 0.5, fontSize: metric{i}LabelSize, color: {json.dumps(text_color, ensure_ascii=False)}, align: 'center' }});"
            )

        js_lines.append("")

        return "\n".join(js_lines) + "\n"

    def _build_timeline_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for timeline slide."""
        milestones = slide.get("milestones", [])[:4]

        js_lines = []
        text_color = colors.get("text", "#1E2761")

        js_lines.append("var _slide_timeline = pres.addSlide();")
        js_lines.append("")

        for i, ms in enumerate(milestones):
            ms_title = ms.get("title", f"Milestone {i + 1}")
            ms_desc = ms.get("description", "")
            x_pos = 1 + i * 2.2

            js_lines.append(f"// {ms_title}")
            # Use dynamic font size for milestone titles
            js_lines.append(
                f"const ms{i}TitleSize = estimateFontSize({json.dumps(ms_title, ensure_ascii=False)}, 16, 12);"
            )
            js_lines.append(
                f"_slide_timeline.addText({json.dumps(ms_title, ensure_ascii=False)}, {{ x: {x_pos}, y: 1.5, w: 2, h: 0.5, fontSize: ms{i}TitleSize, bold: true, color: {json.dumps(text_color, ensure_ascii=False)} }});"
            )
            # Use dynamic font size for milestone descriptions
            js_lines.append(
                f"const ms{i}DescSize = estimateFontSize({json.dumps(ms_desc, ensure_ascii=False)}, 11, 9);"
            )
            js_lines.append(
                f"_slide_timeline.addText({json.dumps(ms_desc, ensure_ascii=False)}, {{ x: {x_pos}, y: 2.1, w: 2, h: 2.5, fontSize: ms{i}DescSize, color: {json.dumps(text_color, ensure_ascii=False)} }});"
            )
            # Note: connector lines removed due to API compatibility issues
            js_lines.append("")

        return "\n".join(js_lines) + "\n"

    def _build_comparison_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for comparison slide."""
        left_title = slide.get("left_title", "Left")
        right_title = slide.get("right_title", "Right")
        left_items = slide.get("left_items", [])[:5]
        right_items = slide.get("right_items", [])[:5]

        js_lines = []
        text_color = colors.get("text", "#1E2761")

        js_lines.append("var _slide_comparison = pres.addSlide();")
        # Use dynamic font size for comparison titles
        js_lines.append(
            f"const leftTitleSize = estimateFontSize({json.dumps(left_title, ensure_ascii=False)}, 18, 14);"
        )
        js_lines.append(
            f"const rightTitleSize = estimateFontSize({json.dumps(right_title, ensure_ascii=False)}, 18, 14);"
        )
        js_lines.append(
            f"_slide_comparison.addText({json.dumps(left_title, ensure_ascii=False)}, {{ x: 0.5, y: 1, w: 4, h: 0.5, fontSize: leftTitleSize, bold: true, color: {json.dumps(text_color, ensure_ascii=False)} }});"
        )
        js_lines.append(
            f"_slide_comparison.addText({json.dumps(right_title, ensure_ascii=False)}, {{ x: 5.5, y: 1, w: 4, h: 0.5, fontSize: rightTitleSize, bold: true, color: {json.dumps(text_color, ensure_ascii=False)} }});"
        )
        js_lines.append(
            f"_slide_comparison.addText('VS', {{ x: 4.6, y: 2, w: 0.8, h: 0.5, fontSize: 14, bold: true, color: {json.dumps(colors.get('accent', '#FFFFFF'), ensure_ascii=False)}, align: 'center' }});"
        )
        js_lines.append("")

        for i, item in enumerate(left_items):
            y_pos = 2 + i * 0.6
            # Use dynamic font size for comparison items
            js_lines.append(
                f"const leftItem{i}Size = estimateFontSize({json.dumps(item, ensure_ascii=False)}, 14, 11);"
            )
            js_lines.append(
                f"_slide_comparison.addText({json.dumps(item, ensure_ascii=False)}, {{ x: 0.5, y: {y_pos}, w: 4, h: 0.5, fontSize: leftItem{i}Size, color: {json.dumps(text_color, ensure_ascii=False)} }});"
            )

        js_lines.append("")

        for i, item in enumerate(right_items):
            y_pos = 2 + i * 0.6
            # Use dynamic font size for comparison items
            js_lines.append(
                f"const rightItem{i}Size = estimateFontSize({json.dumps(item, ensure_ascii=False)}, 14, 11);"
            )
            js_lines.append(
                f"_slide_comparison.addText({json.dumps(item, ensure_ascii=False)}, {{ x: 5.5, y: {y_pos}, w: 4, h: 0.5, fontSize: rightItem{i}Size, color: {json.dumps(text_color, ensure_ascii=False)} }});"
            )

        js_lines.append("")

        return "\n".join(js_lines) + "\n"

    def _build_statement_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for statement slide."""
        text = slide.get("text", "")
        text_color = colors.get("text", "#1E2761")

        return f"""var _slide_statement = pres.addSlide();
const statementSize = estimateFontSize({json.dumps(text, ensure_ascii=False)}, 36, 24);
_slide_statement.addText({json.dumps(text, ensure_ascii=False)}, {{
  x: 1,
  y: 2.5,
  w: 8,
  h: 2,
  fontSize: statementSize,
  bold: true,
  color: {json.dumps(text_color, ensure_ascii=False)},
  align: 'center',
  valign: 'middle'
}});

"""

    def _build_image_highlight_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for image highlight slide."""
        title = slide.get("title", "")
        caption = slide.get("caption", "")
        text_color = colors.get("text", "#1E2761")

        return f"""var _slide_image = pres.addSlide();
const imgTitleSize = estimateFontSize({json.dumps(title, ensure_ascii=False)}, 24, 18);
_slide_image.addText({json.dumps(title, ensure_ascii=False)}, {{
  x: 0.5,
  y: 0.5,
  w: 9,
  h: 0.5,
  fontSize: imgTitleSize,
  bold: true,
  color: {json.dumps(text_color, ensure_ascii=False)}
}});
_slide_image.addText({json.dumps(caption, ensure_ascii=False)}, {{
  x: 0.5,
  y: 5.5,
  w: 9,
  h: 0.5,
  fontSize: 14,
  color: {json.dumps(text_color, ensure_ascii=False)},
  align: 'center'
}});

"""

    def _build_flow_slide_js(
        self,
        colors: Dict[str, str],
        typography: Dict[str, Any],
        layout: Dict[str, Any],
        visual: Dict[str, Any],
        slide: Dict[str, Any],
    ) -> str:
        """Build JavaScript for flow/process slide."""
        steps = slide.get("steps", [])[:4]

        js_lines = []
        text_color = colors.get("text", "#1E2761")

        js_lines.append("var _slide_flow = pres.addSlide();")
        js_lines.append("")

        for i, step in enumerate(steps):
            x = 1.5 + i * 2.2

            js_lines.append(f"// Step {i + 1}")
            # Use dynamic font size for flow steps
            js_lines.append(
                f"const flowStep{i}Size = estimateFontSize({json.dumps(step, ensure_ascii=False)}, 14, 10);"
            )
            js_lines.append(
                f"_slide_flow.addText({json.dumps(step, ensure_ascii=False)}, {{ x: {x}, y: 2.5, w: 2, h: 1, fontSize: flowStep{i}Size, color: {json.dumps(text_color, ensure_ascii=False)}, align: 'center', valign: 'middle' }});"
            )
            # Note: arrows removed due to API compatibility issues
            js_lines.append("")

        return "\n".join(js_lines) + "\n"


def read_pptx(
    pptx_path: str,
    extract_text: bool = False,
    workspace: Optional[TaskWorkspace] = None,
) -> Dict[str, Any]:
    """Read PPTX file and extract information.

    Args:
        pptx_path: Path to .pptx file
        extract_text: If True, extracts all text content
        workspace: Optional workspace for resolving file paths

    Returns:
        Dictionary with slide information or extracted text
    """
    with PresentationReader(pptx_path, workspace=workspace) as reader:
        if extract_text:
            return {"text": reader.extract_text(), "success": True}
        else:
            result = reader.read_presentation()
            result["success"] = True
            return result


def generate_pptx(
    output_path: str,
    title: str = "Presentation",
    theme: str = "aurora",
    theme_config: Optional[Dict[str, Any]] = None,
    slide_contents: Optional[Union[list, str]] = None,
    workspace: Optional[TaskWorkspace] = None,
) -> Dict[str, Any]:
    """Generate PPTX file from slide definitions.

    Args:
        output_path: Output .pptx file path (or filename if workspace provided)
        title: Presentation title
        theme: Preset theme name (default: aurora)
        theme_config: Structured theme config (overrides preset)
        slide_contents: List of slide definition objects (contents of each slide)
        workspace: Optional workspace for saving generated files

    Returns:
        Dictionary with success/error status
    """
    gen = PresentationGenerator(workspace=workspace)
    gen.create(title)

    # Validate slide_contents parameter
    if slide_contents is not None:
        # Handle JSON string (common mistake from LLMs)
        if isinstance(slide_contents, str):
            try:
                import json

                slide_contents = json.loads(slide_contents)
                logger.info(
                    f"Parsed slide_contents from JSON string: {len(slide_contents)} slides"
                )
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid slide_contents parameter: JSON string could not be parsed. {str(e)}. "
                    "Pass slide_contents as a Python list object, not a JSON string.",
                }

        if not isinstance(slide_contents, list):
            return {
                "success": False,
                "error": f"Invalid slide_contents parameter: expected list of slide objects, got {type(slide_contents).__name__}. "
                "Pass slide_contents as an array: slide_contents=[{'type': 'content', 'title': 'Title', 'bullets': ['A', 'B']}]",
            }
        if not slide_contents:
            return {
                "success": False,
                "error": "slide_contents parameter cannot be empty. Provide at least one slide object.",
            }

        for slide in slide_contents:
            slide_type = slide.get("type", "content")
            slide_data = {k: v for k, v in slide.items() if k != "type"}
            gen.add_slide(slide_type, **slide_data)

    return gen.generate(
        output_path, title=title, theme=theme, theme_config=theme_config
    )


def unpack_pptx(
    pptx_path: str, output_dir: str, workspace: Optional[TaskWorkspace] = None
) -> Dict[str, Any]:
    """Unpack PPTX file to directory.

    Args:
        pptx_path: Path to .pptx file
        output_dir: Directory to extract files to (relative to workspace if workspace provided)
        workspace: Optional workspace for resolving paths

    Returns:
        Dictionary with success status and output directory
    """
    with PresentationReader(pptx_path, workspace=workspace) as reader:
        return reader.unpack(output_dir)


def pack_pptx(
    input_dir: str,
    output_path: str,
    validate: bool = True,
    workspace: Optional[TaskWorkspace] = None,
) -> Dict[str, Any]:
    """Pack directory into PPTX file.

    Args:
        input_dir: Directory containing unpacked PPTX files
        output_path: Output .pptx file path (or filename if workspace provided)
        validate: Whether to validate structure
        workspace: Optional workspace for resolving paths and saving output

    Returns:
        Dictionary with success status and output file path
    """
    # For packing, we need to point to a directory that's already unpacked
    # We create a reader that doesn't extract from a zip but uses the dir directly
    reader = PresentationReader(input_dir, workspace=workspace)
    # Manually set temp_dir to point to the unpacked directory
    reader.temp_dir = Path(input_dir)

    # Resolve output path
    output_file = Path(output_path)
    if workspace:
        if not output_file.is_absolute():
            output_file = workspace.output_dir / output_file

    return reader.pack(str(output_file), validate=validate)


def add_slide_pptx(
    unpacked_dir: str, source: str, workspace: Optional[TaskWorkspace] = None
) -> Dict[str, Any]:
    """Add slide to unpacked PPTX directory.

    Args:
        unpacked_dir: Path to unpacked PPTX directory
        source: Source slide file or layout
        workspace: Optional workspace for resolving paths

    Returns:
        Dictionary with success status and slide information
    """
    reader = PresentationReader(unpacked_dir, workspace=workspace)
    reader.temp_dir = Path(unpacked_dir)
    return reader.add_slide(source)


def clean_pptx(
    unpacked_dir: str, workspace: Optional[TaskWorkspace] = None
) -> Dict[str, Any]:
    """Clean orphaned files from unpacked PPTX directory.

    Args:
        unpacked_dir: Path to unpacked PPTX directory
        workspace: Optional workspace for resolving paths

    Returns:
        Dictionary with success status and count of removed files
    """
    reader = PresentationReader(unpacked_dir, workspace=workspace)
    reader.temp_dir = Path(unpacked_dir)
    return reader.clean()
