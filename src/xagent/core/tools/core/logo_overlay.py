"""
Pure Logo Overlay Tool
Standalone logo overlay functionality without framework dependencies
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, cast

import httpx
from PIL import Image

from ...utils.svg import rasterize_svg_bytes

logger = logging.getLogger(__name__)


class LogoOverlayCore:
    """Pure logo overlay tool without framework dependencies"""

    def __init__(self, output_directory: Optional[str] = None):
        """
        Initialize the logo overlay tool.

        Args:
            output_directory: Directory to save output images. Defaults to './output'
        """
        self.output_directory = (
            Path(output_directory) if output_directory else Path("./output")
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)

    async def overlay_logo(
        self,
        base_image_uri: str,
        logo_image_uri: str,
        position: str = "bottom-right",
        size_ratio: float = 0.2,
        opacity: float = 1.0,
        padding: int = 20,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Overlay a logo on a base image.

        Args:
            base_image_uri: Base image URI (local path or remote URL)
            logo_image_uri: Logo image URI (local path or remote URL)
            position: Logo position (top-left, top-right, bottom-left, bottom-right, center)
            size_ratio: Logo size relative to base image (0.1 to 0.5)
            opacity: Logo opacity (0.0 to 1.0)
            padding: Padding from edges in pixels
            output_filename: Custom output filename (without extension)

        Returns:
            Dictionary with success status, output path, and message
        """
        logger.info(
            f"🎨 Starting logo overlay: base={base_image_uri}, logo={logo_image_uri}, "
            f"position={position}, size_ratio={size_ratio}, opacity={opacity}"
        )

        # Validate parameters
        size_ratio = max(0.1, min(0.5, size_ratio))
        opacity = max(0.0, min(1.0, opacity))
        padding = max(0, min(100, padding))

        try:
            # Load base image
            base_image = await self._load_image(base_image_uri, "base image")
            if base_image is None:
                return {
                    "success": False,
                    "output_path": "",
                    "message": "Failed to load base image",
                    "error": "Base image could not be loaded",
                }

            # Load logo image
            logo_image = await self._load_image(logo_image_uri, "logo image")
            if logo_image is None:
                return {
                    "success": False,
                    "output_path": "",
                    "message": "Failed to load logo image",
                    "error": "Logo image could not be loaded",
                }

            # Process images
            result_image = self._process_logo_overlay(
                base_image, logo_image, position, size_ratio, opacity, padding
            )

            # Save result
            output_path = self._save_result_image(result_image, output_filename)

            logger.info(f"✅ Logo overlay completed successfully: {output_path}")
            return {
                "success": True,
                "output_path": output_path,
                "message": "Logo overlay completed successfully",
                "error": None,
            }

        except Exception as e:
            logger.error(f"❌ Error during logo overlay: {str(e)}")
            return {
                "success": False,
                "output_path": "",
                "message": "Logo overlay failed",
                "error": str(e),
            }

    async def _load_image(
        self, image_uri: str, image_type: str
    ) -> Optional[Image.Image]:
        """Load image from URI (local path or remote URL)"""
        try:
            if image_uri.startswith(("http://", "https://")):
                # Remote URL - download image
                logger.info(f"📥 Downloading {image_type} from: {image_uri}")
                return await self._download_image(image_uri)
            else:
                # Local path - load directly
                logger.info(f"📂 Loading {image_type} from: {image_uri}")
                return self._load_local_image(image_uri)

        except Exception as e:
            logger.error(f"❌ Failed to load {image_type} from {image_uri}: {str(e)}")
            return None

    async def _download_image(self, image_url: str) -> Image.Image:
        """Download image from URL"""
        proxy_url = self._get_proxy_url()
        client_kwargs: Dict[str, Any] = {"timeout": 30}
        if proxy_url:
            client_kwargs["proxy"] = proxy_url
            logger.info(f"🌐 Using proxy: {proxy_url}")

        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(image_url)
            response.raise_for_status()

            import io

            raw_image = response.content
            content_type = response.headers.get("content-type", "").split(";", 1)[0]
            if content_type == "image/svg+xml" or image_url.lower().endswith(".svg"):
                raw_image = rasterize_svg_bytes(raw_image)
            image_data = io.BytesIO(raw_image)
            image = cast(Image.Image, Image.open(image_data))
            image.load()  # Load image data

        # Convert to RGBA for transparency support
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        logger.info(f"✅ Image downloaded successfully: {image.size}")
        return image

    def _load_local_image(self, image_path: str) -> Image.Image:
        """Load image from local path"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Local file not found: {image_path}")

        if image_path.lower().endswith(".svg"):
            import io

            image = cast(
                Image.Image,
                Image.open(
                    io.BytesIO(rasterize_svg_bytes(Path(image_path).read_bytes()))
                ),
            )
        else:
            image = cast(Image.Image, Image.open(image_path))
        image.load()  # Load image data

        # Convert to RGBA for transparency support
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        logger.info(f"✅ Image loaded successfully: {image.size}")
        return image

    def _process_logo_overlay(
        self,
        base_image: Image.Image,
        logo_image: Image.Image,
        position: str,
        size_ratio: float,
        opacity: float,
        padding: int,
    ) -> Image.Image:
        """Process logo overlay on base image"""
        logger.info("🔄 Processing logo overlay...")

        # Calculate logo size based on base image
        base_width, base_height = base_image.size
        logo_width = int(base_width * size_ratio)

        # Maintain aspect ratio
        logo_aspect_ratio = logo_image.width / logo_image.height
        logo_height = int(logo_width / logo_aspect_ratio)

        # Resize logo
        logo_resized = logo_image.resize(
            (logo_width, logo_height), Image.Resampling.LANCZOS
        )

        # Apply opacity if needed
        if opacity < 1.0:
            logo_resized = self._apply_opacity(logo_resized, opacity)

        # Calculate position
        x, y = self._calculate_position(
            base_width, base_height, logo_width, logo_height, position, padding
        )

        # Create a copy of base image
        result_image = base_image.copy()

        # Paste logo (use logo as mask for transparency)
        result_image.paste(logo_resized, (x, y), logo_resized)

        logger.info(
            f"✅ Logo overlay processed: position=({x}, {y}), size={logo_width}x{logo_height}"
        )
        return result_image

    def _apply_opacity(self, image: Image.Image, opacity: float) -> Image.Image:
        """Apply opacity to image"""
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Split channels
        r, g, b, a = image.split()

        # Apply opacity to alpha channel
        a = a.point(lambda i: int(i * opacity))

        # Merge channels back
        return Image.merge("RGBA", (r, g, b, a))

    def _calculate_position(
        self,
        base_width: int,
        base_height: int,
        logo_width: int,
        logo_height: int,
        position: str,
        padding: int,
    ) -> tuple[int, int]:
        """Calculate logo position coordinates"""
        position_map = {
            "top-left": (padding, padding),
            "top-right": (base_width - logo_width - padding, padding),
            "bottom-left": (padding, base_height - logo_height - padding),
            "bottom-right": (
                base_width - logo_width - padding,
                base_height - logo_height - padding,
            ),
            "center": (
                (base_width - logo_width) // 2,
                (base_height - logo_height) // 2,
            ),
        }

        # Default to bottom-right if position not recognized
        x, y = position_map.get(
            position,
            (base_width - logo_width - padding, base_height - logo_height - padding),
        )

        return x, y

    def _save_result_image(
        self, image: Image.Image, output_filename: Optional[str] = None
    ) -> str:
        """Save result image to output directory"""
        # Generate filename
        if output_filename:
            safe_filename = "".join(
                c for c in output_filename[:50] if c.isalnum() or c in ("-", "_", " ")
            ).strip()
            safe_filename = safe_filename.replace(" ", "_")
        else:
            safe_filename = f"logo_overlay_{uuid.uuid4().hex[:8]}"

        # Save with different formats based on image mode
        if image.mode == "RGBA":
            # Save as PNG for transparency
            output_path = self.output_directory / f"{safe_filename}.png"
            image.save(output_path, "PNG")
        else:
            # Save as JPG for non-transparent images
            output_path = self.output_directory / f"{safe_filename}.jpg"
            image.convert("RGB").save(output_path, "JPEG", quality=95)

        logger.info(f"💾 Image saved to: {output_path}")
        return str(output_path)

    def _get_proxy_url(self) -> Optional[str]:
        """Get proxy URL from environment variables"""
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        return https_proxy or http_proxy


# Convenience function for direct usage
async def overlay_logo(
    base_image_uri: str,
    logo_image_uri: str,
    position: str = "bottom-right",
    size_ratio: float = 0.2,
    opacity: float = 1.0,
    padding: int = 20,
    output_filename: Optional[str] = None,
    output_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Overlay a logo on a base image.

    Args:
        base_image_uri: Base image URI (local path or remote URL)
        logo_image_uri: Logo image URI (local path or remote URL)
        position: Logo position (top-left, top-right, bottom-left, bottom-right, center)
        size_ratio: Logo size relative to base image (0.1 to 0.5)
        opacity: Logo opacity (0.0 to 1.0)
        padding: Padding from edges in pixels
        output_filename: Custom output filename (without extension)
        output_directory: Directory to save output. Defaults to './output'

    Returns:
        Dictionary with success status, output path, and message
    """
    overlay_tool = LogoOverlayCore(output_directory)
    return await overlay_tool.overlay_logo(
        base_image_uri=base_image_uri,
        logo_image_uri=logo_image_uri,
        position=position,
        size_ratio=size_ratio,
        opacity=opacity,
        padding=padding,
        output_filename=output_filename,
    )
