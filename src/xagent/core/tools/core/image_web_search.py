"""
Pure Image Web Search Tool
Standalone image search functionality without framework dependencies.
Supports Google Custom Search (default) and Tencent Cloud 联网图像搜索 (国内可用).
"""

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# 图像搜索提供商：google（默认，需翻墙）| tencent（腾讯云联网图像搜索，国内可用）
IMAGE_SEARCH_PROVIDER_ENV = "IMAGE_SEARCH_PROVIDER"


class ImageWebSearchCore:
    """Pure image web search tool without framework dependencies"""

    def __init__(self, save_directory: Optional[str] = None):
        """
        Initialize the image search tool.

        Args:
            save_directory: Directory to save downloaded images. Defaults to './downloads'
        """
        self.save_directory = (
            Path(save_directory) if save_directory else Path("./downloads")
        )
        self.save_directory.mkdir(parents=True, exist_ok=True)

    def _get_provider(self) -> str:
        """Return image search provider: 'google' or 'tencent' (lowercase)."""
        p = (os.getenv(IMAGE_SEARCH_PROVIDER_ENV) or "google").strip().lower()
        return p if p in ("google", "tencent") else "google"

    async def search_images(
        self,
        query: str,
        num_results: int = 5,
        image_size: str = "medium",
        image_type: str = "photo",
        save_images: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for images. Provider is chosen by env IMAGE_SEARCH_PROVIDER:
        - google: Google Custom Search API (default)
        - tencent: 腾讯云联网图像搜索 (国内可用，需 TENCENT_SECRET_ID / TENCENT_SECRET_KEY)

        Args:
            query: The image search query string
            num_results: Number of images to return (max 10 for Google, max 20 for Tencent)
            image_size: Image size (Google only): small, medium, large, ...
            image_type: Image type (Google only): photo, clipart, ...
            save_images: Whether to download and save images locally

        Returns:
            List of image results with metadata and local paths
        """
        provider = self._get_provider()
        logger.info(
            f"🔍 Starting image search for: '{query}' "
            f"(provider={provider}, results={num_results})"
        )

        if provider == "tencent":
            return await self._search_images_tencent(query, num_results, save_images)
        return await self._search_images_google(
            query, num_results, image_size, image_type, save_images
        )

    async def _search_images_tencent(
        self, query: str, num_results: int, save_images: bool
    ) -> List[Dict[str, Any]]:
        """腾讯云联网图像搜索（文搜图），国内可访问。需安装: pip install tencentcloud-sdk-python-wimgs"""
        secret_id = os.getenv("TENCENT_SECRET_ID") or os.getenv("TENCENTCLOUD_SECRET_ID")
        secret_key = os.getenv("TENCENT_SECRET_KEY") or os.getenv(
            "TENCENTCLOUD_SECRET_KEY"
        )
        if not secret_id or not secret_key:
            raise ValueError(
                "Missing Tencent Cloud credentials for image search. "
                "Set TENCENT_SECRET_ID and TENCENT_SECRET_KEY (or TENCENTCLOUD_*). "
                "Optional: pip install tencentcloud-sdk-python-wimgs"
            )

        num_results = min(max(1, num_results), 20)

        def _call_tencent() -> List[Dict[str, Any]]:
            try:
                from tencentcloud.common import credential
                from tencentcloud.common.profile.client_profile import ClientProfile
                from tencentcloud.wimgs.v20251106 import wimgs_client, models
            except ImportError as e:
                raise ValueError(
                    "Tencent image search requires tencentcloud-sdk-python-wimgs. "
                    "Install with: pip install tencentcloud-sdk-python-wimgs"
                ) from e

            cred = credential.Credential(secret_id, secret_key)
            profile = ClientProfile()
            profile.httpProfile.endpoint = "wimgs.tencentcloudapi.com"
            client = wimgs_client.WimgsClient(cred, "", profile)
            req = models.SearchByTextRequest()
            req.Query = query
            resp = client.SearchByText(req)
            if not resp.Images:
                return []
            out: List[Dict[str, Any]] = []
            for i, img_str in enumerate(resp.Images[:num_results], 1):
                try:
                    img = json.loads(img_str) if isinstance(img_str, str) else img_str
                except Exception:
                    continue
                out.append({
                    "title": img.get("title", ""),
                    "link": img.get("siteUrl", ""),
                    "snippet": img.get("title", ""),
                    "image_link": img.get("origPicUrl") or img.get("thumbnailUrl", ""),
                    "context_link": img.get("siteUrl", ""),
                    "height": img.get("origPicHeight") or img.get("thumbnailHeight", 0),
                    "width": img.get("origPicWidth") or img.get("thumbnailWidth", 0),
                    "file_format": "unknown",
                    "local_path": None,
                })
            return out

        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(None, _call_tencent)
        if not raw_results:
            logger.warning("⚠️ No image search results found (Tencent)")
            return []

        logger.info(f"📋 Found {len(raw_results)} image results (Tencent)")
        results: List[Dict[str, Any]] = []
        for i, result in enumerate(raw_results, 1):
            if save_images and result.get("image_link"):
                try:
                    result["local_path"] = await self._download_image(
                        result["image_link"], result["title"], i
                    )
                except Exception as e:
                    logger.warning(f"   Failed to download: {e}")
                    result["local_path"] = None
            results.append(result)
        logger.info(f"🎯 Search completed with {len(results)} results")
        return results

    async def _search_images_google(
        self,
        query: str,
        num_results: int,
        image_size: str,
        image_type: str,
        save_images: bool,
    ) -> List[Dict[str, Any]]:
        """Google Custom Search API（国内需代理）。"""
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")
        if not api_key or not cse_id:
            raise ValueError(
                "Missing required environment variables. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID. "
                "For China, you can use Tencent: set IMAGE_SEARCH_PROVIDER=tencent and TENCENT_SECRET_ID/TENCENT_SECRET_KEY."
            )

        num_results = min(max(1, num_results), 10)
        proxy_url = self._get_proxy_url()
        if proxy_url:
            logger.info(f"🌐 Using proxy: {proxy_url}")

        params: Dict[str, Any] = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": num_results,
            "searchType": "image",
            "imgSize": image_size,
            "imgType": image_type,
            "safe": "active",
        }

        try:
            client_kwargs: Dict[str, Any] = {}
            if proxy_url:
                client_kwargs["proxy"] = proxy_url
            logger.info("📡 Making request to Google Custom Search API...")
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params,
                    timeout=10,
                )
                if response.status_code == 403:
                    self._handle_403_error(response)
                response.raise_for_status()
                data = response.json()
            logger.info("✅ Google API request successful")
            return await self._process_search_results(data, save_images)
        except httpx.RequestError as e:
            logger.error(f"❌ Network error: {str(e)}")
            raise ValueError(f"Network error during image search: {str(e)}") from e
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            raise ValueError(f"Unexpected error during image search: {str(e)}") from e

    async def _process_search_results(
        self, data: Dict[str, Any], save_images: bool
    ) -> List[Dict[str, Any]]:
        """Process search results and optionally download images"""
        results: List[Dict[str, Any]] = []

        if "items" not in data:
            logger.warning("⚠️ No image search results found")
            return results

        logger.info(f"📋 Found {len(data['items'])} image results")

        for i, item in enumerate(data["items"], 1):
            result = {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "image_link": item.get("image", {}).get("thumbnailLink", ""),
                "context_link": item.get("image", {}).get("contextLink", ""),
                "height": item.get("image", {}).get("height", 0),
                "width": item.get("image", {}).get("width", 0),
                "file_format": item.get("image", {}).get("fileFormat", "unknown"),
                "local_path": None,
            }

            logger.info(f"🖼️  Image {i}: {result['title']}")
            logger.info(f"   URL: {result['link']}")
            logger.info(f"   Size: {result['width']}x{result['height']}")

            if save_images and result["image_link"]:
                try:
                    result["local_path"] = await self._download_image(
                        result["image_link"], result["title"], i
                    )
                    logger.info(f"   Saved to: {result['local_path']}")
                except Exception as e:
                    logger.warning(f"   Failed to download: {e}")
                    result["local_path"] = None

            results.append(result)

        logger.info(f"🎯 Search completed with {len(results)} results")
        return results

    async def _download_image(self, image_url: str, title: str, index: int) -> str:
        """Download image from URL and save to local directory using atomic write."""
        filename = self._generate_filename(title, index, image_url)
        save_path = self.save_directory / filename
        # Write to temp file first, then atomically rename to avoid partial reads
        temp_path = save_path.with_suffix(save_path.suffix + ".tmp")

        proxy_url = self._get_proxy_url()
        client_kwargs: Dict[str, Any] = {"timeout": 30}
        if proxy_url:
            client_kwargs["proxy"] = proxy_url

        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(image_url)
            response.raise_for_status()

            # Write to temp file first
            with open(temp_path, "wb") as f:
                f.write(response.content)
            # Atomic rename to prevent partial reads during write
            import os
            os.replace(temp_path, save_path)

        logger.info(f"Downloaded image to: {save_path}")
        return str(save_path)

    def _generate_filename(self, title: str, index: int, image_url: str) -> str:
        """Generate safe filename for downloaded image"""
        # Clean title for filename
        safe_title = "".join(
            c for c in title[:50] if c.isalnum() or c in ("-", "_", " ")
        ).strip()
        safe_title = safe_title.replace(" ", "_")

        # Extract extension from URL
        parsed_url = urlparse(image_url)
        extension = os.path.splitext(parsed_url.path)[1]

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".svg",
            ".tiff",
            ".tif",
        }

        if not extension or extension.lower() not in image_extensions:
            extension = ".jpg"

        return f"image_search_{index}_{safe_title}_{uuid.uuid4().hex[:8]}{extension}"

    def _get_proxy_url(self) -> Optional[str]:
        """Get proxy URL from environment variables"""
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        return https_proxy or http_proxy

    def _handle_403_error(self, response: httpx.Response) -> None:
        """Handle 403 Forbidden errors from Google API"""
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
            error_reason = (
                error_data.get("error", {})
                .get("errors", [{}])[0]
                .get("reason", "Unknown")
            )
            logger.error(
                f"❌ Google API 403 Error: {error_message} (reason: {error_reason})"
            )
            raise ValueError(
                f"Google API 403 Error: {error_message}\n"
                f"Reason: {error_reason}\n"
                f"This usually means:\n"
                f"- API quota exceeded\n"
                f"- Invalid API key\n"
                f"- Custom Search Engine ID is incorrect\n"
                f"- Custom Search API is not enabled\n"
                f"Please check your Google Cloud Console settings."
            )
        except Exception:
            logger.error("❌ Google API 403 Forbidden error")
            raise ValueError(
                "Google API 403 Forbidden error. This usually means:\n"
                "- API quota exceeded\n"
                "- Invalid API key\n"
                "- Custom Search Engine ID is incorrect\n"
                "- Custom Search API is not enabled\n"
                "Please check your Google Cloud Console settings."
            )


# Convenience function for direct usage
async def search_images(
    query: str,
    num_results: int = 5,
    image_size: str = "medium",
    image_type: str = "photo",
    save_images: bool = True,
    save_directory: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for images using Google Custom Search API.

    Args:
        query: The image search query string
        num_results: Number of images to return (max 10)
        image_size: Image size: small, medium, large, xlarge, xxlarge, huge
        image_type: Image type: photo, clipart, lineart, animated, transparent
        save_images: Whether to download and save images locally
        save_directory: Directory to save images. Defaults to './downloads'

    Returns:
        List of image results with metadata and local paths
    """
    searcher = ImageWebSearchCore(save_directory)
    return await searcher.search_images(
        query=query,
        num_results=num_results,
        image_size=image_size,
        image_type=image_type,
        save_images=save_images,
    )
