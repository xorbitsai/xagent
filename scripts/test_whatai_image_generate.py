#!/usr/bin/env python3
"""
Test OpenAI-compatible image generation endpoint (e.g., whatai.cc).

Reads config from .env / environment:
  - IMAGE_GENERATION_BASE_URL (e.g., https://api.whatai.cc)
  - IMAGE_GENERATION_API_KEY  (sk-...)
  - IMAGE_GENERATION_MODEL_NAME (e.g., gemini-3.1-flash-image-preview)

This script prints:
  - HTTP status / content-type
  - First part of raw response text (even if not JSON)
  - Parsed JSON keys (if JSON)
  - Whether it contains image URL or base64 payload
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional, Tuple

import httpx
from dotenv import load_dotenv


def _norm_base_url(raw: str) -> str:
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        return raw
    # OpenAI compatible base is usually .../v1
    if not raw.endswith("/v1"):
        raw = f"{raw}/v1"
    return raw


def _try_parse_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
        return None, f"JSON is not an object (type={type(obj).__name__})"
    except Exception as e:
        return None, str(e)


def _extract_image_from_openai_images(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    # OpenAI Images API: { data: [ {url: "..."} ] } OR { data: [ {b64_json: "..."} ] }
    data = obj.get("data")
    if not isinstance(data, list) or not data:
        return None, "Missing/empty 'data' field"
    item = data[0]
    if not isinstance(item, dict):
        return None, "data[0] is not an object"
    if isinstance(item.get("url"), str) and item["url"]:
        return item["url"], None
    if isinstance(item.get("b64_json"), str) and item["b64_json"]:
        return f"data:image/png;base64,{item['b64_json']}", None
    return None, "data[0] contains neither 'url' nor 'b64_json'"


def main() -> None:
    # Load project .env if present
    load_dotenv()

    base_url = _norm_base_url(os.getenv("IMAGE_GENERATION_BASE_URL", ""))
    api_key = (os.getenv("IMAGE_GENERATION_API_KEY") or "").strip()
    model = (os.getenv("IMAGE_GENERATION_MODEL_NAME") or "").strip() or "dall-e-3"

    if not base_url or not api_key:
        raise SystemExit(
            "Missing env. Please set IMAGE_GENERATION_BASE_URL and IMAGE_GENERATION_API_KEY."
        )

    # Common OpenAI-compatible endpoint
    url = f"{base_url}/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": "A cute orange tabby cat sitting on a sofa, high quality, realistic photo.",
        "size": "1024x1024",
    }

    print("== whatai image generation test ==")
    print(f"URL: {url}")
    print(f"Model: {model}")

    # IMPORTANT: do not print api_key
    timeout = httpx.Timeout(60.0, connect=10.0)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.post(url, headers=headers, json=payload)

    ct = resp.headers.get("content-type", "")
    print(f"HTTP: {resp.status_code}")
    print(f"Content-Type: {ct}")

    # Always show raw text prefix to catch HTML / empty body / gateway errors
    raw = resp.text or ""
    print("-- raw response (first 800 chars) --")
    print(raw[:800])

    if not raw.strip():
        print("-- diagnosis --")
        print("Empty response body. This often indicates a proxy/gateway/network issue upstream.")
        raise SystemExit(2)

    obj, err = _try_parse_json(raw)
    if obj is None:
        print("-- diagnosis --")
        print(f"Response is not JSON. json.loads error: {err}")
        raise SystemExit(3)

    print("-- parsed json keys --")
    print(sorted(list(obj.keys())))

    if resp.status_code >= 400:
        print("-- error object --")
        print(json.dumps(obj, ensure_ascii=False, indent=2)[:2000])
        raise SystemExit(4)

    image, image_err = _extract_image_from_openai_images(obj)
    if image is None:
        print("-- diagnosis --")
        print(f"JSON parsed but no image found: {image_err}")
        print(json.dumps(obj, ensure_ascii=False, indent=2)[:2000])
        raise SystemExit(5)

    if image.startswith("data:image/") and "base64," in image:
        b64 = image.split("base64,", 1)[1]
        try:
            _ = base64.b64decode(b64[:100], validate=False)
            print("Image returned as base64 data URL (preview ok).")
        except Exception:
            print("Image returned as base64 data URL (could not decode preview).")
    else:
        print(f"Image URL: {image}")

    print("OK")


if __name__ == "__main__":
    main()

