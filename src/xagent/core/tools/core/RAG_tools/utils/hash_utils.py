"""Hash utility functions for RAG tools.

This module provides common hash computation functions including
content hashing and hash validation.
"""

import hashlib
import json
from typing import Any, Dict, Optional


def compute_content_hash(content: bytes, algorithm: str = "sha256") -> str:
    """Compute hash from content bytes.


    Args:
        content: Content bytes to hash.
        algorithm: Hash algorithm to use (default: sha256).

    Returns:
        Hexadecimal string of the computed hash.


    Raises:
        ValueError: If unsupported algorithm is specified.
    """
    if algorithm not in hashlib.algorithms_guaranteed:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hash_obj = hashlib.new(algorithm)
    hash_obj.update(content)
    return hash_obj.hexdigest()


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Compute hash from file path.


    Args:
        file_path: Path to the file to hash.
        algorithm: Hash algorithm to use (default: sha256).

    Returns:
        Hexadecimal string of the computed hash.


    Raises:
        ValueError: If unsupported algorithm is specified.
        IOError: If file cannot be read.
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        return compute_content_hash(content, algorithm)
    except IOError as e:
        raise IOError(f"Failed to read file for hashing: {e}") from e


def validate_hash_format(hash_str: str, expected_length: Optional[int] = None) -> bool:
    """Validate hash string format.


    Args:
        hash_str: Hash string to validate.
        expected_length: Expected length of hash (optional).

    Returns:
        True if hash format is valid.


    Raises:
        ValueError: If hash format is invalid.
    """
    if not hash_str:
        raise ValueError("Hash string cannot be empty")

    # Check if it's a valid hexadecimal string
    try:
        int(hash_str, 16)
    except ValueError:
        raise ValueError(f"Invalid hash format: {hash_str}")

    # Check length if specified
    if expected_length and len(hash_str) != expected_length:
        raise ValueError(
            f"Hash length mismatch: expected {expected_length}, got {len(hash_str)}"
        )

    return True


def compute_parse_hash(
    parse_method: str, params: Optional[Dict[str, Any]] = None
) -> str:
    """Compute parse_hash based on parse_method and parameters.

    Args:
        parse_method: Parsing method (e.g., 'pypdf', 'pdfplumber').
        params: Parameters dictionary (optional).

    Returns:
        SHA256 hash string of canonical parse configuration.

    Note:
        Parse hash is computed from parse_method and whitelisted parameters
        to detect changes in parsing configuration.
    """
    # Create canonical parameter dictionary
    canonical_params = {}
    if params:
        # Filter to whitelisted parameters based on parse_method
        whitelist = get_parse_params_whitelist(parse_method)
        for key, value in params.items():
            if key in whitelist:
                canonical_params[key] = value

    # Create canonical JSON string
    canonical_data = {"parse_method": parse_method, "params": canonical_params}

    # Sort keys for consistent ordering
    canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))

    # Compute SHA256 hash
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_parse_params_whitelist(parse_method: str) -> list[str]:
    """Get whitelist of parameters for a specific parse method.

    Args:
        parse_method: Parsing method name.

    Returns:
        List of whitelisted parameter names.
    """
    whitelists = {
        # PDF parsers
        "pypdf": [],
        "pdfplumber": ["extract_tables", "extract_images"],
        "unstructured": ["strategy", "include_metadata"],
        "pymupdf": ["extract_images", "extract_tables"],
        # Office parsers
        "docx_default": [],
        "xlsx_default": ["sheet_names", "include_headers"],
        "pptx_default": [],
        # Text parsers
        "txt_default": [],
        "md_default": [],
        "json_default": ["extract_arrays", "flatten_nested"],
        "html_default": ["extract_tables", "remove_tags"],
    }

    return whitelists.get(parse_method, [])


def compute_chunk_hash(text: str, chunk_params: Optional[Dict[str, Any]] = None) -> str:
    """Compute chunk_hash based on text content and chunking parameters.

    Args:
        text: Text content of the chunk.
        chunk_params: Chunking parameters dictionary (optional).

    Returns:
        SHA256 hash string of text and chunk configuration.

    Note:
        Chunk hash is computed from text content and core chunking parameters
        to detect changes in chunking configuration or content.
    """
    # Create canonical parameter dictionary
    canonical_params = {}
    if chunk_params:
        # Filter to core chunking parameters
        whitelist = get_chunk_params_whitelist()
        for key, value in chunk_params.items():
            if key in whitelist:
                canonical_params[key] = value

    # Create canonical JSON string
    canonical_data = {"text": text, "chunk_params": canonical_params}

    # Sort keys for consistent ordering
    canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))

    # Compute SHA256 hash
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_chunk_params_whitelist() -> list[str]:
    """Get whitelist of core chunking parameters.

    Returns:
        List of whitelisted chunking parameter names.
    """
    return [
        "chunk_strategy",
        "chunk_size",
        "chunk_overlap",
        "headers_to_split_on",
        "separators",
        "use_token_count",
        "tiktoken_encoding",
        "enable_protected_content",
        "protected_patterns",
        "table_context_size",
        "image_context_size",
    ]


def compute_embed_hash(
    text: str, model: str, embed_params: Optional[Dict[str, Any]] = None
) -> str:
    """Compute embed_hash based on text, model and embedding parameters.

    Args:
        text: Text content to be embedded.
        model: Embedding model name.
        embed_params: Embedding parameters dictionary (optional).

    Returns:
        SHA256 hash string of text, model and embedding configuration.

    Note:
        Embed hash is computed from text content, model name and embedding parameters
        to detect changes in embedding configuration or content.
    """
    # Create canonical parameter dictionary
    canonical_params = {}
    if embed_params:
        # Filter to core embedding parameters
        whitelist = get_embed_params_whitelist()
        for key, value in embed_params.items():
            if key in whitelist:
                canonical_params[key] = value

    # Create canonical JSON string
    canonical_data = {"text": text, "model": model, "embed_params": canonical_params}

    # Sort keys for consistent ordering
    canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))

    # Compute SHA256 hash
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_embed_params_whitelist() -> list[str]:
    """Get whitelist of core embedding parameters.

    Returns:
        List of whitelisted embedding parameter names.
    """
    return ["normalize_embeddings", "batch_size", "device", "max_length"]
