"""
Web services module.
"""

from .model_service import (
    get_default_image_edit_model,
    get_default_image_generate_model,
    get_default_model,
    get_default_vision_model,
)

__all__ = [
    "get_default_model",
    "get_default_vision_model",
    "get_default_image_generate_model",
    "get_default_image_edit_model",
]
