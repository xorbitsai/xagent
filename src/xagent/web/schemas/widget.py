"""Shared request contracts for widget configuration."""

from typing import Annotated

from pydantic import AfterValidator

from ..services.widget_domains import normalize_widget_allowed_domain


def _require_nonblank_widget_allowed_domain(value: str) -> str:
    if not normalize_widget_allowed_domain(value):
        raise ValueError("Widget allowed domain must not be blank")
    return value


WidgetAllowedDomain = Annotated[
    str, AfterValidator(_require_nonblank_widget_allowed_domain)
]

__all__ = ["WidgetAllowedDomain"]
