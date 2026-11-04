"""Composition root. Add checks here without modifying tools or delivery code."""

from .formats import check_csv, check_image, check_pdf
from .models import ArtifactCheck
from .office import check_office_document, check_package
from .registry import ArtifactCheckRegistry


def default_registry() -> ArtifactCheckRegistry:
    registry = ArtifactCheckRegistry()
    office = frozenset({".xlsx", ".docx", ".pptx"})
    for check in (
        ArtifactCheck("office-package", office, check_package),
        ArtifactCheck("office-reader", office, check_office_document),
        ArtifactCheck("csv-reader", frozenset({".csv", ".tsv"}), check_csv),
        ArtifactCheck("pdf-reader", frozenset({".pdf"}), check_pdf),
        ArtifactCheck(
            "image-decoder",
            frozenset(
                {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
            ),
            check_image,
        ),
    ):
        registry.register(check)
    return registry
