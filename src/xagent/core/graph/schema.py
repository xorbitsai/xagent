from typing import Any

import jsonschema  # type: ignore[import-untyped]
from jsonschema import Draft7Validator


class SchemaParser:
    """Parser for validating data against JSON schemas."""

    def __init__(self, schemas: dict[str, dict[str, Any]]):
        """Initialize the schema parser.

        Args:
            schemas: Dictionary mapping schema IDs to JSON schema definitions.

        Raises:
            jsonschema.SchemaError: If any schema is invalid.
        """
        self.schemas = self._normalize_schemas(schemas)
        self._validate_schemas()

    def _normalize_schemas(
        self, schemas: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Add title and description to schemas if missing."""
        normalized = {}
        for schema_id, schema in schemas.items():
            schema_copy = schema.copy()
            if "title" not in schema_copy:
                schema_copy["title"] = schema_id
            if "description" not in schema_copy:
                schema_copy["description"] = schema_id
            normalized[schema_id] = schema_copy
        return normalized

    def _validate_schemas(self) -> None:
        """Validate that all schemas conform to JSON Schema format."""
        for schema_id, schema in self.schemas.items():
            try:
                Draft7Validator.check_schema(schema)
            except jsonschema.SchemaError as e:
                raise jsonschema.SchemaError(
                    f"Invalid JSON schema for '{schema_id}': {e.message}"
                )
