from copy import deepcopy
from typing import Any, Callable

from .error import InvalidSchemaIDError, MismatchInputOutputError
from .node import InputSchema, Node, OutputSchema


def get_validate_input_output_fn(
    all_schemas: dict[str, dict[str, Any]] | None = None,
) -> Callable[[Node, Any], Any]:
    """
    Validate that node input schemas are satisfied by upstream output schemas
    using recursive structural comparison.
    """
    schemas = all_schemas or {}

    def get_schema_props(schema_id: str) -> dict[str, Any]:
        """Retrieve the 'properties' dictionary from a schema definition."""
        if schema_id not in schemas:
            raise InvalidSchemaIDError(schema_id, list(schemas.keys()))
        return dict(schemas[schema_id].get("properties", {}))

    def validate_recursive(
        required_props: dict[str, Any], available_props: dict[str, Any], path: str = ""
    ) -> None:
        """
        Recursively check if 'required_props' are satisfied by 'available_props'.
        """

        def is_all_const_properties(definition: dict[str, Any]) -> bool:
            """Check if an object definition has only const properties."""
            if definition.get("type") != "object" or "properties" not in definition:
                return False
            try:
                validate_recursive(definition["properties"], {}, path)
                return True
            except MismatchInputOutputError:
                return False

        for field, req_def in required_props.items():
            current_path = f"{path}.{field}" if path else field

            # 1. Check for 'const' (Self-satisfied)
            if "const" in req_def:
                continue

            # 2. Check Existence
            if field not in available_props:
                if is_all_const_properties(req_def):
                    continue
                raise MismatchInputOutputError(
                    "SCHEMA_VALIDATION",
                    f"Missing required field, current path: '{current_path}', available: {available_props}.",
                )

            prov_def = available_props[field]
            req_type = req_def.get("type")
            prov_type = prov_def.get("type")

            # 3. Check Type
            if req_type and prov_type and req_type != prov_type:
                raise MismatchInputOutputError(
                    "",
                    f"Type mismatch at path: '{current_path}': expected '{req_type}', got '{prov_type}'.",
                )

            # 4. Recurse for Objects
            if req_type == "object" and "properties" in req_def:
                req_nested = req_def.get("properties", {})
                prov_nested = prov_def.get("properties", {})
                validate_recursive(req_nested, prov_nested, current_path)

    def merge_properties(target: dict[str, Any], source: dict[str, Any]) -> None:
        """
        Deep merge 'source' properties into 'target' properties.
        """
        for key, val in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(val, dict)
                and target[key].get("type") == "object"
                and val.get("type") == "object"
            ):
                target_props = target[key].setdefault("properties", {})
                source_props = val.get("properties", {})
                merge_properties(target_props, source_props)
            else:
                target[key] = val

    def validate_node(node: Node, available_props: dict[str, Any]) -> dict[str, Any]:
        if isinstance(node, InputSchema) and node.input_schema is not None:
            required_props = get_schema_props(node.input_schema)
            if node.input_overrides is not None:
                override_keys = set(node.input_overrides.keys())
                required_props = {
                    k: v for k, v in required_props.items() if k not in override_keys
                }
            try:
                validate_recursive(required_props, available_props)
            except MismatchInputOutputError as e:
                # Re-raise with the correct Node ID
                raise MismatchInputOutputError(node.id, str(e))

        # Deepcopy to ensure branches in the graph don't pollute each other
        next_props = deepcopy(available_props)

        if isinstance(node, OutputSchema) and node.output_schema is not None:
            output_props = get_schema_props(node.output_schema)
            merge_properties(next_props, output_props)

        return next_props

    return validate_node
