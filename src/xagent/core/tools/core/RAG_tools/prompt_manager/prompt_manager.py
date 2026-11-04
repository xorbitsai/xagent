"""Core prompt manager for CRUD operations and version management.

This module provides functions for managing prompt templates
with full CRUD operations and transparent version management using LanceDB.

Phase 1A Part 2: Refactored to use PromptTemplateStore abstraction layer
for basic operations while preserving complex business logic.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentNotFoundError,
)
from ..core.schemas import PromptTemplate
from ..storage.factory import get_prompt_template_store

logger = logging.getLogger(__name__)


def _serialize_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Serialize metadata dictionary to JSON string.

    Args:
        metadata: Metadata dictionary to serialize.

    Returns:
        JSON string or None.
    """
    if metadata is None:
        return None
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


# ------------------------- Public Functions -------------------------


def create_prompt_template(
    collection: str,
    name: str,
    template: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> PromptTemplate:
    """Create a new prompt template or new version.

    Args:
        collection: Collection name for data isolation.
        name: Human-readable name for the prompt template.
        template: The actual prompt template content.
        metadata: Optional metadata dictionary.

    Returns:
        Created PromptTemplate instance.

    Raises:
        ConfigurationError: If collection, name or template is empty.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")
    if not name or not name.strip():
        raise ConfigurationError("Prompt template name cannot be empty.")
    if not template or not template.strip():
        raise ConfigurationError("Prompt template content cannot be empty.")

    # Normalize name
    name = name.strip()

    try:
        store = get_prompt_template_store()

        # Save template via store (handles version management automatically)
        template_id = store.save_prompt_template(
            name=name,
            template=template.strip(),
            user_id=None,  # No multi-tenancy in current implementation
            metadata=_serialize_metadata(metadata),
        )

        # Get the created template to return full PromptTemplate object
        template_data = store.get_prompt_template(template_id, user_id=None)
        if template_data is None:
            raise DatabaseOperationError("Failed to retrieve created template")

        prompt_template = PromptTemplate(
            id=template_data["id"],
            name=template_data["name"],
            template=template_data["template"],
            version=template_data["version"],
            is_latest=template_data["is_latest"],
            metadata=template_data["metadata"],
            user_id=template_data["user_id"],
            created_at=template_data["created_at"],
            updated_at=template_data["updated_at"],
        )

        logger.info(
            "Created prompt template '%s' version %s", name, prompt_template.version
        )
        return prompt_template

    except (ConfigurationError, DatabaseOperationError):
        raise
    except Exception as e:
        logger.error("Failed to create prompt template '%s': %s", name, str(e))
        raise DatabaseOperationError(
            f"Failed to create prompt template: {str(e)}"
        ) from e


def read_prompt_template(
    collection: str,
    prompt_id: Optional[str] = None,
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> PromptTemplate:
    """Read a specific prompt template.

    Args:
        collection: Collection name for data isolation.
        prompt_id: UUID of the prompt template.
        name: Name of the prompt template.
        version: Specific version number (if not provided, latest is returned).

    Returns:
        PromptTemplate instance.

    Raises:
        ConfigurationError: If collection is empty or neither prompt_id nor name is provided.
        DocumentNotFoundError: If prompt template is not found.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")
    if not prompt_id and not name:
        raise ConfigurationError("Either prompt_id or name must be provided.")

    try:
        store = get_prompt_template_store()

        if prompt_id:
            # Search by ID
            template_data = store.get_prompt_template(prompt_id, user_id=None)
            if template_data is None:
                raise DocumentNotFoundError(
                    f"Prompt template with ID '{prompt_id}' not found."
                )
        else:
            # Search by name
            assert (
                name is not None
            )  # Type narrowing: name must be provided if prompt_id is None
            name = name.strip() if name else name
            if version is not None:
                # Get specific version - need to search through list
                templates = store.list_prompt_templates(
                    name_filter=name,
                    latest_only=False,
                    user_id=None,
                    limit=100,
                )
                matching = [
                    t
                    for t in templates
                    if t["name"] == name and t["version"] == version
                ]
                if not matching:
                    raise DocumentNotFoundError(
                        f"Prompt template with name '{name}' version {version} not found."
                    )
                template_data = matching[0]
            else:
                # Get latest version
                template_data = store.get_latest_prompt_template(name, user_id=None)
                if template_data is None:
                    raise DocumentNotFoundError(
                        f"Prompt template with name '{name}' not found."
                    )

        # Convert to PromptTemplate
        return PromptTemplate(
            id=template_data["id"],
            name=template_data["name"],
            template=template_data["template"],
            version=template_data["version"],
            is_latest=template_data["is_latest"],
            metadata=template_data["metadata"],
            user_id=template_data["user_id"],
            created_at=template_data["created_at"],
            updated_at=template_data["updated_at"],
        )

    except (ConfigurationError, DocumentNotFoundError):
        raise
    except Exception as e:
        logger.error("Failed to read prompt template: %s", str(e))
        raise DatabaseOperationError(f"Failed to read prompt template: {str(e)}") from e


def get_latest_prompt_template(collection: str, name: str) -> PromptTemplate:
    """Get the latest version of a prompt template by name.

    Args:
        collection: Collection name for data isolation.
        name: Name of the prompt template.

    Returns:
        Latest PromptTemplate instance.

    Raises:
        ConfigurationError: If collection or name is empty.
        DocumentNotFoundError: If prompt template is not found.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")
    if not name or not name.strip():
        raise ConfigurationError("Prompt template name cannot be empty.")

    return read_prompt_template(collection=collection, name=name.strip())


def update_prompt_template(
    collection: str,
    prompt_id: str,
    template: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PromptTemplate:
    """Update a prompt template.

    If template content is changed, creates a new version.
    If only metadata is changed, updates the current version.

    Args:
        collection: Collection name for data isolation.
        prompt_id: UUID of the prompt template to update.
        template: New template content (creates new version if provided).
        metadata: New metadata (updates current version if provided).

    Returns:
        Updated PromptTemplate instance.

    Raises:
        ConfigurationError: If collection or prompt_id is empty.
        DocumentNotFoundError: If prompt template is not found.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")
    if not prompt_id:
        raise ConfigurationError("Prompt ID must be provided.")

    if template is None and metadata is None:
        raise ConfigurationError(
            "Either template or metadata must be provided for update."
        )

    try:
        # First, get the current template
        current_template = read_prompt_template(
            collection=collection, prompt_id=prompt_id
        )

        if template is not None:
            # Template content changed - create new version
            if not template.strip():
                raise ConfigurationError("Template content cannot be empty.")

            # Create new version using store (handles version management automatically)
            new_metadata = (
                _serialize_metadata(metadata)
                if metadata is not None
                else current_template.metadata
            )
            new_template_id = get_prompt_template_store().save_prompt_template(
                name=current_template.name,
                template=template.strip(),
                user_id=None,
                metadata=new_metadata,
            )

            # Get the created template
            new_template_data = get_prompt_template_store().get_prompt_template(
                new_template_id, user_id=None
            )
            if new_template_data is None:
                raise DatabaseOperationError("Failed to retrieve updated template")

            updated_template = PromptTemplate(
                id=new_template_data["id"],
                name=new_template_data["name"],
                template=new_template_data["template"],
                version=new_template_data["version"],
                is_latest=new_template_data["is_latest"],
                metadata=new_template_data["metadata"],
                user_id=new_template_data["user_id"],
                created_at=new_template_data["created_at"],
                updated_at=new_template_data["updated_at"],
            )

            logger.info(
                "Created new version %s for prompt template '%s'",
                updated_template.version,
                current_template.name,
            )
            return updated_template

        else:
            # Only metadata changed - update in-place using store method
            new_metadata = _serialize_metadata(metadata)
            updated_data = get_prompt_template_store().update_metadata(
                template_id=prompt_id,
                metadata=new_metadata,
                user_id=None,
            )
            if updated_data is None:
                raise DatabaseOperationError("Failed to retrieve updated template")

            updated_template = PromptTemplate(
                id=updated_data["id"],
                name=updated_data["name"],
                template=updated_data["template"],
                version=updated_data["version"],
                is_latest=updated_data["is_latest"],
                metadata=updated_data["metadata"],
                user_id=updated_data["user_id"],
                created_at=updated_data["created_at"],
                updated_at=updated_data["updated_at"],
            )

            logger.info(
                "Updated metadata for prompt template '%s' (version %s)",
                current_template.name,
                updated_template.version,
            )
            return updated_template

    except (ConfigurationError, DocumentNotFoundError):
        raise
    except Exception as e:
        logger.error("Failed to update prompt template %s: %s", prompt_id, str(e))
        raise DatabaseOperationError(
            f"Failed to update prompt template: {str(e)}"
        ) from e


def delete_prompt_template(
    collection: str,
    prompt_id: Optional[str] = None,
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> bool:
    """Delete a prompt template or specific version.

    Args:
        collection: Collection name for data isolation.
        prompt_id: UUID of the prompt template to delete.
        name: Name of the prompt template to delete.
        version: Specific version to delete (if not provided, all versions are deleted).

    Returns:
        True if deletion was successful.

    Raises:
        ConfigurationError: If collection is empty or neither prompt_id nor name is provided.
        DocumentNotFoundError: If prompt template is not found.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")
    if not prompt_id and not name:
        raise ConfigurationError("Either prompt_id or name must be provided.")

    try:
        store = get_prompt_template_store()

        if prompt_id:
            # Delete by ID
            result = store.delete_prompt_template(template_id=prompt_id, user_id=None)
            if not result:
                raise DocumentNotFoundError(
                    f"Prompt template with ID '{prompt_id}' not found."
                )
            logger.info("Deleted prompt template with ID '%s'", prompt_id)
            return True
        else:
            # Normalize name
            assert (
                name is not None
            )  # Type narrowing: name must be provided if prompt_id is None
            name = name.strip() if name else name
            # Delete by name using store method (handles version management automatically)
            store.delete_by_name(name=name, version=version, user_id=None)
            if version is not None:
                logger.info("Deleted prompt template '%s' version %s", name, version)
            else:
                logger.info("Deleted all versions of prompt template '%s'", name)
            return True

    except (ConfigurationError, DocumentNotFoundError):
        raise
    except Exception as e:
        logger.error("Failed to delete prompt template: %s", str(e))
        raise DatabaseOperationError(
            f"Failed to delete prompt template: {str(e)}"
        ) from e


def list_prompt_templates(
    collection: str,
    name_filter: Optional[str] = None,
    latest_only: bool = False,
    metadata_filter: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> List[PromptTemplate]:
    """List prompt templates with optional filtering.

    Args:
        collection: Collection name for data isolation.
        name_filter: Filter by name (partial match).
        latest_only: If True, only return latest versions.
        metadata_filter: Filter by metadata fields (not yet implemented).
        limit: Maximum number of results to return (default: 100).

    Returns:
        List of PromptTemplate instances.

    Raises:
        ConfigurationError: If collection is empty.
        DatabaseOperationError: If database operation fails.
    """
    if not collection:
        raise ConfigurationError("Collection name cannot be empty.")

    try:
        store = get_prompt_template_store()

        # Note: metadata filtering would require more complex logic
        if metadata_filter:
            logger.warning("Metadata filtering is not yet implemented")

        # Use store method to list templates
        templates_data = store.list_prompt_templates(
            name_filter=name_filter,
            latest_only=latest_only,
            user_id=None,
            limit=limit,
        )

        # Convert to PromptTemplate objects
        templates = []
        for template_data in templates_data:
            templates.append(
                PromptTemplate(
                    id=template_data["id"],
                    name=template_data["name"],
                    template=template_data["template"],
                    version=template_data["version"],
                    is_latest=template_data["is_latest"],
                    metadata=template_data["metadata"],
                    user_id=template_data["user_id"],
                    created_at=template_data["created_at"],
                    updated_at=template_data["updated_at"],
                )
            )

        logger.info("Listed %s prompt templates (limit: %s)", len(templates), limit)
        return templates

    except (ConfigurationError, DatabaseOperationError):
        raise
    except Exception as e:
        logger.error("Failed to list prompt templates: %s", str(e))
        raise DatabaseOperationError(
            f"Failed to list prompt templates: {str(e)}"
        ) from e
