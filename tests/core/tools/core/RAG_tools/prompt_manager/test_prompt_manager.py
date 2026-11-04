"""Tests for prompt manager CRUD operations and version management."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from uuid import uuid4

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    ConfigurationError,
    DocumentNotFoundError,
)
from xagent.core.tools.core.RAG_tools.prompt_manager import (
    create_prompt_template,
    delete_prompt_template,
    get_latest_prompt_template,
    list_prompt_templates,
    read_prompt_template,
    update_prompt_template,
)


class TestCreatePromptTemplate:
    """Test cases for creating prompt templates."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = self.temp_dir
        self.collection = "test"

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        if self.original_env is not None:
            os.environ["LANCEDB_DIR"] = self.original_env
        elif "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_first_prompt_template(self) -> None:
        """Test creating the first prompt template."""
        metadata_dict = {"category": "greetings", "author": "test"}
        result = create_prompt_template(
            collection=self.collection,
            name="greeting",
            template="Hello {name}!",
            metadata=metadata_dict,
        )

        assert result.name == "greeting"
        assert result.template == "Hello {name}!"
        assert result.version == 1
        assert result.is_latest is True
        # metadata is stored as JSON string
        assert result.metadata == json.dumps(
            metadata_dict, ensure_ascii=False, sort_keys=True
        )
        assert isinstance(result.id, str)
        assert isinstance(result.created_at, datetime)

    def test_create_new_version_increments_version(self) -> None:
        """Test that creating a new template with same name creates a new version."""
        # Create first version
        v1 = create_prompt_template(
            collection=self.collection, name="greeting", template="Hello {name}!"
        )

        # Create second version
        v2 = create_prompt_template(
            collection=self.collection, name="greeting", template="Hi {name}, welcome!"
        )

        assert v1.version == 1
        assert v2.version == 2
        assert v2.is_latest is True

        # Verify first version is no longer latest
        v1_updated = read_prompt_template(collection=self.collection, prompt_id=v1.id)
        assert v1_updated.is_latest is False

    def test_create_with_empty_collection_raises_error(self) -> None:
        """Test that creating with empty collection raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Collection name cannot be empty"):
            create_prompt_template(collection="", name="test", template="test")

    def test_create_with_empty_name_raises_error(self) -> None:
        """Test that creating with empty name raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="name cannot be empty"):
            create_prompt_template(collection=self.collection, name="", template="test")

    def test_create_with_whitespace_name_raises_error(self) -> None:
        """Test that creating with whitespace-only name raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="name cannot be empty"):
            create_prompt_template(
                collection=self.collection, name="   ", template="test"
            )

    def test_create_with_empty_template_raises_error(self) -> None:
        """Test that creating with empty template raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="content cannot be empty"):
            create_prompt_template(collection=self.collection, name="test", template="")

    def test_create_with_whitespace_template_raises_error(self) -> None:
        """Test that creating with whitespace-only template raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="content cannot be empty"):
            create_prompt_template(
                collection=self.collection, name="test", template="   "
            )

    def test_create_normalizes_name(self) -> None:
        """Test that names are normalized (trimmed) during creation."""
        result = create_prompt_template(
            collection=self.collection, name="  greeting  ", template="Hello!"
        )
        assert result.name == "greeting"

    def test_create_normalizes_template(self) -> None:
        """Test that templates are normalized (trimmed) during creation."""
        result = create_prompt_template(
            collection=self.collection, name="greeting", template="  Hello {name}!  "
        )
        assert result.template == "Hello {name}!"


class TestReadPromptTemplate:
    """Test cases for reading prompt templates."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = self.temp_dir
        self.collection = "test"

        # Create test data
        self.template_v1 = create_prompt_template(
            collection=self.collection, name="greeting", template="Hello {name}!"
        )
        self.template_v2 = create_prompt_template(
            collection=self.collection, name="greeting", template="Hi {name}, welcome!"
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        if self.original_env is not None:
            os.environ["LANCEDB_DIR"] = self.original_env
        elif "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_by_id(self) -> None:
        """Test reading prompt template by ID."""
        result = read_prompt_template(
            collection=self.collection, prompt_id=self.template_v1.id
        )
        assert result.id == self.template_v1.id
        assert result.name == "greeting"
        assert result.version == 1
        assert result.is_latest is False

    def test_read_by_name_latest(self) -> None:
        """Test reading latest version by name."""
        result = read_prompt_template(collection=self.collection, name="greeting")
        assert result.name == "greeting"
        assert result.version == 2
        assert result.is_latest is True

    def test_read_by_name_specific_version(self) -> None:
        """Test reading specific version by name."""
        result = read_prompt_template(
            collection=self.collection, name="greeting", version=1
        )
        assert result.name == "greeting"
        assert result.version == 1
        assert result.template == "Hello {name}!"

    def test_read_with_empty_collection_raises_error(self) -> None:
        """Test that reading with empty collection raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Collection name cannot be empty"):
            read_prompt_template(collection="", name="test")

    def test_read_without_id_or_name_raises_error(self) -> None:
        """Test that reading without ID or name raises ConfigurationError."""
        with pytest.raises(
            ConfigurationError, match="Either prompt_id or name must be provided"
        ):
            read_prompt_template(collection=self.collection)

    def test_read_nonexistent_id_raises_error(self) -> None:
        """Test that reading nonexistent ID raises DocumentNotFoundError."""
        fake_id = str(uuid4())
        with pytest.raises(DocumentNotFoundError, match=f"ID '{fake_id}'"):
            read_prompt_template(collection=self.collection, prompt_id=fake_id)

    def test_read_nonexistent_name_raises_error(self) -> None:
        """Test that reading nonexistent name raises DocumentNotFoundError."""
        with pytest.raises(DocumentNotFoundError, match="name 'nonexistent'"):
            read_prompt_template(collection=self.collection, name="nonexistent")

    def test_read_nonexistent_version_raises_error(self) -> None:
        """Test that reading nonexistent version raises DocumentNotFoundError."""
        with pytest.raises(DocumentNotFoundError, match="version 999"):
            read_prompt_template(
                collection=self.collection, name="greeting", version=999
            )

    def test_get_latest_by_name(self) -> None:
        """Test getting latest prompt template by name."""
        result = get_latest_prompt_template(collection=self.collection, name="greeting")
        assert result.name == "greeting"
        assert result.version == 2
        assert result.is_latest is True

    def test_get_latest_with_empty_collection_raises_error(self) -> None:
        """Test that getting latest with empty collection raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Collection name cannot be empty"):
            get_latest_prompt_template(collection="", name="test")

    def test_get_latest_with_empty_name_raises_error(self) -> None:
        """Test that getting latest with empty name raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="name cannot be empty"):
            get_latest_prompt_template(collection=self.collection, name="")


class TestUpdatePromptTemplate:
    """Test cases for updating prompt templates."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = self.temp_dir
        self.collection = "test"

        # Create test data
        self.template = create_prompt_template(
            collection=self.collection, name="greeting", template="Hello {name}!"
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        if self.original_env is not None:
            os.environ["LANCEDB_DIR"] = self.original_env
        elif "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update_template_creates_new_version(self) -> None:
        """Test that updating template content creates a new version."""
        result = update_prompt_template(
            collection=self.collection,
            prompt_id=self.template.id,
            template="Hi {name}, welcome!",
        )

        assert result.name == "greeting"
        assert result.version == 2
        assert result.template == "Hi {name}, welcome!"
        assert result.is_latest is True

        # Verify old version is no longer latest
        old_version = read_prompt_template(
            collection=self.collection, prompt_id=self.template.id
        )
        assert old_version.is_latest is False

    def test_update_metadata_only_updates_current_version(self) -> None:
        """Test that updating only metadata updates the current version."""
        metadata_dict = {"category": "friendly", "author": "test"}
        result = update_prompt_template(
            collection=self.collection,
            prompt_id=self.template.id,
            metadata=metadata_dict,
        )

        assert result.id == self.template.id
        assert result.version == 1
        assert result.template == "Hello {name}!"
        # metadata is stored as JSON string
        assert result.metadata == json.dumps(
            metadata_dict, ensure_ascii=False, sort_keys=True
        )

    def test_update_both_template_and_metadata(self) -> None:
        """Test updating both template and metadata creates new version with new metadata."""
        metadata_dict = {"new": "metadata"}
        result = update_prompt_template(
            collection=self.collection,
            prompt_id=self.template.id,
            template="New template!",
            metadata=metadata_dict,
        )

        assert result.version == 2
        assert result.template == "New template!"
        # metadata is stored as JSON string
        assert result.metadata == json.dumps(
            metadata_dict, ensure_ascii=False, sort_keys=True
        )

    def test_update_with_empty_collection_raises_error(self) -> None:
        """Test that updating with empty collection raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Collection name cannot be empty"):
            update_prompt_template(
                collection="", prompt_id=self.template.id, template="test"
            )

    def test_update_without_prompt_id_raises_error(self) -> None:
        """Test that updating without prompt_id raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Prompt ID must be provided"):
            update_prompt_template(
                collection=self.collection, prompt_id="", template="test"
            )

    def test_update_without_template_or_metadata_raises_error(self) -> None:
        """Test that updating without template or metadata raises ConfigurationError."""
        with pytest.raises(
            ConfigurationError, match="Either template or metadata must be provided"
        ):
            update_prompt_template(
                collection=self.collection, prompt_id=self.template.id
            )

    def test_update_nonexistent_id_raises_error(self) -> None:
        """Test that updating nonexistent ID raises DocumentNotFoundError."""
        fake_id = str(uuid4())
        with pytest.raises(DocumentNotFoundError):
            update_prompt_template(
                collection=self.collection, prompt_id=fake_id, template="test"
            )

    def test_update_with_empty_template_raises_error(self) -> None:
        """Test that updating with empty template raises ConfigurationError."""
        with pytest.raises(
            ConfigurationError, match="Template content cannot be empty"
        ):
            update_prompt_template(
                collection=self.collection, prompt_id=self.template.id, template=""
            )


class TestDeletePromptTemplate:
    """Test cases for deleting prompt templates."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = self.temp_dir
        self.collection = "test"

        # Create test data with multiple versions
        self.template_v1 = create_prompt_template(
            collection=self.collection, name="greeting", template="Hello {name}!"
        )
        self.template_v2 = create_prompt_template(
            collection=self.collection, name="greeting", template="Hi {name}!"
        )
        self.template_v3 = create_prompt_template(
            collection=self.collection, name="greeting", template="Hey {name}!"
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        if self.original_env is not None:
            os.environ["LANCEDB_DIR"] = self.original_env
        elif "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_delete_by_id(self) -> None:
        """Test deleting prompt template by ID."""
        result = delete_prompt_template(
            collection=self.collection, prompt_id=self.template_v1.id
        )
        assert result is True

        # Verify deletion
        with pytest.raises(DocumentNotFoundError):
            read_prompt_template(
                collection=self.collection, prompt_id=self.template_v1.id
            )

    def test_delete_latest_version_updates_is_latest(self) -> None:
        """Test that deleting latest version updates is_latest for remaining versions."""
        # Delete v3 (latest)
        delete_prompt_template(
            collection=self.collection, prompt_id=self.template_v3.id
        )

        # Verify v2 is now latest
        v2 = read_prompt_template(
            collection=self.collection, prompt_id=self.template_v2.id
        )
        assert v2.is_latest is True

    def test_delete_specific_version_by_name(self) -> None:
        """Test deleting specific version by name."""
        result = delete_prompt_template(
            collection=self.collection, name="greeting", version=1
        )
        assert result is True

        # Verify deletion
        with pytest.raises(DocumentNotFoundError):
            read_prompt_template(collection=self.collection, name="greeting", version=1)

    def test_delete_all_versions_by_name(self) -> None:
        """Test deleting all versions by name."""
        result = delete_prompt_template(collection=self.collection, name="greeting")
        assert result is True

        # Verify all deleted
        with pytest.raises(DocumentNotFoundError):
            read_prompt_template(collection=self.collection, name="greeting")

    def test_delete_with_empty_collection_raises_error(self) -> None:
        """Test that deleting with empty collection raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Collection name cannot be empty"):
            delete_prompt_template(collection="", prompt_id=self.template_v1.id)

    def test_delete_without_id_or_name_raises_error(self) -> None:
        """Test that deleting without ID or name raises ConfigurationError."""
        with pytest.raises(
            ConfigurationError, match="Either prompt_id or name must be provided"
        ):
            delete_prompt_template(collection=self.collection)

    def test_delete_nonexistent_id_raises_error(self) -> None:
        """Test that deleting nonexistent ID raises DocumentNotFoundError."""
        fake_id = str(uuid4())
        with pytest.raises(DocumentNotFoundError):
            delete_prompt_template(collection=self.collection, prompt_id=fake_id)

    def test_delete_nonexistent_name_raises_error(self) -> None:
        """Test that deleting nonexistent name raises DocumentNotFoundError."""
        with pytest.raises(DocumentNotFoundError):
            delete_prompt_template(collection=self.collection, name="nonexistent")


class TestListPromptTemplates:
    """Test cases for listing prompt templates."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = self.temp_dir
        self.collection = "test"

        # Create test data
        create_prompt_template(
            collection=self.collection, name="greeting", template="Hello!"
        )
        create_prompt_template(
            collection=self.collection, name="greeting", template="Hi!"
        )
        create_prompt_template(
            collection=self.collection, name="farewell", template="Goodbye!"
        )
        create_prompt_template(
            collection=self.collection, name="welcome", template="Welcome!"
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        if self.original_env is not None:
            os.environ["LANCEDB_DIR"] = self.original_env
        elif "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_all_templates(self) -> None:
        """Test listing all prompt templates."""
        result = list_prompt_templates(collection=self.collection)
        assert len(result) == 4  # 2 versions of greeting + farewell + welcome

    def test_list_latest_only(self) -> None:
        """Test listing only latest versions."""
        result = list_prompt_templates(collection=self.collection, latest_only=True)
        assert len(result) == 3  # Only latest versions
        assert all(t.is_latest for t in result)

    def test_list_with_name_filter(self) -> None:
        """Test listing with name filter."""
        result = list_prompt_templates(collection=self.collection, name_filter="greet")
        assert len(result) == 2  # Both versions of greeting
        assert all("greet" in t.name for t in result)

    def test_list_with_name_filter_and_latest_only(self) -> None:
        """Test listing with both name filter and latest_only."""
        result = list_prompt_templates(
            collection=self.collection, name_filter="greet", latest_only=True
        )
        assert len(result) == 1
        assert result[0].name == "greeting"
        assert result[0].version == 2
        assert result[0].is_latest is True

    def test_list_with_limit(self) -> None:
        """Test listing with limit."""
        result = list_prompt_templates(collection=self.collection, limit=2)
        assert len(result) == 2

    def test_list_with_empty_collection_raises_error(self) -> None:
        """Test that listing with empty collection raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Collection name cannot be empty"):
            list_prompt_templates(collection="")

    def test_list_empty_result(self) -> None:
        """Test listing with filter that matches nothing."""
        result = list_prompt_templates(
            collection=self.collection, name_filter="nonexistent"
        )
        assert len(result) == 0


class TestVersionConflictScenarios:
    """Test cases for version conflict scenarios."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = self.temp_dir
        self.collection = "test"

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        if self.original_env is not None:
            os.environ["LANCEDB_DIR"] = self.original_env
        elif "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update_by_id_with_higher_version_exists(self) -> None:
        """Test that updating by ID correctly handles version when higher version exists."""
        # Create v1 and v2
        v1 = create_prompt_template(
            collection=self.collection, name="test", template="v1"
        )
        v2 = create_prompt_template(
            collection=self.collection, name="test", template="v2"
        )

        # Update v1 - should create v3, not v2
        v3 = update_prompt_template(
            collection=self.collection, prompt_id=v1.id, template="v3"
        )

        assert v3.version == 3
        assert v3.is_latest is True

        # Verify v2 is no longer latest
        v2_updated = read_prompt_template(collection=self.collection, prompt_id=v2.id)
        assert v2_updated.is_latest is False
