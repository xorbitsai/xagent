"""
Tool Configuration Management

Provides abstract and concrete configuration classes for tool creation.
This allows different contexts (web, standalone) to provide configuration
to the ToolFactory in a unified way.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

from ..... import config as _root_config


def normalize_tool_allowlist(value: Any) -> Optional[List[str]]:
    """Coerce a tool-allowlist hook result into a list of tool-name strings.

    The positive tool-allowlist hook is documented as ``Optional[list]``, but
    it is user-registered and consumed against a duck-typed config, so callers
    cannot assume the contract holds. Normalizing here keeps a single, uniform
    policy at every consumption point:

    - ``None`` passes through unchanged ("no allowlist configured").
    - A bare scalar — ``str``/``bytes`` or any non-iterable such as
      ``int``/``float``/``bool`` — is treated as a single tool name rather than
      being iterated character-by-character or raising on a non-iterable.
    - Any other iterable yields one stringified name per item.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return [str(value)]
    return [str(item) for item in value]


class BaseToolConfig(ABC):
    """Abstract base class for tool configuration."""

    @abstractmethod
    def get_workspace_config(self) -> Optional[Dict[str, Any]]:
        """Get workspace configuration."""
        pass

    @abstractmethod
    def get_vision_model(self) -> Optional[Any]:
        """Get vision model."""
        pass

    @abstractmethod
    def get_image_models(self) -> Dict[str, Any]:
        """Get image models."""
        pass

    def get_video_models(self) -> Dict[str, Any]:
        """Get video models."""
        return {}

    @abstractmethod
    def get_asr_models(self) -> Dict[str, Any]:
        """Get ASR (speech-to-text) models."""
        pass

    @abstractmethod
    def get_tts_models(self) -> Dict[str, Any]:
        """Get TTS (text-to-speech) models."""
        pass

    def get_sound_effect_models(self) -> Dict[str, Any]:
        """Get sound effect generation models."""
        return {}

    def get_music_models(self) -> Dict[str, Any]:
        """Get music generation models."""
        return {}

    @abstractmethod
    async def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
        """Get MCP server configurations."""
        pass

    def release_db_connection(self) -> None:
        """Release any pooled DB connection held by this config's session.

        Tool creators call this right before long non-DB awaits (e.g. remote
        MCP initialize/list-tools) so a config backed by a live SQLAlchemy
        session does not pin a pool slot in ``idle in transaction`` for the
        whole wait (issue #889). The session must remain usable afterwards —
        it re-acquires a connection on its next query. Default: no-op for
        configs without a DB session.
        """

    @abstractmethod
    def get_file_tools_enabled(self) -> bool:
        """Whether to include file tools."""
        pass

    @abstractmethod
    def get_basic_tools_enabled(self) -> bool:
        """Whether to include basic tools."""
        pass

    @abstractmethod
    def get_embedding_model(self) -> Optional[str]:
        """Get embedding model ID."""
        pass

    def get_rerank_model(self) -> Optional[str]:
        """Get rerank model ID (registered in model hub).

        Default implementation returns ``None``; web/tool implementations
        should resolve the user's default rerank model from the database.
        """
        return None

    @abstractmethod
    def get_browser_tools_enabled(self) -> bool:
        """Whether to include browser automation tools."""
        pass

    @abstractmethod
    def get_task_id(self) -> Optional[str]:
        """Get task ID for session tracking."""
        pass

    @abstractmethod
    def get_allowed_collections(self) -> Optional[List[str]]:
        """Get allowed knowledge base collections. None means all collections are allowed."""
        pass

    @abstractmethod
    def get_allowed_skills(self) -> Optional[List[str]]:
        """Get allowed skill names. None means all skills are allowed."""
        pass

    @abstractmethod
    def get_user_id(self) -> Optional[int]:
        """Get current user ID for multi-tenancy."""
        pass

    @abstractmethod
    def is_admin(self) -> bool:
        """Whether current user is admin."""
        pass

    @abstractmethod
    def get_enable_agent_tools(self) -> bool:
        """Whether to include published agents as tools."""
        pass

    @abstractmethod
    def get_image_generate_model(self) -> Optional[Any]:
        """Get default image generation model."""
        pass

    @abstractmethod
    def get_custom_api_configs(self) -> List[Dict[str, Any]]:
        """Get custom API configurations."""
        pass

    @abstractmethod
    def get_image_edit_model(self) -> Optional[Any]:
        """Get default image editing model."""
        pass

    def get_video_model(self) -> Optional[Any]:
        """Get default video generation model."""
        return None

    def get_sound_effect_model(self) -> Optional[Any]:
        """Get default sound effect generation model."""
        return None

    def get_music_model(self) -> Optional[Any]:
        """Get default music generation model."""
        return None

    @abstractmethod
    def get_sandbox(self) -> Optional[Any]:
        """Get sandbox instance for sandboxed executors. Returns None if not available."""
        pass

    def get_tool_credential(self, tool_name: str, field_name: str) -> Optional[str]:
        return None

    def get_sql_connections(self) -> Dict[str, str]:
        return {}

    def get_allowed_agent_ids(self) -> Optional[List[int]]:
        """Get explicitly allowed published agent IDs. None means use defaults."""
        return None

    def get_agent_tool_overrides(self) -> Dict[int, Dict[str, Any]]:
        """Get per-agent tool metadata/runtime overrides for delegation."""
        return {}

    def get_a2a_agent_configs(self) -> List[Dict[str, Any]]:
        """Get remote A2A agent tool configurations.

        Private endpoints are rejected unless an entry explicitly sets
        ``allow_private_networks`` to ``True``.
        """
        return []

    def get_enable_global_agent_tools(self) -> bool:
        """Whether to include globally visible published agents as tools."""
        return True

    def get_allow_cross_user_agent_ids(self) -> bool:
        """Whether explicit allowed agent IDs may cross the current user boundary."""
        return False

    def get_parent_task_id(self) -> Optional[str]:
        """Get parent task ID for delegated tool execution."""
        return None

    def get_parent_tracer(self) -> Optional[Any]:
        """Get parent tracer for delegated tool execution."""
        return None

    def get_agent_call_stack(self) -> List[int]:
        """Get active agent delegation call stack for recursion prevention."""
        return []

    def get_excluded_agent_id(self) -> Optional[int]:
        """Get agent ID to exclude from agent tools."""
        return None

    @abstractmethod
    def get_db(self) -> Optional[Any]:
        """Get database session. Returns None for standalone usage."""
        pass

    @abstractmethod
    def get_asr_model(self) -> Optional[Any]:
        """Get default ASR (speech-to-text) model."""
        pass

    @abstractmethod
    def get_tts_model(self) -> Optional[Any]:
        """Get default TTS (text-to-speech) model."""
        pass

    @abstractmethod
    def get_llm(self) -> Optional[Any]:
        """Get default LLM for general tasks."""
        pass

    def get_max_output_length(self) -> int:
        """Get maximum output length in characters.

        Reads from XAGENT_TOOL_MAX_OUTPUT_LENGTH env var if set.
        See :mod:`xagent.config` for details.
        """
        return _root_config.get_tool_max_output_length()

    def get_max_field_count(self) -> int:
        """Get maximum number of fields/items in dict/list for output filtering.

        Reads from XAGENT_TOOL_MAX_FIELD_COUNT env var if set.
        See :mod:`xagent.config` for details.
        """
        return _root_config.get_tool_max_field_count()

    def get_max_recursion_depth(self) -> int:
        """Get maximum recursion depth for output filtering.

        Reads from XAGENT_TOOL_MAX_RECURSION_DEPTH env var if set.
        See :mod:`xagent.config` for details.
        """
        return _root_config.get_tool_max_recursion_depth()


class ToolConfig(BaseToolConfig):
    """Tool configuration that uses provided config dict for standalone usage."""

    def __init__(self, config_dict: Dict[str, Any]):
        # Extract configurations from dict
        workspace_config = config_dict.get("workspace")
        config_dict.get("vision_model")  # Unused in base config
        config_dict.get("image_models", [])  # Unused in base config
        config_dict.get("video_models", [])  # Unused in base config
        config_dict.get("asr_models", [])  # Unused in base config
        config_dict.get("tts_models", [])  # Unused in base config
        sound_effect_models = config_dict.get("sound_effect_models") or {}
        sound_effect_model = config_dict.get("sound_effect_model")
        music_models = config_dict.get("music_models") or {}
        music_model = config_dict.get("music_model")
        mcp_server_configs = config_dict.get("mcp_servers", [])
        file_tools_enabled = config_dict.get("file_tools_enabled", True)
        basic_tools_enabled = config_dict.get("basic_tools_enabled", True)
        embedding_model = config_dict.get("embedding_model")
        browser_tools_enabled = config_dict.get("browser_tools_enabled", True)
        task_id = config_dict.get("task_id")
        allowed_collections = config_dict.get("allowed_collections")
        allowed_skills = config_dict.get("allowed_skills")
        allowed_tools = config_dict.get("allowed_tools")
        allowed_agent_ids = config_dict.get("allowed_agent_ids")
        agent_tool_overrides = config_dict.get("agent_tool_overrides") or {}
        a2a_agent_configs = config_dict.get("a2a_agent_configs") or []
        enable_global_agent_tools = config_dict.get("enable_global_agent_tools", True)
        allow_cross_user_agent_ids = config_dict.get(
            "allow_cross_user_agent_ids", False
        )
        parent_task_id = config_dict.get("parent_task_id")
        parent_tracer = config_dict.get("parent_tracer")
        agent_call_stack = config_dict.get("agent_call_stack") or []
        user_id = config_dict.get("user_id")
        is_admin = config_dict.get("is_admin", False)
        tool_credentials = config_dict.get("tool_credentials", {})

        # Output limit configuration (uses environment variable as default)
        # Store custom values if provided, otherwise use None to fall back to base class defaults
        self._custom_max_output_length: int | None = None
        try:
            self._custom_max_output_length = int(
                config_dict.get("max_output_length")  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            pass
        self._custom_max_field_count: int | None = None
        try:
            self._custom_max_field_count = int(
                config_dict.get("max_field_count")  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            pass
        self._custom_max_recursion_depth: int | None = None
        try:
            self._custom_max_recursion_depth = int(
                config_dict.get("max_recursion_depth")  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            pass

        self.workspace_config: Optional[Dict[str, Any]] = workspace_config
        self.vision_model: Optional[Any] = (
            None  # Standalone usage typically doesn't have web context
        )
        self.image_models: Dict[
            str, Any
        ] = {}  # Standalone usage typically doesn't have web context
        self.video_models: Dict[
            str, Any
        ] = {}  # Standalone usage typically doesn't have web context
        self.asr_models: Dict[
            str, Any
        ] = {}  # Standalone usage typically doesn't have web context
        self.tts_models: Dict[
            str, Any
        ] = {}  # Standalone usage typically doesn't have web context
        self.sound_effect_models: Dict[str, Any] = (
            sound_effect_models if isinstance(sound_effect_models, dict) else {}
        )
        self.sound_effect_model: Optional[Any] = sound_effect_model
        self.music_models: Dict[str, Any] = (
            music_models if isinstance(music_models, dict) else {}
        )
        self.music_model: Optional[Any] = music_model
        self.mcp_server_configs: List[Dict[str, Any]] = mcp_server_configs
        self.file_tools_enabled: bool = bool(file_tools_enabled)
        self.basic_tools_enabled: bool = bool(basic_tools_enabled)
        self.embedding_model: Optional[str] = embedding_model
        self.browser_tools_enabled: bool = bool(browser_tools_enabled)
        self.task_id: Optional[str] = task_id
        self.allowed_collections: Optional[List[str]] = allowed_collections
        self.allowed_skills: Optional[List[str]] = allowed_skills
        self.allowed_tools: Optional[List[str]] = allowed_tools
        self.allowed_agent_ids: Optional[List[int]] = allowed_agent_ids
        self.agent_tool_overrides: Dict[int, Dict[str, Any]] = (
            agent_tool_overrides if isinstance(agent_tool_overrides, dict) else {}
        )
        self.a2a_agent_configs: List[Dict[str, Any]] = (
            a2a_agent_configs if isinstance(a2a_agent_configs, list) else []
        )
        self.enable_global_agent_tools: bool = bool(enable_global_agent_tools)
        self.allow_cross_user_agent_ids: bool = bool(allow_cross_user_agent_ids)
        self.parent_task_id: Optional[str] = parent_task_id
        self.parent_tracer: Optional[Any] = parent_tracer
        self.agent_call_stack: List[int] = list(agent_call_stack)
        self.user_id: Optional[int] = user_id
        self.is_admin_value: bool = bool(is_admin)
        self.tool_credentials: Dict[str, Dict[str, str]] = tool_credentials

    def get_workspace_config(self) -> Optional[Dict[str, Any]]:
        return self.workspace_config

    def get_vision_model(self) -> Optional[Any]:
        return self.vision_model

    def get_image_models(self) -> Dict[str, Any]:
        return self.image_models

    def get_video_models(self) -> Dict[str, Any]:
        return self.video_models

    def get_asr_models(self) -> Dict[str, Any]:
        return self.asr_models

    def get_tts_models(self) -> Dict[str, Any]:
        return self.tts_models

    def get_sound_effect_models(self) -> Dict[str, Any]:
        return self.sound_effect_models

    def get_music_models(self) -> Dict[str, Any]:
        return self.music_models

    async def get_mcp_server_configs(self) -> List[Dict[str, Any]]:
        return self.mcp_server_configs

    def get_file_tools_enabled(self) -> bool:
        return self.file_tools_enabled

    def get_basic_tools_enabled(self) -> bool:
        return self.basic_tools_enabled

    def get_embedding_model(self) -> Optional[str]:
        return self.embedding_model

    def get_browser_tools_enabled(self) -> bool:
        return self.browser_tools_enabled

    def get_task_id(self) -> Optional[str]:
        return self.task_id

    def get_allowed_collections(self) -> Optional[List[str]]:
        return self.allowed_collections

    def get_allowed_skills(self) -> Optional[List[str]]:
        return self.allowed_skills

    def get_user_id(self) -> Optional[int]:
        return self.user_id

    def is_admin(self) -> bool:
        return self.is_admin_value

    def get_enable_agent_tools(self) -> bool:
        return True

    def get_image_generate_model(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_custom_api_configs(self) -> List[Dict[str, Any]]:
        return []  # Standalone config doesn't have web context for custom APIs by default

    def get_image_edit_model(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_video_model(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_sound_effect_model(self) -> Optional[Any]:
        return self.sound_effect_model

    def get_music_model(self) -> Optional[Any]:
        return self.music_model

    def get_asr_model(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_tts_model(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_llm(self) -> Optional[Any]:
        return None  # Standalone config doesn't have web context

    def get_allowed_tools(self) -> Optional[List[str]]:
        """Legacy raw-list accessor.

        Kept for backward compat with callers that haven't migrated to
        :class:`ToolSelectionSpec`. New code SHOULD construct a spec via
        :meth:`ToolSelectionSpec.from_raw` and pass it through
        :attr:`_tool_selection_spec` instead. The factory consults the
        spec first; this method only fires for the no-spec path.
        """
        return self.allowed_tools

    def get_tool_selection_spec(self) -> Optional[Any]:
        """Typed spec accessor (preferred over :meth:`get_allowed_tools`).

        Subclasses set ``self._tool_selection_spec`` to a
        :class:`ToolSelectionSpec` instance constructed via
        :meth:`ToolSelectionSpec.from_raw`. The factory reads this in
        ``create_all_tools`` and dispatches mode-aware filtering through
        ``spec.compute_allowed_names(tools)``.
        """
        return getattr(self, "_tool_selection_spec", None)

    def get_allowed_agent_ids(self) -> Optional[List[int]]:
        return self.allowed_agent_ids

    def get_agent_tool_overrides(self) -> Dict[int, Dict[str, Any]]:
        return self.agent_tool_overrides

    def get_a2a_agent_configs(self) -> List[Dict[str, Any]]:
        return self.a2a_agent_configs

    def get_enable_global_agent_tools(self) -> bool:
        return self.enable_global_agent_tools

    def get_allow_cross_user_agent_ids(self) -> bool:
        return self.allow_cross_user_agent_ids

    def get_parent_task_id(self) -> Optional[str]:
        return self.parent_task_id

    def get_parent_tracer(self) -> Optional[Any]:
        return self.parent_tracer

    def get_agent_call_stack(self) -> List[int]:
        return self.agent_call_stack

    def get_sandbox(self) -> Optional[Any]:
        return None  # Standalone config doesn't have sandbox

    def get_tool_credential(self, tool_name: str, field_name: str) -> Optional[str]:
        tool_data = self.tool_credentials.get(tool_name)
        if not isinstance(tool_data, dict):
            return None
        value = tool_data.get(field_name)
        return value if isinstance(value, str) and value else None

    def get_sql_connections(self) -> Dict[str, str]:
        return {}

    def get_max_output_length(self) -> int:
        if self._custom_max_output_length is not None:
            return self._custom_max_output_length
        return super().get_max_output_length()

    def get_max_field_count(self) -> int:
        if self._custom_max_field_count is not None:
            return self._custom_max_field_count
        return super().get_max_field_count()

    def get_max_recursion_depth(self) -> int:
        if self._custom_max_recursion_depth is not None:
            return self._custom_max_recursion_depth
        return super().get_max_recursion_depth()

    def get_db(self) -> Optional[Any]:
        """ToolConfig (standalone) does not have database access."""
        return None
