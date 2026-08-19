"""
Template Manager - Manages the scanning and retrieval of templates
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Mirrors xagent.web.models.agent.ExecutionMode's values. Not imported
# directly to avoid pulling the SQLAlchemy-backed web.models layer into
# this DB-agnostic YAML loader for one small, stable set of literals.
_VALID_EXECUTION_MODES = frozenset({"flash", "balanced", "think", "auto"})


class TemplateManager:
    """Core manager for the Template system"""

    def __init__(self, templates_root: Path):
        """
        Args:
            templates_root: Path to the templates directory
        """
        self.templates_root = Path(templates_root)

        # Ensure directory exists
        self.templates_root.mkdir(parents=True, exist_ok=True)

        self._templates_cache: Dict[str, Dict] = {}
        self._initialized = False
        self._init_task: Optional[Any] = None

    async def ensure_initialized(self) -> None:
        """Ensure initialization is complete (lazy loading)"""
        if self._initialized:
            return

        # If there is already an initialization task running, wait for it to complete
        if self._init_task is not None:
            await self._init_task
            return

        # Create and execute the initialization task
        self._init_task = asyncio.create_task(self._do_initialize())
        await self._init_task

    async def _do_initialize(self) -> None:
        """Actual initialization logic"""
        await self.initialize()
        self._init_task = None

    async def initialize(self) -> None:
        """Initialization: scan all templates"""
        logger.info("📂 Scanning templates...")
        logger.info(f"  from {self.templates_root}...")
        await self.reload()
        self._initialized = True
        logger.info(f"✓ Loaded {len(self._templates_cache)} templates")

    async def reload(self) -> None:
        """Reload all templates"""
        self._templates_cache.clear()

        if not self.templates_root.exists():
            logger.warning(f"Templates directory does not exist: {self.templates_root}")
            return

        logger.debug(f"Scanning directory: {self.templates_root}")
        found_count = 0

        for yaml_file in self.templates_root.glob("*.yaml"):
            try:
                template_info = self._parse_yaml_file(yaml_file)
                template_id = template_info.get("id")
                if not template_id:
                    logger.warning(f"Skipping {yaml_file.name}: missing 'id' field")
                    continue

                self._templates_cache[template_id] = template_info
                logger.info(f"  ✓ Loaded: {template_info['name']}")
                found_count += 1
            except Exception as e:
                logger.error(f"  ✗ Error loading {yaml_file.name}: {e}", exc_info=True)

        self._warn_on_dangling_workforce_references()

        logger.info(f"Total templates loaded: {len(self._templates_cache)}")

    def _warn_on_dangling_workforce_references(self) -> None:
        """A workforce template's `workforce_config.agents[].template_id`
        values are only checked at parse time for being non-empty strings
        (per-file validation can't know about other files yet, and can't
        know the referenced template's own `type`). Once every template is
        loaded, cross-check each one against the full cache so a typo'd or
        wrongly-typed reference is logged loudly at startup instead of only
        surfacing as a 400 the first time a user clicks "Use" on that
        template. Logged at `warning`, not `error`: this is a template
        authoring problem the process can fully continue past, not a
        request-time failure.
        """
        for template in self._templates_cache.values():
            if template.get("type") != "workforce":
                continue
            workforce_config = template.get("workforce_config") or {}
            for agent in workforce_config.get("agents") or []:
                referenced_id = agent.get("template_id")
                if not referenced_id:
                    continue
                referenced_template = self._templates_cache.get(referenced_id)
                if referenced_template is None:
                    logger.warning(
                        "Workforce template %r references unknown template_id "
                        "%r in workforce_config.agents - instantiating it will "
                        "fail until this is fixed",
                        template.get("id"),
                        referenced_id,
                    )
                elif referenced_template.get("type", "agent") != "agent":
                    # A worker must resolve to a single-agent template - see
                    # the matching runtime guard in
                    # workforce_creator._get_or_create_quick_access_worker_agent,
                    # which would otherwise try to read a null agent_config
                    # off the referenced (workforce-type) template.
                    logger.warning(
                        "Workforce template %r references template_id %r in "
                        "workforce_config.agents, but that template's type is "
                        "%r, not 'agent' - instantiating it will fail until "
                        "this is fixed",
                        template.get("id"),
                        referenced_id,
                        referenced_template.get("type"),
                    )

    def _parse_yaml_file(self, yaml_file: Path) -> Dict[str, Any]:
        """Parse a single YAML file"""
        with open(yaml_file, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        # Validate required fields
        required_fields = ["id", "name", "category", "descriptions"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate descriptions contains English
        descriptions = data.get("descriptions", {})
        if not isinstance(descriptions, dict):
            raise ValueError("'descriptions' must be a dictionary")
        if "en" not in descriptions:
            raise ValueError("'descriptions' must contain at least 'en' key")

        # Ensure agent_config exists
        if "agent_config" not in data:
            data["agent_config"] = {}

        # Set default values. tags/features/sample_prompts are per-locale
        # dicts (like descriptions), so their "not authored" default must be
        # {} not [] - get_localized_value falls back to [] per-call anyway,
        # but a [] here would silently bypass localization if ever read
        # directly instead of through get_localized_value.
        data.setdefault("tags", {})
        data.setdefault("features", {})
        data.setdefault("connections", [])
        data.setdefault("setup_time", "5 min setup")
        data.setdefault("author", "Xagent")
        data.setdefault("version", "1.0")
        data.setdefault("featured", False)
        data.setdefault("sample_prompts", {})
        self._validate_sample_prompts(data["sample_prompts"])
        data.setdefault("persona", None)
        self._validate_persona(data["persona"])

        # agent_config default values
        agent_config = data["agent_config"]
        agent_config.setdefault("instructions", "")
        agent_config.setdefault("skills", [])
        agent_config.setdefault("tool_categories", [])
        agent_config.setdefault("execution_mode", "balanced")

        # type distinguishes a single-agent template (default) from a
        # workforce template, which is instantiated as a manager agent plus
        # N worker agents rather than a single agent.
        data.setdefault("type", "agent")
        if data["type"] not in ("agent", "workforce"):
            raise ValueError(
                f"'type' must be 'agent' or 'workforce', got {data['type']!r}"
            )
        data.setdefault("workforce_config", None)
        if data["type"] == "workforce":
            self._validate_workforce_config(data["workforce_config"])

        return data

    @staticmethod
    def _require_string(
        container: Dict[str, Any], key: str, *, path: str, required: bool = True
    ) -> None:
        """Enforce that `container[key]` is a real (optionally absent)
        string, and write the stripped value back in place.

        Strict `isinstance` rather than `str(...)` coercion: a list/dict/int
        that YAML happened to parse for one of these fields used to slip
        through load-time validation and only surface downstream - as a
        garbage prompt or template id, or as an `AttributeError` → 500 when
        `normalize_text(...).strip()` hit a non-string `description`/`alias`.
        Writing the stripped value back also keeps the duplicate-check and
        the runtime lookup seeing the same `template_id` - previously a
        template id with surrounding whitespace was stripped for the
        duplicate check but looked up raw, guaranteeing a "references an
        unknown template" 400 at use time.
        """
        value = container.get(key)
        if value is None and not required:
            return
        if not isinstance(value, str) or not value.strip():
            requirement = "a non-empty string" if required else "a string or omitted"
            raise ValueError(f"'{path}.{key}' must be {requirement}")
        container[key] = value.strip()

    def _validate_workforce_config(self, workforce_config: Any) -> None:
        """Validate the shape of `workforce_config` for a workforce-type
        template, normalizing (stripping) its string fields in place. A
        workforce template is instantiated as a fresh manager agent
        (defined inline via `manager.instructions`) plus one worker agent
        per `agents[]` entry, each reused/created from an existing template
        id (`agents[].template_id`) - see `create_workforce_from_template`
        in `xagent.web.services.workforce_creator`.
        """
        if not isinstance(workforce_config, dict):
            raise ValueError(
                "'workforce_config' is required and must be a mapping when 'type' is 'workforce'"
            )

        manager = workforce_config.get("manager")
        if not isinstance(manager, dict):
            raise ValueError("'workforce_config.manager' must be a mapping")
        manager_path = "workforce_config.manager"
        self._require_string(manager, "instructions", path=manager_path)
        self._require_string(manager, "name", path=manager_path)
        self._require_string(manager, "description", path=manager_path, required=False)
        execution_mode = manager.get("execution_mode")
        if execution_mode is not None and execution_mode not in _VALID_EXECUTION_MODES:
            # Passed straight into AgentStore.add_agent with no further
            # validation - a typo here would only surface as a broken
            # manager agent at instantiation time, out of step with how
            # carefully the rest of workforce_config is checked at load
            # time.
            raise ValueError(
                "'workforce_config.manager.execution_mode' must be one of "
                f"{sorted(_VALID_EXECUTION_MODES)}, got {execution_mode!r}"
            )
        for list_field in ("tool_categories", "skills"):
            value = manager.get(list_field)
            if value is not None and (
                not isinstance(value, list)
                or not all(isinstance(v, str) for v in value)
            ):
                raise ValueError(
                    f"'workforce_config.manager.{list_field}' must be a list of strings"
                )

        agents = workforce_config.get("agents")
        if not isinstance(agents, list) or not agents:
            raise ValueError("'workforce_config.agents' must be a non-empty list")
        seen_template_ids: set[str] = set()
        for index, agent in enumerate(agents):
            if not isinstance(agent, dict):
                raise ValueError(
                    f"'workforce_config.agents[{index}]' must be a mapping"
                )
            agent_path = f"workforce_config.agents[{index}]"
            self._require_string(agent, "template_id", path=agent_path)
            self._require_string(agent, "assignment_instructions", path=agent_path)
            self._require_string(agent, "alias", path=agent_path, required=False)
            # Required so every worker has a stable display name - the
            # card's agent-count badge used to silently omit a nameless
            # worker rather than surface the gap, undercounting.
            self._require_string(agent, "name", path=agent_path)

            template_id = agent["template_id"]
            if template_id in seen_template_ids:
                # Two agents[] entries resolving to the same quick-access
                # worker agent make the template permanently unusable: the
                # second create_workforce_worker() call 409s on the
                # (workforce_id, agent_id) unique constraint every time this
                # template is instantiated.
                raise ValueError(
                    f"'workforce_config.agents' has a duplicate template_id: "
                    f"{template_id!r}"
                )
            seen_template_ids.add(template_id)

    def _validate_sample_prompts(self, sample_prompts: Any) -> None:
        """Validate the (optional) sample_prompts shape at parse time, the
        same way `descriptions` is validated above. Without this, a
        malformed entry only surfaces as an uncaught pydantic
        ValidationError deep inside the TemplateInfo response model,
        which would take down GET /api/templates/ - and therefore the
        whole template list - for every user, not just break this one
        template.
        """
        if not sample_prompts:
            return
        if not isinstance(sample_prompts, dict):
            raise ValueError(
                "'sample_prompts' must be a dict keyed by locale (e.g. "
                "{'en': [...], 'zh': [...]}), not a flat list - a flat list "
                "would silently bypass per-locale resolution"
            )
        for locale, prompts in sample_prompts.items():
            if not isinstance(prompts, list):
                raise ValueError(f"'sample_prompts.{locale}' must be a list")
            for index, prompt in enumerate(prompts):
                if not isinstance(prompt, dict):
                    raise ValueError(
                        f"'sample_prompts.{locale}[{index}]' must be a mapping"
                    )
                title = prompt.get("title")
                if not isinstance(title, str) or not title:
                    raise ValueError(
                        f"'sample_prompts.{locale}[{index}].title' must be a non-empty string"
                    )
                prompt_text = prompt.get("prompt")
                if not isinstance(prompt_text, str) or not prompt_text:
                    raise ValueError(
                        f"'sample_prompts.{locale}[{index}].prompt' must be a non-empty string"
                    )
                highlights = prompt.get("highlights", [])
                if not isinstance(highlights, list) or not all(
                    isinstance(h, str) for h in highlights
                ):
                    raise ValueError(
                        f"'sample_prompts.{locale}[{index}].highlights' must be a list of strings"
                    )

    def _validate_persona(self, persona: Any) -> None:
        """Validate the (optional) `persona` shape at parse time - the
        "AI Team Marketplace" card content (display name, avatar, and the
        chat-opening intro/questions a Hire flow seeds). `None` (the
        default) means the template shows up in the marketplace with no
        persona treatment, e.g. a workforce-type template's card is
        rendered from `workforce_config` instead. Locale-keyed fields
        follow the same {'en': ..., 'zh': ...} shape as `descriptions` /
        `sample_prompts` above, validated the same way so a malformed entry
        fails loudly at load time rather than as a 500 the first time
        `TemplateInfo`/`PersonaInfo` tries to build a response from it.
        """
        if persona is None:
            return
        if not isinstance(persona, dict):
            raise ValueError("'persona' must be a mapping or omitted")

        name = persona.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'persona.name' must be a non-empty string")
        persona["name"] = name.strip()

        role = persona.get("role")
        if not isinstance(role, dict) or "en" not in role:
            raise ValueError(
                "'persona.role' must be a dict keyed by locale (e.g. "
                "{'en': ..., 'zh': ...}) with at least an 'en' key"
            )
        for locale, value in role.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"'persona.role.{locale}' must be a non-empty string")

        avatar = persona.get("avatar")
        if avatar is not None and (not isinstance(avatar, str) or not avatar.strip()):
            raise ValueError("'persona.avatar' must be a non-empty string or omitted")
        persona.setdefault("avatar", None)

        intro = persona.get("intro", {})
        if not isinstance(intro, dict):
            raise ValueError(
                "'persona.intro' must be a dict keyed by locale, not a flat string"
            )
        if intro and "en" not in intro:
            # Unlike sample_prompts (which has no primary locale to fall
            # back to), intro is user-facing chat copy seeded verbatim into
            # the first message a Hire flow sends - get_localized_value's
            # fallback-to-'en' would otherwise resolve to "" for an English
            # requester, silently sending a blank opening message instead
            # of failing loudly here at load time.
            raise ValueError(
                "'persona.intro' must contain at least an 'en' key when authored"
            )
        for locale, value in intro.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"'persona.intro.{locale}' must be a non-empty string")
        persona["intro"] = intro

        kickoff_questions = persona.get("kickoff_questions", {})
        if not isinstance(kickoff_questions, dict):
            raise ValueError(
                "'persona.kickoff_questions' must be a dict keyed by locale "
                "(e.g. {'en': [...], 'zh': [...]}), not a flat list"
            )
        if kickoff_questions and "en" not in kickoff_questions:
            # Same fallback-to-empty hazard as intro above.
            raise ValueError(
                "'persona.kickoff_questions' must contain at least an 'en' "
                "key when authored"
            )
        for locale, questions in kickoff_questions.items():
            if not isinstance(questions, list) or not all(
                isinstance(q, str) and q.strip() for q in questions
            ):
                raise ValueError(
                    f"'persona.kickoff_questions.{locale}' must be a list of "
                    "non-empty strings"
                )
        persona["kickoff_questions"] = kickoff_questions

    def _enrich_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Merge connections into agent_config.tool_categories.

        `agent_config` is only meaningful for an 'agent'-type template; a
        'workforce'-type template's real configuration lives in
        `workforce_config` instead (its own top-level `agent_config` is an
        unused parse-time placeholder - see `_parse_yaml_file`). Nulling it
        out here, rather than in each caller, is deliberate: every consumer
        of a template dict (the Templates API, the home page feed, the
        /task quick-access resolver, the v1 SDK API, and anything added
        later) reads this same enriched dict, and a non-agent template
        must fail safe by construction rather than by every caller
        remembering to check `type` first - a workforce template's
        `agent_config` used to leak real (if useless) `instructions`/
        `tool_categories` here, which is exactly what silently produced a
        published, empty-instruction agent on every un-gated path.
        """
        connections = template.get("connections", [])
        template_type = template.get("type", "agent")

        agent_config_dict: Optional[Dict[str, Any]] = None
        if template_type == "agent":
            # The agent_config could be an AgentConfig pydantic model or a dict
            agent_config = template.get("agent_config", {})

            if hasattr(agent_config, "model_dump"):
                agent_config_dict = agent_config.model_dump()
            elif hasattr(agent_config, "dict"):
                agent_config_dict = agent_config.dict()
            else:
                agent_config_dict = dict(agent_config)

            tool_categories = agent_config_dict.get("tool_categories", [])
            if not isinstance(tool_categories, list):
                tool_categories = list(tool_categories) if tool_categories else []

            for conn in connections:
                conn_name = conn.get("name") if isinstance(conn, dict) else conn
                if not conn_name:
                    continue
                mcp_category = f"mcp:{conn_name}"
                if mcp_category not in tool_categories:
                    tool_categories.append(mcp_category)

            agent_config_dict["tool_categories"] = tool_categories

        return {
            "id": template["id"],
            "name": template["name"],
            "category": template.get("category", ""),
            "featured": template.get("featured", False),
            "descriptions": template.get("descriptions", {}),
            "features": template.get("features", {}),
            "sample_prompts": template.get("sample_prompts", {}),
            "persona": template.get("persona"),
            "connections": connections,
            "setup_time": template.get("setup_time", "5 min setup"),
            "tags": template.get("tags", {}),
            "author": template.get("author", ""),
            "version": template.get("version", ""),
            "agent_config": agent_config_dict,
            "type": template_type,
            "workforce_config": template.get("workforce_config"),
        }

    async def list_templates(self) -> List[Dict]:
        """List all templates (summary information)"""
        await self.ensure_initialized()

        result = []
        for template in self._templates_cache.values():
            result.append(self._enrich_template(template))
        return result

    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a single template (full information)"""
        await self.ensure_initialized()
        template = self._templates_cache.get(template_id)
        if template:
            return self._enrich_template(template)
        return None

    def has_templates(self) -> bool:
        """Check if any templates are available"""
        return len(self._templates_cache) > 0
