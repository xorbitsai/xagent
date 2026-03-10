"""
Skill Manager - Manage skill scanning and retrieval
"""

import asyncio
import logging
from pathlib import Path

# TYPE_CHECKING import to avoid circular dependency
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .parser import SkillParser
from .selector import SkillSelector

if TYPE_CHECKING:
    from xagent.core.agent.context import AgentContext

logger = logging.getLogger(__name__)


class SkillManager:
    """Core manager for the skill system"""

    def __init__(self, skills_roots: List[Path]):
        """
        Args:
            skills_roots: List of skills directory paths (supports multiple directories)
                - First is the built-in skills directory (read-only)
                - Subsequent ones are user-defined skills directories (writable)
        """
        self.skills_roots = [Path(p) for p in skills_roots]

        # Ensure user-defined directories exist
        for root in self.skills_roots[1:]:
            root.mkdir(parents=True, exist_ok=True)

        self._skills_cache: Dict[str, Dict] = {}
        self._initialized = False
        self._init_task: Optional[Any] = None

    async def ensure_initialized(self) -> None:
        """Ensure initialization is complete (lazy loading mode)"""
        if self._initialized:
            return

        # If there's an initialization task running, wait for it to complete
        if self._init_task is not None:
            await self._init_task
            return

        # Create and execute initialization task
        self._init_task = asyncio.create_task(self._do_initialize())
        await self._init_task

    async def _do_initialize(self) -> None:
        """Actual initialization logic"""
        await self.initialize()
        self._init_task = None

    async def initialize(self) -> None:
        """Initialize: scan all skills"""
        logger.info("📂 Scanning skills...")
        for root in self.skills_roots:
            logger.info(f"  from {root}...")
        await self.reload()
        self._initialized = True
        logger.info(f"✓ Loaded {len(self._skills_cache)} skills")

    async def reload(self) -> None:
        """Reload all skills"""
        self._skills_cache.clear()

        # Scan all directories in order (later ones override earlier ones)
        for skills_root in self.skills_roots:
            if not skills_root.exists():
                logger.warning(f"Skills directory does not exist: {skills_root}")
                continue

            logger.debug(f"Scanning directory: {skills_root}")
            found_count = 0

            for skill_dir in skills_root.iterdir():
                if not skill_dir.is_dir():
                    continue

                if not (skill_dir / "SKILL.md").exists():
                    logger.debug(f"Skipping {skill_dir.name}: no SKILL.md found")
                    continue

                try:
                    skill_info = SkillParser.parse(skill_dir)
                    self._skills_cache[skill_info["name"]] = skill_info
                    source = (
                        "builtin" if skills_root == self.skills_roots[0] else "user"
                    )
                    logger.info(f"  ✓ Loaded: {skill_info['name']} ({source})")
                    found_count += 1
                except Exception as e:
                    logger.error(
                        f"  ✗ Error loading {skill_dir.name}: {e}", exc_info=True
                    )

            logger.debug(f"Found {found_count} skills in {skills_root}")

        logger.info(f"Total skills loaded: {len(self._skills_cache)}")

    async def select_skill(
        self,
        task: str,
        llm: Any,
        tracer: Optional[Any] = None,
        task_id: Optional[str] = None,
        allowed_skills: Optional[List[str]] = None,
        context: Optional["AgentContext"] = None,
    ) -> Optional[Dict]:
        """
        Select appropriate skill based on task

        Args:
            task: User task
            llm: LLM instance for skill selection
            tracer: Tracer instance for sending trace events (optional)
            task_id: Task ID for trace events (optional)
            allowed_skills: Optional list of allowed skills for filtering
            context: Optional AgentContext containing agent instructions (system_prompt)

        Returns:
            Selected skill, or None
        """
        await self.ensure_initialized()

        if not self._skills_cache:
            logger.debug("No skills available for selection")
            return None

        # Filter by allowed_skills if specified
        candidates = list(self._skills_cache.values())
        if allowed_skills is not None:
            allowed_set = set(allowed_skills)
            candidates = [s for s in candidates if s["name"] in allowed_set]
            logger.info(
                f"Filtered to {len(candidates)} allowed skills: {allowed_skills}"
            )

        if not candidates:
            logger.debug("No skills available after filtering")
            return None

        logger.debug(f"Selecting skill for task: {task[:100]}...")
        logger.debug(f"Available skills: {len(candidates)}")

        # Extract agent instructions from context if available
        agent_instructions = None
        if context and context.state.get("system_prompt"):
            agent_instructions = context.state["system_prompt"]
            logger.debug(
                f"Agent instructions extracted from context ({len(agent_instructions)} chars)"
            )

        # Send skill selection start event if tracer is provided
        if tracer and task_id:
            from xagent.core.agent.trace import (
                trace_skill_select_end,
                trace_skill_select_start,
            )

            await trace_skill_select_start(
                tracer,
                task_id,
                data={
                    "task": task[:200],  # Limit task length
                    "available_skills_count": len(candidates),
                    "allowed_skills": allowed_skills,
                },
            )

        selector = SkillSelector(llm)

        try:
            selected_skill = await selector.select(
                task=task, candidates=candidates, agent_instructions=agent_instructions
            )

            # Send skill selection end event if tracer is provided
            if tracer and task_id:
                from xagent.core.agent.trace import trace_skill_select_end

                await trace_skill_select_end(
                    tracer,
                    task_id,
                    data={
                        "task": task[:200],
                        "selected": selected_skill is not None,
                        "skill_name": selected_skill.get("name")
                        if selected_skill
                        else None,
                    },
                )

            return selected_skill
        except Exception as e:
            # Send skill selection error event if tracer is provided
            if tracer and task_id:
                from xagent.core.agent.trace import trace_error

                await trace_error(
                    tracer,
                    task_id=task_id,
                    error_type="SkillSelectionError",
                    error_message=str(e),
                )
            raise

    async def list_skills(self) -> List[Dict]:
        """List all skills (brief information)"""
        await self.ensure_initialized()
        return [
            {
                "name": skill["name"],
                "description": skill.get("description", ""),
                "when_to_use": skill.get("when_to_use", ""),
                "tags": skill.get("tags", []),
            }
            for skill in self._skills_cache.values()
        ]

    async def get_skill(self, name: str) -> Optional[Dict]:
        """Get single skill (full information including template)"""
        await self.ensure_initialized()
        return self._skills_cache.get(name)

    def has_skills(self) -> bool:
        """Check if there are available skills"""
        return len(self._skills_cache) > 0

    def get_builtin_root(self) -> Path:
        """Get built-in skills directory"""
        return self.skills_roots[0]

    def get_user_root(self) -> Path:
        """Get user-defined skills directory (returns builtin if not available)"""
        if len(self.skills_roots) > 1:
            return self.skills_roots[1]
        return self.skills_roots[0]
