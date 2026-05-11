"""
Skill Selector - Use LLM to select the most appropriate skill
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SkillSelector:
    """Use LLM to select appropriate skill (JSON mode)"""

    SELECTOR_SYSTEM = """You are a skill selection system. Analyze the user's TRUE INTENT before selecting a skill.

## Critical Rules

1. **Understand the task type FIRST**
   - Match the user's requested final artifact, not adjacent implementation details.
   - Is this a presentation/slide/PPTX/deck? → Select a presentation skill only when the user explicitly asks for that artifact.
   - Is this a poster/image/banner/visual asset? → Prefer a visual/poster/image skill; do NOT select a presentation/document skill just because it can contain images.
   - Is this a document/report? → Do NOT select poster-design
   - Is this a web page? → Do NOT select poster-design
   - Does a skill require a specific source scope (knowledge base, uploaded files, provided documents, repository, private data, etc.)? → Select it only when the user explicitly names or provides that source scope
   - Is this public web research, recent/latest/current news, or open-ended factual discovery? → Do NOT select a source-bound skill unless the user explicitly scopes the task to that source
   - Is this about creating an agent, chatbot, or assistant? → Consider agent-builder

2. **Check for NEGATIVE signals**
   - If user wants "slide", "presentation", "deck" → Reject poster-design
   - If user wants "image", "poster", "banner", "illustration", "visual", or "graphic" without asking for slides/PPTX → Reject presentation skills
   - If user wants "document", "report" → Reject poster-design
   - If user wants "web page", "landing page" → Reject poster-design
   - If user wants "code", "script" → Reject all non-coding skills
   - If user wants "create agent", "build chatbot", "create ai assistant" → Reject all non-agent-creation skills
   - If user asks for "recent", "latest", "current", "today", news, public incidents, or web facts without an explicit private/source scope → Reject skills that are limited to private/source-bound evidence

3. **Select ONLY when:**
   - The skill's PRIMARY purpose matches the task type
   - The skill's output contract matches the final artifact the user asked for
   - The skill is SPECIFICALLY designed for this use case
   - Using the skill would SIGNIFICANTLY improve the result

4. **When in doubt, return selected: false**
   - It's better to use general agent capabilities than to force a wrong skill

## Examples of WRONG Selections

| User Task | Wrong Skill | Why |
|-----------|-------------|-----|
| "Create a presentation slide" | poster-design | User wants slides, not poster |
| "Write a marketing report" | poster-design | User wants document, not visual |
| "Generate HTML landing page" | poster-design | User wants web page, not poster |
| "Fix this Python bug" | any non-coding skill | Task requires coding, not other skills |

## Decision Process

1. Identify the CORE OUTPUT TYPE (slide/poster/document/code/etc)
2. Check if any skill is DESIGNED for that output type
3. Check the skill's output contract against the requested artifact
4. Verify there are NO conflicting signals
5. Only then select the skill

If no skill is directly relevant, return selected: false."""

    def __init__(self, llm: Any) -> None:
        """
        Args:
            llm: BaseLLM instance
        """
        self.llm = llm

    async def select(self, task: str, candidates: List[Dict]) -> Optional[Dict]:
        """
        Select the most appropriate skill, or return None

        Args:
            task: User task
            candidates: List of candidate skills

        Returns:
            Selected skill, or None
        """
        if not candidates:
            logger.warning("No candidate skills available for selection")
            return None

        logger.info(f"Selecting skill for task: {task[:100]}...")
        logger.info(f"Available candidates: {len(candidates)} skills")

        prompt = self._build_prompt(task, candidates)

        logger.info("Calling LLM for skill selection...")

        # First try JSON mode, fall back to normal mode if not supported
        try:
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": self.SELECTOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"JSON mode not supported, falling back to normal mode: {e}")
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": self.SELECTOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ]
            )

        # Handle different return types
        if isinstance(response, str):
            content = response
        elif isinstance(response, dict):
            # Handle dictionary format response (e.g., OpenAI format)
            if "content" in response:
                content = response["content"]
            else:
                content = str(response)
        elif hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        logger.info(f"LLM response received: {len(content)} chars")
        logger.debug(f"Raw response: {content[:500]}...")

        # Try to parse JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown
            logger.warning(
                f"Response is not valid JSON: {e}, trying to extract from markdown"
            )
            content = content.strip()
            # Remove markdown code block markers
            if content.startswith("```"):
                # Find the first newline
                newline_idx = content.find("\n")
                if newline_idx > 0:
                    content = content[newline_idx:].strip()
                # Remove trailing ```
                if content.endswith("```"):
                    content = content[:-3].strip()

            logger.debug(f"Extracted content: {content[:500]}...")

            try:
                result = json.loads(content)
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse JSON after markdown extraction: {e2}")
                logger.error(f"Content was: {content}")
                return None

        if not isinstance(result, dict):
            logger.info("No skill selected. Reasoning: Invalid JSON response")
            return None

        if self._llm_bool(result.get("selected")) is not True:
            reasoning = result.get("reasoning", "No reasoning provided")
            logger.info(f"No skill selected. Reasoning: {reasoning}")
            return None

        skill_name = result.get("skill_name")
        reasoning = result.get("reasoning", "No reasoning provided")

        # Find the selected skill
        selected_skill = next((s for s in candidates if s["name"] == skill_name), None)

        if selected_skill and self._should_reject_selected_skill(result):
            source_reasoning = result.get(
                "source_scope_reasoning", "No source reasoning provided"
            )
            logger.info(
                "Rejected selected skill '%s' after LLM source-scope check. "
                "Reasoning: %s",
                skill_name,
                source_reasoning,
            )
            return None

        if selected_skill:
            logger.info(f"✓ Skill selected: '{skill_name}'")
            logger.info(
                f"  Description: {selected_skill.get('description', 'N/A')[:100]}..."
            )
            logger.info(f"  Reasoning: {reasoning}")
        else:
            logger.error(
                f"LLM selected skill '{skill_name}' but it was not found in candidates!"
            )

        return selected_skill

    def _should_reject_selected_skill(self, result: Dict) -> bool:
        """Reject when the LLM says the selected skill needs a missing source scope."""
        if self._llm_bool(result.get("source_scope_required")) is not True:
            return False
        return self._llm_bool(result.get("source_scope_satisfied")) is not True

    def _llm_bool(self, value: Any) -> bool | None:
        """Normalize boolean fields from LLM JSON responses."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
        return None

    def _build_prompt(self, task: str, candidates: List[Dict]) -> str:
        """Build selection prompt"""
        skills_desc = []

        for i, skill in enumerate(candidates):
            desc = f"""{i + 1}. **{skill["name"]}**
   Description: {skill.get("description", "N/A")}
   When to use: {skill.get("when_to_use", "N/A")}
   Tags: {", ".join(skill.get("tags", []))}"""
            skills_desc.append(desc)

        return f"""## User Task
{task}

## Available Skills
{chr(10).join(skills_desc)}

## Important
- Analyze the TRUE INTENT, not just keyword matches
- Consider the OUTPUT TYPE the user wants and reject skills whose output contract conflicts with it
- Presentation skills require an explicit request for slides, a deck, PPT, PPTX, or editing/reading a presentation file. Do not choose them for standalone images, posters, banners, illustrations, or visual assets.
- Check for NEGATIVE signals before selecting
- For any skill that relies on a particular source scope, select it only when the user explicitly scopes the answer to that source (for example a knowledge base, uploaded/provided documents, repository, or internal/private data). Do not select source-bound skills for public web research or recent/latest/current facts.

Respond with JSON:
{{
  "selected": true/false,
  "skill_name": "name of selected skill (or null)",
  "reasoning": "brief explanation of why this skill is (not) suitable for the task type",
  "source_scope_required": true/false,
  "source_scope_satisfied": true/false,
  "source_scope_reasoning": "brief explanation of the selected skill's source assumptions and whether the user provided the required source scope"
}}"""
