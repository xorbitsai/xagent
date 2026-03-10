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
   - Is this a presentation/slide? → Do NOT select poster-design
   - Is this a document/report? → Do NOT select poster-design
   - Is this a web page? → Do NOT select poster-design
   - Is this a knowledge base QA/evidence retrieval? → Consider evidence-based-rag

2. **Check for NEGATIVE signals**
   - If user wants "slide", "presentation", "deck" → Reject poster-design
   - If user wants "document", "report" → Reject poster-design
   - If user wants "web page", "landing page" → Reject poster-design
   - If user wants "code", "script" → Reject all non-coding skills

3. **Select ONLY when:**
   - The skill's PRIMARY purpose matches the task type
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
3. Verify there are NO conflicting signals
4. Only then select the skill

If no skill is directly relevant, return selected: false."""

    def __init__(self, llm: Any) -> None:
        """
        Args:
            llm: BaseLLM instance
        """
        self.llm = llm

    async def select(
        self,
        task: str,
        candidates: List[Dict],
        agent_instructions: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Select the most appropriate skill, or return None

        Args:
            task: User task
            candidates: List of candidate skills
            agent_instructions: Optional agent instructions to guide selection

        Returns:
            Selected skill, or None
        """
        if not candidates:
            logger.warning("No candidate skills available for selection")
            return None

        logger.info(f"Selecting skill for task: {task[:100]}...")
        logger.info(f"Available candidates: {len(candidates)} skills")
        if agent_instructions:
            logger.info(f"Agent instructions provided: {len(agent_instructions)} chars")

        # Build system message with agent instructions if provided
        system_message = self.SELECTOR_SYSTEM
        if agent_instructions:
            system_message += f"""

## AGENT INSTRUCTIONS (CRITICAL - MUST FOLLOW)
{agent_instructions}

Follow these AGENT INSTRUCTIONS above all else when selecting a skill.
"""

        prompt = self._build_prompt(task, candidates)

        logger.info("Calling LLM for skill selection...")

        # First try JSON mode, fall back to normal mode if not supported
        try:
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"JSON mode not supported, falling back to normal mode: {e}")
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": system_message},
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

        if not result.get("selected"):
            reasoning = result.get("reasoning", "No reasoning provided")
            logger.info(f"No skill selected. Reasoning: {reasoning}")
            return None

        skill_name = result.get("skill_name")
        reasoning = result.get("reasoning", "No reasoning provided")

        # Find the selected skill
        selected_skill = next((s for s in candidates if s["name"] == skill_name), None)

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

    def _build_prompt(self, task: str, candidates: List[Dict]) -> str:
        """Build selection prompt"""
        skills_desc = []

        for i, skill in enumerate(candidates):
            desc = f"""{i + 1}. **{skill["name"]}**
   Description: {skill.get("description", "N/A")}
   When to use: {skill.get("when_to_use", "N/A")}
   Tags: {", ".join(skill.get("tags", []))}"""
            skills_desc.append(desc)

        # Extract key signal words from task
        task_lower = task.lower()
        signal_words = {
            "slide": "slide" in task_lower or "presentation" in task_lower,
            "poster": "poster" in task_lower or "banner" in task_lower,
            "document": "document" in task_lower or "report" in task_lower,
            "web": "web" in task_lower
            or "landing" in task_lower
            or "html page" in task_lower,
            "code": "code" in task_lower
            or "script" in task_lower
            or "fix bug" in task_lower,
            "knowledge_base_qa": "knowledge base" in task_lower
            or "evidence" in task_lower
            or "verification" in task_lower
            or "due diligence" in task_lower
            or "retrieval" in task_lower,
        }

        detected_types = [k for k, v in signal_words.items() if v]

        return f"""## User Task
{task}

## Detected Task Types
{", ".join(detected_types) if detected_types else "General task (no specific type detected)"}

## Available Skills
{chr(10).join(skills_desc)}

## Important
- Analyze the TRUE INTENT, not just keyword matches
- Consider the OUTPUT TYPE the user wants
- Check for NEGATIVE signals before selecting

Respond with JSON:
{{"selected": true/false, "skill_name": "name of selected skill (or null)", "reasoning": "brief explanation of why this skill is (not) suitable for the task type"}}"""
