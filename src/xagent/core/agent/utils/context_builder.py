"""
Context builder utilities for managing message-based context in nested agents.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...model.chat.basic.base import BaseLLM
from ..trace import Tracer, trace_compact_end, trace_compact_start
from .compact import CompactConfig, CompactUtils

logger = logging.getLogger(__name__)


@dataclass
class StepExecutionResult:
    """Result of a step execution with complete message history."""

    step_id: str
    messages: List[Dict[str, str]]  # Complete ReAct conversation
    final_result: Dict[str, Any]
    agent_name: Optional[str] = None
    compact_available: bool = True


class ContextBuilder:
    """
    Builds context for DAG steps by merging and optionally compacting
    dependency step messages.
    """

    def __init__(
        self,
        llm: BaseLLM,
        compact_threshold: Optional[int] = None,
        compact_llm: Optional[BaseLLM] = None,
        tracer: Optional[Tracer] = None,
    ):
        """
        Initialize context builder.

        Args:
            llm: Language model for compacting
            compact_threshold: Token threshold for triggering compaction, defaults to CompactConfig.threshold
            compact_llm: Optional LLM for context compaction, defaults to main LLM
            tracer: Optional tracer for compact events
        """
        self.llm = llm
        self.compact_llm = (
            compact_llm or llm
        )  # Use main LLM if compact_llm not provided
        self.tracer = tracer
        # Store configuration for compaction
        self.compact_config = CompactConfig(
            enabled=True, threshold=compact_threshold or CompactConfig().threshold
        )

    async def build_context_for_step(
        self,
        step_name: str,
        step_description: str,
        dependencies: List[str],
        dependency_results: Dict[str, StepExecutionResult],
        task_id: Optional[str] = None,
        original_goal: Optional[str] = None,
        skill_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build context messages for a step based on its dependencies.

        Args:
            step_name: Name of the current step
            step_description: Description of what this step should do
            dependencies: List of dependency step IDs
            dependency_results: Results from dependency steps
            task_id: Optional task ID for tracing
            original_goal: Optional original user goal for context preservation
            skill_context: Optional skill context with domain knowledge and templates

        Returns:
            List of messages forming the context for this step
        """
        # Start with system prompt
        messages = [
            {
                "role": "system",
                "content": self._build_step_system_prompt(
                    step_name, step_description, original_goal, skill_context
                ),
            }
        ]

        if not dependencies:
            return messages

        # Process dependencies individually with pre-compaction
        processed_dependency_messages = []
        individual_threshold = self.compact_config.threshold // max(
            len(dependencies), 1
        )  # Divide threshold among dependencies

        for dep_id in dependencies:
            if dep_id in dependency_results:
                dep_result = dependency_results[dep_id]

                # Check if this dependency alone exceeds individual threshold
                dep_tokens = CompactUtils.estimate_tokens(dep_result.messages)

                # Create separator comment
                separator_msg = {
                    "role": "user",
                    "content": f"=== Results from dependency step: {dep_id} ({dep_result.agent_name or 'unknown'}) ===",
                }

                if dep_tokens > individual_threshold and dep_result.compact_available:
                    logger.info(
                        f"Dependency {dep_id} ({dep_tokens} tokens) exceeds individual threshold ({individual_threshold}), compacting individually with {self.compact_llm.model_name}..."
                    )
                    try:
                        # Trace compact start for individual dependency
                        if self.tracer and task_id:
                            await trace_compact_start(
                                self.tracer,
                                task_id,
                                step_name,
                                data={
                                    "compact_type": "individual_dependency",
                                    "dependency_id": dep_id,
                                    "original_tokens": dep_tokens,
                                    "threshold": individual_threshold,
                                    "compact_model": self.compact_llm.model_name,
                                },
                            )

                        # Compact this dependency individually
                        compacted_dep_messages = (
                            await self._compact_individual_dependency(
                                dep_result.messages, dep_id, step_name, step_description
                            )
                        )

                        # Trace compact end for individual dependency
                        if self.tracer and task_id:
                            compacted_tokens = CompactUtils.estimate_tokens(
                                compacted_dep_messages
                            )
                            await trace_compact_end(
                                self.tracer,
                                task_id,
                                step_name,
                                data={
                                    "compact_type": "individual_dependency",
                                    "dependency_id": dep_id,
                                    "original_tokens": dep_tokens,
                                    "compacted_tokens": compacted_tokens,
                                    "compression_ratio": f"{(compacted_tokens / dep_tokens * 100):.1f}%",
                                    "compact_model": self.compact_llm.model_name,
                                },
                            )

                        # Add separator and compacted messages
                        processed_dependency_messages.append(separator_msg)
                        processed_dependency_messages.extend(compacted_dep_messages)
                    except Exception as e:
                        # Trace compact error
                        if self.tracer and task_id:
                            await trace_compact_end(
                                self.tracer,
                                task_id,
                                step_name,
                                data={
                                    "compact_type": "individual_dependency",
                                    "dependency_id": dep_id,
                                    "original_tokens": dep_tokens,
                                    "error": str(e),
                                    "compact_model": self.compact_llm.model_name,
                                },
                            )

                        logger.error(
                            f"Compact failed for dependency {dep_id}: {e}. This may indicate the dependency content exceeds the model's context length limit."
                        )
                        # Instead of silently truncating, provide a more informative fallback
                        error_msg = {
                            "role": "user",
                            "content": f"ERROR: Could not compact dependency {dep_id} due to size or context length limits. "
                            f"Dependency contains {len(dep_result.messages)} messages ({dep_tokens} tokens). "
                            f"Consider reducing dependency complexity or increasing context limits.",
                        }
                        processed_dependency_messages.append(separator_msg)
                        processed_dependency_messages.append(error_msg)
                else:
                    # Add separator and original messages
                    processed_dependency_messages.append(separator_msg)
                    processed_dependency_messages.extend(dep_result.messages)

        if not processed_dependency_messages:
            return messages

        # Final check: if total still exceeds threshold, compact the whole thing
        total_tokens = CompactUtils.estimate_tokens(processed_dependency_messages)

        if total_tokens > self.compact_config.threshold:
            logger.info(
                f"Total context size ({total_tokens} tokens) still exceeds threshold after individual compaction, compacting entire context with {self.compact_llm.model_name}..."
            )
            try:
                # Trace compact start for entire context
                if self.tracer and task_id:
                    await trace_compact_start(
                        self.tracer,
                        task_id,
                        step_name,
                        data={
                            "compact_type": "entire_context",
                            "original_tokens": total_tokens,
                            "threshold": self.compact_config.threshold,
                            "compact_model": self.compact_llm.model_name,
                        },
                    )

                compact_messages = await self._compact_dependency_messages(
                    processed_dependency_messages, step_name, step_description
                )

                # Trace compact end for entire context
                if self.tracer and task_id:
                    compacted_tokens = CompactUtils.estimate_tokens(compact_messages)
                    await trace_compact_end(
                        self.tracer,
                        task_id,
                        step_name,
                        data={
                            "compact_type": "entire_context",
                            "original_tokens": total_tokens,
                            "compacted_tokens": compacted_tokens,
                            "compression_ratio": f"{(compacted_tokens / total_tokens * 100):.1f}%",
                            "compact_model": self.compact_llm.model_name,
                        },
                    )

                messages.extend(compact_messages)
            except Exception as e:
                # Trace compact error for entire context
                if self.tracer and task_id:
                    await trace_compact_end(
                        self.tracer,
                        task_id,
                        step_name,
                        data={
                            "compact_type": "entire_context",
                            "original_tokens": total_tokens,
                            "error": str(e),
                            "compact_model": self.compact_llm.model_name,
                        },
                    )

                logger.warning(
                    f"Failed to compact total context: {e}, using truncated context"
                )
                # Fallback: use truncated context
                messages.extend(processed_dependency_messages[-20:])  # Last 20 messages
        else:
            messages.extend(processed_dependency_messages)

        return messages

    async def _compact_individual_dependency(
        self,
        messages: List[Dict[str, str]],
        dependency_id: str,
        target_step_name: str,
        target_step_description: str,
    ) -> List[Dict[str, str]]:
        """Compact a single dependency's messages using custom compact logic."""

        if not self.compact_config.enabled:
            return messages

        original_tokens = CompactUtils.estimate_tokens(messages)
        individual_threshold = self.compact_config.threshold // max(
            1, len([messages])
        )  # Rough estimate

        if original_tokens <= individual_threshold:
            return messages

        logger.info(
            f"Compacting dependency {dependency_id} ({original_tokens} tokens) with threshold {individual_threshold}"
        )

        try:
            # Format messages for compaction
            conversation_text = CompactUtils.format_messages_for_compact(messages)

            # Build dependency-specific compaction prompt
            compact_prompt = [
                {
                    "role": "system",
                    "content": "You are tasked with compacting a long conversation history from a previous step. "
                    "Preserve key insights, important data, final results, and relevant context, "
                    "but remove redundant reasoning steps and failed attempts. "
                    "CRITICAL: You must return the response in the exact same format: \n"
                    "USER: message content\n"
                    "ASSISTANT: message content\n"
                    "SYSTEM: message content\n\n"
                    "Each message must start with the role followed by a colon and space.",
                },
                {
                    "role": "user",
                    "content": f"Dependency: {dependency_id}\n"
                    f"Target step: {target_step_name}\n"
                    f"Target task: {target_step_description}\n\n"
                    f"Conversation to compact:\n{conversation_text}\n\n"
                    f"IMPORTANT: Return the compacted conversation in the exact same format shown above. "
                    f"Each line must start with USER:, ASSISTANT:, or SYSTEM: followed by the message content.",
                },
            ]

            # Get compacted response
            response = await self.compact_llm.chat(messages=compact_prompt)
            content = (
                response
                if isinstance(response, str)
                else response.get("content", str(response))
            )

            # Parse back to messages format
            compacted_messages = CompactUtils.parse_compact_response(content)

            # Validate compact result
            if not compacted_messages:
                logger.warning(
                    "Compact resulted in empty messages, using smart truncation fallback"
                )
                # Smart truncation: preserve first system message and last few messages
                return self._smart_truncate_messages(
                    messages, target_tokens=individual_threshold
                )

            final_tokens = CompactUtils.estimate_tokens(compacted_messages)
            tokens_saved = original_tokens - final_tokens

            logger.info(
                f"Successfully compacted dependency {dependency_id}: {len(messages)} -> {len(compacted_messages)} messages, "
                f"{original_tokens} -> {final_tokens} tokens ({tokens_saved} saved)"
            )

            return compacted_messages

        except Exception as e:
            logger.error(f"Dependency compact failed: {e}, using fallback truncation")
            # Fallback: truncate to last N messages
            truncated_messages = CompactUtils.truncate_messages(
                messages, self.compact_config.fallback_truncate_count
            )
            final_tokens = CompactUtils.estimate_tokens(truncated_messages)
            tokens_saved = original_tokens - final_tokens

            logger.info(
                f"Fallback compact for dependency {dependency_id}: {len(messages)} -> {len(truncated_messages)} messages, "
                f"{original_tokens} -> {final_tokens} tokens ({tokens_saved} saved)"
            )

            return truncated_messages

    async def _compact_dependency_messages(
        self,
        messages: List[Dict[str, str]],
        target_step_name: str,
        target_step_description: str,
    ) -> List[Dict[str, str]]:
        """Compact dependency messages using custom compact logic."""

        if not self.compact_config.enabled:
            return messages

        original_tokens = CompactUtils.estimate_tokens(messages)
        if original_tokens <= self.compact_config.threshold:
            return messages

        logger.info(
            f"Compacting total dependency context ({original_tokens} tokens) with threshold {self.compact_config.threshold}"
        )

        try:
            # Format messages for compaction
            conversation_text = CompactUtils.format_messages_for_compact(messages)

            # Build dependency-specific compaction prompt
            compact_prompt = [
                {
                    "role": "system",
                    "content": "You are tasked with compacting a long conversation history from multiple previous steps. "
                    "Preserve key insights, important data, final results, and relevant context, "
                    "but remove redundant reasoning steps and failed attempts. "
                    "CRITICAL: You must return the response in the exact same format: \n"
                    "USER: message content\n"
                    "ASSISTANT: message content\n"
                    "SYSTEM: message content\n\n"
                    "Each message must start with the role followed by a colon and space.",
                },
                {
                    "role": "user",
                    "content": f"Target step: {target_step_name}\n"
                    f"Target task: {target_step_description}\n\n"
                    f"Combined conversation to compact:\n{conversation_text}\n\n"
                    f"IMPORTANT: Return the compacted conversation in the exact same format shown above. "
                    f"Each line must start with USER:, ASSISTANT:, or SYSTEM: followed by the message content.",
                },
            ]

            # Get compacted response
            response = await self.compact_llm.chat(messages=compact_prompt)
            content = (
                response
                if isinstance(response, str)
                else response.get("content", str(response))
            )

            # Parse back to messages format
            compacted_messages = CompactUtils.parse_compact_response(content)

            # Validate compact result
            if not compacted_messages:
                logger.warning(
                    "Compact resulted in empty messages, using smart truncation fallback"
                )
                # Smart truncation: preserve first system message and last few messages
                return self._smart_truncate_messages(
                    messages, target_tokens=self.compact_config.threshold
                )

            final_tokens = CompactUtils.estimate_tokens(compacted_messages)
            tokens_saved = original_tokens - final_tokens

            logger.info(
                f"Successfully compacted total context: {len(messages)} -> {len(compacted_messages)} messages, "
                f"{original_tokens} -> {final_tokens} tokens ({tokens_saved} saved)"
            )

            return compacted_messages

        except Exception as e:
            logger.error(
                f"Total dependency compact failed: {e}, using fallback truncation"
            )
            # Fallback: truncate to last N messages
            truncated_messages = CompactUtils.truncate_messages(
                messages, self.compact_config.fallback_truncate_count
            )
            final_tokens = CompactUtils.estimate_tokens(truncated_messages)
            tokens_saved = original_tokens - final_tokens

            logger.info(
                f"Fallback compact for total context: {len(messages)} -> {len(truncated_messages)} messages, "
                f"{original_tokens} -> {final_tokens} tokens ({tokens_saved} saved)"
            )

            return truncated_messages

    def _smart_truncate_messages(
        self, messages: List[Dict[str, str]], target_tokens: int
    ) -> List[Dict[str, str]]:
        """Smart truncate messages to preserve important content while reducing tokens."""
        if not messages:
            return messages

        # Always keep the first system message if it exists
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]

        # Start with system messages
        result = system_messages.copy()

        # Add messages from the end (most recent) until we reach target token count
        current_tokens = CompactUtils.estimate_tokens(result)
        for msg in reversed(other_messages):
            msg_tokens = CompactUtils.estimate_tokens([msg])
            if current_tokens + msg_tokens > target_tokens:
                break
            result.insert(len(system_messages), msg)  # Insert after system messages
            current_tokens += msg_tokens

        logger.info(
            f"Smart truncation: {len(messages)} -> {len(result)} messages, "
            f"{CompactUtils.estimate_tokens(messages)} -> {current_tokens} tokens"
        )

        return result

    def _build_step_system_prompt(
        self,
        step_name: str,
        step_description: str,
        original_goal: Optional[str] = None,
        skill_context: Optional[str] = None,
    ) -> str:
        """Build system prompt for a step.

        Args:
            step_name: Name of the step
            step_description: Description of the step
            original_goal: Optional overall goal
            skill_context: Optional skill context with domain knowledge
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        goal_context = ""
        if original_goal:
            goal_context = f"\nOVERALL GOAL: {original_goal}\n"
            goal_context += "This step is part of achieving the above overall goal. "
            goal_context += "Always keep the overall goal in mind while executing this specific step. "
            goal_context += (
                "Your step contributes to achieving the larger objective.\n\n"
            )

        # Add skill context if available
        skill_section = ""
        if skill_context:
            skill_section = f"\n{skill_context}\n\n"
            skill_section += (
                "IMPORTANT: The skill above provides domain knowledge and templates. "
            )
            skill_section += "Use this knowledge to improve the quality and relevance of your work.\n\n"

        return f"""You are executing a specific step in a larger plan: {step_name}

{goal_context}{skill_section}Your task: {step_description}

You have access to the results and context from previous dependency steps.
Use this information to complete your specific task effectively.

CRITICAL REQUIREMENTS:
1. If your task involves file operations (save, write, create, read, delete files), you MUST call the appropriate file tool (write_file, read_file, etc.)
2. If your task involves data processing or analysis that requires tools, you MUST call the required tools
3. Describing what you will do is NOT sufficient - you MUST actually execute the tool calls
4. Only analysis steps without tool requirements can be completed without tool calls
5. After completing tool calls, provide a clear final answer confirming the actual results

Remember: Describing what you will do is NOT equivalent to actual execution. You must call the appropriate tools to complete the task.

When you have sufficient information or have completed your task, provide a clear final answer.

Current time: {current_time}"""


class MessageUtils:
    """Utilities for working with message-based contexts."""

    @staticmethod
    def estimate_tokens(messages: List[Dict[str, str]]) -> int:
        """Rough estimation of token count for messages."""
        return CompactUtils.estimate_tokens(messages)

    @staticmethod
    def truncate_messages(
        messages: List[Dict[str, str]], max_messages: int
    ) -> List[Dict[str, str]]:
        """Truncate messages to keep only the most recent ones."""
        return CompactUtils.truncate_messages(messages, max_messages)

    @staticmethod
    def format_messages_for_display(messages: List[Dict[str, str]]) -> str:
        """Format messages for human-readable display."""
        formatted = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown").title()
            content = msg.get("content", "")
            formatted.append(
                f"{i + 1}. {role}: {content[:200]}{'...' if len(content) > 200 else ''}"
            )
        return "\n".join(formatted)
