"""
Result analysis logic for DAG plan-execute pattern.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from json_repair import loads as repair_loads

from ....model.chat.basic.base import BaseLLM
from ....model.chat.token_context import add_token_usage
from ...exceptions import LLMResponseError
from ...trace import (
    Tracer,
    trace_memory_generate_end,
    trace_memory_generate_start,
    trace_task_llm_call_end,
    trace_task_llm_call_start,
)
from ...utils.llm_utils import clean_messages, extract_json_from_markdown
from .schemas import GoalCheckResponse

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Handles result analysis and goal achievement checking"""

    def __init__(self, llm: BaseLLM, tracer: Tracer):
        self.llm = llm
        self.tracer = tracer

    async def check_goal_achievement(
        self,
        goal: str,
        history: List[Dict[str, Any]],
        file_outputs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Check if the goal has been achieved and generate memory storage insights and final answer

        Returns a dictionary with:
        - achieved: bool - whether the goal was achieved
        - reason: str - explanation of goal achievement status
        - confidence: float - confidence level (0.0-1.0)
        - final_answer: str - comprehensive final answer if achieved, otherwise analysis of what's missing
        - memory_insights: dict - memory storage insights
        """
        logger.info(
            "Checking if goal has been achieved, generating final answer, and memory insights"
        )

        prompt = self._build_comprehensive_goal_check_prompt(
            goal, history, file_outputs
        )

        try:
            # Trace LLM call for goal check
            goal_check_task_id = f"goal_check_{int(time.time())}"
            await trace_task_llm_call_start(
                self.tracer,
                goal_check_task_id,
                data={
                    "goal": goal,
                    "history_length": len(history),
                    "task_type": "comprehensive_goal_check",
                },
            )

            # Trace LLM call for final answer (separate event for frontend compatibility)
            final_answer_task_id = f"final_answer_{int(time.time())}"
            await trace_task_llm_call_start(
                self.tracer,
                final_answer_task_id,
                data={
                    "goal": goal[:200],
                    "history_length": len(history),
                    "task_type": "final_answer_generation",
                },
            )

            # Trace memory generation start
            await trace_memory_generate_start(
                self.tracer,
                goal_check_task_id,
                data={
                    "goal": goal,
                    "history_length": len(history),
                    "analysis_type": "goal_check_with_memory_insights",
                },
            )

            # Clean messages before sending to LLM
            cleaned_prompt = clean_messages(prompt)

            # Prepare output_config with JSON schema from Pydantic model
            # This ensures the required fields are present and provides type safety
            # Note: ClaudeLLM will automatically fix the schema for API compatibility
            output_config = {
                "format": {
                    "type": "json_schema",
                    "schema": GoalCheckResponse.model_json_schema(),
                }
            }

            # Call LLM for comprehensive goal checking and insight generation
            # Using _call_llm_with_retry for automatic fallback from JSON mode to normal mode
            response = await self._call_llm_with_retry(
                messages=cleaned_prompt,
                output_config=output_config,
            )

            # Debug: log response type and structure
            logger.debug(f"LLM response type: {type(response)}, value: {response!r}")

            # Extract content from response
            if isinstance(response, dict):
                content = response.get("content", response)
            elif isinstance(response, tuple):
                # If response is a tuple, extract first element (content)
                logger.warning(f"Unexpected tuple response from LLM: {response}")
                content = response[0] if len(response) > 0 else str(response)
            else:
                # Direct string or other type
                content = response

            logger.info(
                f"Comprehensive goal check response received (first 500 chars): {str(content)[:500]}"
            )
            logger.debug(
                f"Full response: {content!r}"
            )  # Log full response for debugging

            # Extract JSON from markdown if response_format was specified
            content_before_extraction = content
            content = extract_json_from_markdown(content)

            # Debug: Log if extraction occurred
            if content != content_before_extraction:
                logger.info(
                    "⚠️ JSON extraction changed content (markdown code block found)"
                )
                logger.info(
                    f"   Before (first 150 chars): {str(content_before_extraction)[:150]}"
                )
                logger.info(f"   After (first 150 chars):  {str(content)[:150]}")
            else:
                logger.debug(
                    "✅ No markdown extraction needed (content is already JSON)"
                )

            # Parse response
            try:
                result = None

                # Check if content is a JSON array (starts with '[')
                content_stripped = content.strip()
                if content_stripped.startswith("["):
                    # LLM returned a JSON array instead of object
                    # This might happen if the model wrapped the response in an array
                    logger.warning(
                        "LLM returned JSON array instead of object, attempting to extract first element"
                    )
                    try:
                        parsed_array = repair_loads(content, logging=False)
                        if isinstance(parsed_array, list) and len(parsed_array) > 0:
                            # Use first element of array directly as result
                            result = parsed_array[0]
                            logger.info(
                                f"✅ Extracted first element from JSON array: {type(result).__name__}"
                            )
                        else:
                            raise LLMResponseError(
                                "LLM returned empty or invalid JSON array",
                                response=content,
                                expected_format="JSON object with goal achievement and memory insights",
                            )
                    except LLMResponseError:
                        raise
                    except Exception as e:
                        raise LLMResponseError(
                            f"Failed to parse JSON array response: {str(e)}",
                            response=content,
                            expected_format="JSON object with goal achievement and memory insights",
                            cause=e,
                        )
                else:
                    # Normal parsing: content should be a JSON object
                    # Note: logging must be False to avoid tuple return (json, repair_log)
                    result = repair_loads(content, logging=False)

                    # Debug: Log the parsed result type
                    logger.debug(f"repair_loads returned type: {type(result).__name__}")
                    if isinstance(result, list):
                        logger.debug(
                            f"repair_loads returned list with {len(result)} elements"
                        )
                        if len(result) > 0:
                            logger.debug(
                                f"First element type: {type(result[0]).__name__}"
                            )
                            logger.debug(
                                f"First element (first 200 chars): {str(result[0])[:200]}"
                            )

                # Validate result is a dict
                if not isinstance(result, dict):
                    raise LLMResponseError(
                        f"Comprehensive goal check response is not a JSON object. Got: {type(result).__name__}",
                        response=content,
                        expected_format="JSON object with goal achievement and memory insights",
                    )

                # Validate required fields
                if "achieved" not in result:
                    raise LLMResponseError(
                        "Comprehensive goal check response missing 'achieved' field",
                        response=content,
                        expected_format="JSON object with goal achievement and memory insights",
                    )

                if "reason" not in result:
                    result["reason"] = "No reason provided"

                # Ensure memory insights are present
                if "memory_insights" not in result:
                    result["memory_insights"] = {
                        "should_store": False,
                        "reason": "No memory insights provided",
                        "classification": {
                            "primary_domain": "General Task",
                            "secondary_domains": [],
                            "task_type": "General",
                            "complexity_level": "Medium",
                            "keywords": ["general", "task"],
                        },
                        "execution_insights": "",
                        "failure_analysis": "",
                        "success_factors": "",
                        "learned_patterns": "",
                        "improvement_suggestions": "",
                        "user_preferences": "",
                        "behavioral_patterns": "",
                    }

                # Ensure final_answer is present
                if "final_answer" not in result or not result["final_answer"]:
                    # Generate a simple fallback answer if missing
                    if result.get("achieved", False):
                        result["final_answer"] = (
                            f"Task completed successfully. {result.get('reason', 'Goal achieved.')}"
                        )
                    else:
                        result["final_answer"] = (
                            f"TASK FAILED: {result.get('reason', 'Goal not achieved.')}"
                        )

                logger.info(
                    f"Comprehensive goal check result: achieved={result['achieved']}, "
                    f"reason={result['reason']}, "
                    f"should_store_memory={result['memory_insights']['should_store']}, "
                    f"final_answer_length={len(result.get('final_answer', ''))}"
                )

                # Trace goal check LLM call completion
                await trace_task_llm_call_end(
                    self.tracer,
                    goal_check_task_id,
                    data={
                        "goal": goal[:200],
                        "history_length": len(history),
                        "task_type": "comprehensive_goal_check",
                        "response_length": len(content) if content else 0,
                        "goal_achieved": result["achieved"],
                        "goal_reason": result["reason"],
                        "goal_confidence": result.get("confidence", 0.0),
                        "memory_should_store": result["memory_insights"][
                            "should_store"
                        ],
                        "memory_reason": result["memory_insights"]["reason"],
                    },
                )

                # Trace final answer LLM call completion (for frontend compatibility)
                await trace_task_llm_call_end(
                    self.tracer,
                    final_answer_task_id,
                    data={
                        "goal": goal[:200],
                        "history_length": len(history),
                        "task_type": "final_answer_generation",
                        "response_length": len(result.get("final_answer", "")),
                    },
                )

                # Trace memory generation end
                await trace_memory_generate_end(
                    self.tracer,
                    goal_check_task_id,
                    data={
                        "insights_generated": True,
                        "should_store": result["memory_insights"]["should_store"],
                        "reason": result["memory_insights"]["reason"],
                        "source": "goal_check",
                    },
                )

                return result

            except json.JSONDecodeError as e:
                # Check if response looks like a model rejection/safety filter message
                raise LLMResponseError(
                    f"Failed to parse response as JSON. This might be due to model content filtering. Response: {content}",
                    response=content,
                    expected_format="JSON object with goal achievement and memory insights",
                    cause=e,
                )

        except Exception as e:
            if isinstance(e, LLMResponseError):
                raise

            # Wrap other exceptions
            raise LLMResponseError(
                f"Comprehensive goal achievement check failed: {str(e)}",
                context={"goal": goal[:100], "history_length": len(history)},
                cause=e,
            )

    def _build_comprehensive_goal_check_prompt(
        self,
        goal: str,
        history: List[Dict[str, Any]],
        file_outputs: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build prompt for comprehensive goal achievement checking, final answer generation, and memory insight generation"""

        execution_summary = self._summarize_execution_history(history)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build file outputs section
        file_outputs_section = ""
        if file_outputs:
            file_outputs_section = "\n\nOutput Files:\n"
            for i, file_info in enumerate(file_outputs, 1):
                filename = file_info.get("filename", "")
                relative_path = file_info.get("relative_path", "")
                file_outputs_section += f"  {i}. {filename} (path: {relative_path})\n"
            file_outputs_section += "\n"

        # Detect goal language
        goal_has_chinese = any("\u4e00" <= c <= "\u9fff" for c in goal)
        goal_has_english = any(c.isascii() and c.isalpha() for c in goal)
        detected_language = (
            "Chinese"
            if goal_has_chinese
            else ("English" if goal_has_english else "the same")
        )

        system_prompt = (
            "You are an AI assistant that evaluates task execution and synthesizes comprehensive results.\n"
            "Based on the execution history, you must:\n"
            "1. Determine if the stated goal is complete\n"
            "2. Generate a comprehensive final answer that addresses the original goal\n"
            "3. Analyze whether this task contains learnings worth storing as memory\n\n"
            "CRITICAL: When evaluating goal achievement with user modifications:\n"
            "- If you see [User Modification: '...'] markers, the user changed the original requirements\n"
            "- Evaluate goal achievement based on the LATEST user requirements\n"
            "- If the final result satisfies modified requirements, the goal IS achieved\n\n"
            "CRITICAL LANGUAGE REQUIREMENT: YOU MUST RESPOND IN THE SAME LANGUAGE AS THE ORIGINAL GOAL!\n"
            "If the goal is in English, respond ONLY in English.\n"
            "If the goal is in Chinese, respond ONLY in Chinese.\n"
            "NEVER mix languages. Your entire response must be in the same language as the goal.\n\n"
            "Respond with JSON containing goal achievement status, final answer, and memory insights."
        )

        goal_analysis_prompt = f"""GOAL ACHIEVEMENT ANALYSIS AND FINAL ANSWER:
Original Goal: {goal}
Current Time: {current_time}

Execution History Summary:
{execution_summary}
{file_outputs_section}
IMPORTANT: Your final_answer MUST be in {detected_language} ONLY, matching the goal's language.

Please analyze this task execution comprehensively and respond with JSON:

{{
  "achieved": true/false,
  "reason": "Clear explanation of why the goal was/was not achieved",
  "confidence": 0.0-1.0,
  "final_answer": "Comprehensive final answer addressing the original goal by synthesizing all execution results. If achieved: provide a complete, well-structured response with success confirmation. If not achieved: explain what was accomplished and what remains to be done. THIS MUST BE IN THE SAME LANGUAGE AS THE GOAL.",
  "memory_insights": {{
    "should_store": true/false,
    "reason": "Brief explanation of why this should/shouldn't be stored as memory",

    "classification": {{
      "primary_domain": "Main domain category (e.g., Data Analysis, Software Development, Machine Learning)",
      "secondary_domains": ["Related domain categories"],
      "task_type": "Specific task type",
      "complexity_level": "Simple/Medium/Complex",
      "keywords": ["relevant keywords for retrieval"]
    }},

    "execution_insights": "Analysis of successful strategies, efficiency optimization, and resource utilization",
    "failure_analysis": "Root cause analysis of failures and solutions implemented",
    "success_factors": "Key decisions and strategy effectiveness that influenced the outcome",
    "learned_patterns": "Abstract patterns that can be applied to future similar tasks",
    "improvement_suggestions": "Specific actionable suggestions for optimization and improvement",

    "user_preferences": "Analysis of user preferences inferred from task requirements and execution (e.g., preferred output formats, level of detail, communication style, tool usage patterns)",
    "behavioral_patterns": "Observable user behavior patterns (e.g., common task types, time preferences, failure tolerance, iteration patterns)"
  }}
}}

FINAL ANSWER GUIDELINES:
- If the goal was successfully achieved with meaningful results, indicate success clearly
- If the goal could not be achieved due to failures or missing information, indicate failure clearly
- Provide clear reasoning for your success/failure determination
- If failed, explain what went wrong and what could be done differently
- When there are output files, you MUST include them using the EXACT format shown above:
  - For images: `![description](file:path/to/file.png)`
  - For other files: `[filename](file:path/to/file)`
  - ALWAYS use the `file:` prefix followed by the exact relative path
- Group related files together under a clear section heading
- IMPORTANT: If you determine that the goal has NOT been achieved, you MUST start your final_answer with 'TASK FAILED: ' followed by a clear explanation (in the same language as the goal)

STRICT MEMORY STORAGE CRITERIA:

STORAGE DECISION: Only set should_store to true if this execution contains EXCEPTIONAL, NON-OBVIOUS insights with clear future value.

REQUIREMENTS (ALL must be met):
1. **UNIQUE VALUE** - Insights not easily obtainable through general knowledge or simple reasoning
2. **NON-ROUTINE** - Execution goes beyond standard patterns and approaches
3. **REUSABLE** - Learnings applicable to multiple future scenarios, not just this specific case
4. **ACTIONABLE** - Provides concrete guidance for improving future task execution

AUTOMATIC REJECTION (store = false) for:
- Routine task completion following standard workflows
- Generic descriptions of "effective" processes
- Common problem-solving approaches without unique insights
- Obvious strategies that don't provide new learnings
- General information easily available elsewhere
- Simple success/failure without deep analysis

FOCUS ON:
- **User Preferences**: Clear behavioral patterns, communication style preferences, output format requirements
- **Failure Analysis**: Non-obvious root causes and effective solutions discovered
- **Innovation**: Novel approaches that demonstrate significant improvement over standard methods
- **Efficiency**: Discoveries that substantially improve resource utilization or time complexity

STORAGE THRESHOLD: Be extremely conservative. Default to should_store = false unless there's compelling evidence of exceptional value. When in doubt, reject storage."""

        # Try to extract and reuse existing messages context from history
        working_messages = [{"role": "system", "content": system_prompt}]

        # Look for messages in the history (from step execution results)
        context_messages = []
        for iteration in history:
            if "results" in iteration:
                for result in iteration.get("results", []):
                    if "messages" in result.get("result", {}):
                        messages = result["result"]["messages"]
                        if isinstance(messages, list):
                            context_messages.extend(messages)

        # If we found contextual messages, use them as conversation history
        if context_messages:
            # Add relevant contextual messages (excluding system prompts to avoid conflicts)
            for msg in context_messages:
                if msg.get("role") != "system":  # Avoid system prompt conflicts
                    working_messages.append(msg)

        # Add the goal analysis as the final user message
        working_messages.append({"role": "user", "content": goal_analysis_prompt})

        return working_messages

    def _summarize_execution_history(self, history: List[Dict[str, Any]]) -> str:
        """Create a summary of execution history for goal checking"""
        if not history:
            return "No execution history"

        summary_parts = []

        for i, iteration_data in enumerate(history, 1):
            iteration_summary = f"Iteration {i}:"

            # Check if this iteration was triggered by continuation
            if iteration_data.get("continuation"):
                continuation = iteration_data["continuation"]
                user_input = continuation.get("user_input", "")
                iteration_summary += f" [User Modification: '{user_input}']"

            if "plan" in iteration_data:
                plan_data = iteration_data["plan"]
                iteration_summary += (
                    f" Plan with {len(plan_data.get('steps', []))} steps"
                )

            if "results" in iteration_data:
                results = iteration_data["results"]
                completed = len([r for r in results if r.get("status") == "completed"])
                failed = len([r for r in results if r.get("status") == "failed"])
                iteration_summary += (
                    f", Results: {completed} completed, {failed} failed"
                )

                # Add actual content from successful steps for goal checking
                if completed > 0:
                    iteration_summary += "\nSuccessful Step Results:\n"
                    for result in results:
                        if result.get("status") == "completed":
                            step_name = result.get("step_name", "Unknown")
                            step_result = result.get("result", {})
                            content = self._extract_content_from_result(step_result)
                            # Include full content for goal checking
                            iteration_summary += f"- {step_name}: {content}\n"

            summary_parts.append(iteration_summary)

        return "\n".join(summary_parts) if summary_parts else "No execution history"

    async def generate_final_answer(
        self,
        goal: str,
        history: List[Dict[str, Any]],
        file_outputs: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a comprehensive final answer based on execution results"""
        logger.info("Generating comprehensive final answer")

        prompt = self._build_final_answer_prompt(goal, history, file_outputs)

        try:
            # Trace LLM call
            trace_task_id = f"final_answer_{int(time.time())}"
            await trace_task_llm_call_start(
                self.tracer,
                trace_task_id,
                data={
                    "goal": goal[:200],  # Truncate for trace
                    "history_length": len(history),
                    "task_type": "final_answer_generation",
                },
            )

            # Clean messages before sending to LLM
            cleaned_prompt = clean_messages(prompt)
            # Call LLM to generate final answer
            # Use streaming API to collect complete response
            full_content = ""

            async for chunk in self.llm.stream_chat(messages=cleaned_prompt):
                if chunk.is_token():
                    full_content += chunk.delta
                elif chunk.is_error():
                    raise RuntimeError(f"LLM stream error: {chunk.content}")

            content = full_content

            # Trace LLM call completion
            await trace_task_llm_call_end(
                self.tracer,
                trace_task_id,
                data={
                    "goal": goal[:200],  # Truncate for trace
                    "history_length": len(history),
                    "task_type": "final_answer_generation",
                    "response_length": len(content) if content else 0,
                },
            )

            return content.strip()

        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            # Fallback to simple summary
            return self._generate_fallback_summary(history)

    def _build_final_answer_prompt(
        self,
        goal: str,
        history: List[Dict[str, Any]],
        file_outputs: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build prompt for generating comprehensive final answer"""

        execution_summary = self._summarize_execution_results(history)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build file outputs section for the prompt
        file_outputs_section = ""
        if file_outputs:
            file_outputs_section = "\n\nOutput Files:\n"
            for i, file_info in enumerate(file_outputs, 1):
                filename = file_info.get("filename", "")
                relative_path = file_info.get("relative_path", "")
                file_outputs_section += f"  {i}. {filename} (path: {relative_path})\n"
            file_outputs_section += "\n"

        system_prompt = (
            "You are an AI assistant that synthesizes execution results into a comprehensive answer.\n"
            "Based on the execution history and results from multiple steps, generate a complete, "
            "well-structured response that directly addresses the original goal.\n"
            "Integrate information from all successful steps to provide a thorough answer.\n\n"
            "CRITICAL LANGUAGE REQUIREMENT: YOU MUST RESPOND IN THE SAME LANGUAGE AS THE ORIGINAL GOAL!\n"
            "If the goal is in English, respond ONLY in English.\n"
            "If the goal is in Chinese, respond ONLY in Chinese.\n"
            "If the goal is in Japanese, respond ONLY in Japanese.\n"
            "NEVER mix languages. Your entire response must be in the same language as the goal.\n\n"
            "Important: When evaluating the results:\n"
            "1. If the goal was successfully achieved with meaningful results, indicate success\n"
            "2. If the goal could not be achieved due to failures or missing information, indicate failure\n"
            "3. Provide clear reasoning for your success/failure determination\n"
            "4. If failed, explain what went wrong and what could be done differently\n\n"
            "CRITICAL: If you determine that the goal has NOT been achieved, you MUST start your "
            "response with 'TASK FAILED: ' followed by a clear explanation of the failure (in the same language as the goal).\n\n"
            "FILE OUTPUT FORMATTING:\n"
            "When there are output files, you MUST include them in your response using Markdown link format:\n"
            "- Use this EXACT format for each file: `[filename](file:path/to/file)`\n"
            "- Example: `[report.html](file:web_task_123/output/report.html)`\n"
            "- Group related files together under a clear section heading\n"
            "- Always include file links at the end of your response in a clear, organized manner\n"
            "- This allows users to click on the file links to preview them in the right panel"
        )

        # Detect goal language for explicit instruction
        goal_has_chinese = any("\u4e00" <= c <= "\u9fff" for c in goal)
        goal_has_english = any(c.isascii() and c.isalpha() for c in goal)
        detected_language = (
            "Chinese"
            if goal_has_chinese
            else ("English" if goal_has_english else "the same")
        )

        user_prompt = (
            f"Original Goal: {goal}\n\n"
            f"Current Time: {current_time}\n\n"
            f"Execution Results:\n{execution_summary}\n"
            f"{file_outputs_section}"
            f"IMPORTANT: Your response MUST be in {detected_language} ONLY, matching the goal's language.\n\n"
            f"Please generate a comprehensive final answer that addresses the original goal "
            f"by synthesizing all the information gathered during execution."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _summarize_execution_results(self, history: List[Dict[str, Any]]) -> str:
        """Create a detailed summary of execution results with actual content"""
        if not history:
            return "No execution history available"

        summary_parts = []

        for i, iteration_data in enumerate(history, 1):
            iteration_summary = f"Iteration {i}:\n"

            # Check if this iteration was triggered by continuation
            if iteration_data.get("continuation"):
                continuation = iteration_data["continuation"]
                user_input = continuation.get("user_input", "")
                iteration_summary += f"[User Modified Goal: '{user_input}']\n"

            if "results" in iteration_data:
                results = iteration_data["results"]
                successful_results = [
                    r for r in results if r.get("status") == "completed"
                ]

                if successful_results:
                    iteration_summary += "Successful Steps:\n"
                    for result in successful_results:
                        step_name = result.get("step_name", "Unknown")
                        step_result = result.get("result", {})
                        content = self._extract_content_from_result(step_result)
                        iteration_summary += f"- {step_name}: {content[:200]}{'...' if len(content) > 200 else ''}\n"

                failed_results = [r for r in results if r.get("status") == "failed"]
                if failed_results:
                    iteration_summary += "Failed Steps:\n"
                    for result in failed_results:
                        step_name = result.get("step_name", "Unknown")
                        error = result.get("error", "Unknown error")
                        iteration_summary += f"- {step_name}: {error}\n"

            summary_parts.append(iteration_summary)

        return "\n".join(summary_parts)

    def _extract_content_from_result(self, result: Dict[str, Any]) -> str:
        """Extract meaningful content from a step result"""
        if not isinstance(result, dict):
            return str(result)

        # Try common content keys in priority order
        content_keys = [
            "output",
            "analysis_result",
            "direct_analysis",
            "content",
            "text",
            "answer",
            "response",
        ]

        for key in content_keys:
            if key in result and result[key]:
                content = result[key]
                if isinstance(content, dict):
                    # If it's a dict, try nested keys
                    for nested_key in ["output", "content", "text"]:
                        if nested_key in content:
                            return str(content[nested_key])
                return str(content)

        # If no specific content found, return string representation
        return str(result)

    def _generate_fallback_summary(self, history: List[Dict[str, Any]]) -> str:
        """Generate a simple fallback summary when LLM generation fails"""
        if not history:
            return "No execution results available"

        last_iteration = history[-1]
        results = last_iteration.get("results", [])
        successful = len([r for r in results if r.get("status") == "completed"])
        failed = len([r for r in results if r.get("status") == "failed"])

        if failed == 0:
            return f"Task completed successfully with {successful} steps"
        elif successful > 0:
            return (
                f"Partial success: {successful} steps completed, {failed} steps failed"
            )
        else:
            return f"Task failed with {failed} failed steps"

    async def _call_llm_with_retry(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Any:
        """Call LLM with retry mechanism: JSON mode first, then fallback to normal mode.

        This is similar to plan_generator._call_llm_with_retry and uses stream_chat().
        """
        try:
            # Clean messages before sending to LLM
            cleaned_messages = clean_messages(messages)

            # Use streaming API to collect complete response
            full_content = ""
            usage = {}

            # Prefer output_config over response_format for structured output
            if "output_config" in kwargs:
                # Use output_config (better for Gemini with complex schemas)
                output_config = kwargs.pop("output_config")
                async for chunk in self.llm.stream_chat(
                    messages=cleaned_messages,
                    output_config=output_config,
                    **kwargs,
                ):
                    if chunk.is_token():
                        full_content += chunk.delta
                    elif chunk.is_usage():
                        usage = chunk.usage
                    elif chunk.is_error():
                        raise RuntimeError(f"LLM stream error: {chunk.content}")
            else:
                # Fallback to response_format
                response_format = kwargs.pop("response_format", {"type": "json_object"})
                async for chunk in self.llm.stream_chat(
                    messages=cleaned_messages,
                    response_format=response_format,
                    **kwargs,
                ):
                    if chunk.is_token():
                        full_content += chunk.delta
                    elif chunk.is_usage():
                        usage = chunk.usage
                    elif chunk.is_error():
                        raise RuntimeError(f"LLM stream error: {chunk.content}")

            # Record token usage
            if usage:
                logger.info(
                    f"Goal check LLM usage - prompt_tokens: {usage.get('prompt_tokens', 0)}, "
                    f"completion_tokens: {usage.get('completion_tokens', 0)}, "
                    f"total_tokens: {usage.get('total_tokens', 0)}"
                )

                # Add token usage to tracker
                add_token_usage(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    model=self.llm.model_name,
                    call_type="goal_check",
                )

            # Return format (compatible with original chat())
            # Ensure we always return a dict with content key
            if not isinstance(full_content, str):
                logger.error(
                    f"Unexpected full_content type: {type(full_content)}, value: {full_content!r}"
                )
                full_content = str(full_content)
            return {"content": full_content, "usage": usage}

        except Exception as e:
            logger.warning(f"JSON mode call failed, retrying with normal mode: {e}")

            # Fallback: Use streaming API to retry (without JSON mode)
            try:
                full_content = ""
                usage = {}

                # Preserve output_config if provided (for retry)
                output_config = kwargs.get("output_config")
                if output_config:
                    # Try with output_config first
                    async for chunk in self.llm.stream_chat(
                        messages=cleaned_messages,
                        output_config=output_config,
                        **kwargs,
                    ):
                        if chunk.is_token():
                            full_content += chunk.delta
                        elif chunk.is_usage():
                            usage = chunk.usage
                        elif chunk.is_error():
                            raise RuntimeError(f"LLM stream error: {chunk.content}")
                else:
                    # Fallback to no format constraints
                    async for chunk in self.llm.stream_chat(
                        messages=cleaned_messages,
                        **kwargs,
                    ):
                        if chunk.is_token():
                            full_content += chunk.delta
                        elif chunk.is_usage():
                            usage = chunk.usage
                        elif chunk.is_error():
                            raise RuntimeError(f"LLM stream error: {chunk.content}")

                logger.info(
                    f"Normal mode call succeeded (fallback), response length: {len(full_content)}"
                )

                # Record token usage for fallback
                if usage:
                    add_token_usage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        model=self.llm.model_name,
                        call_type="goal_check_fallback",
                    )

                # Ensure we always return a dict with content key
                if not isinstance(full_content, str):
                    logger.error(
                        f"Unexpected full_content type in fallback: {type(full_content)}, value: {full_content!r}"
                    )
                    full_content = str(full_content)
                return {"content": full_content, "usage": usage}

            except Exception as e2:
                logger.error(f"Normal mode call also failed: {e2}")
                raise
