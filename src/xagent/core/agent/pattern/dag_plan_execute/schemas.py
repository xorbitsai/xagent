"""Pydantic schemas for DAG plan-execute pattern."""

from typing import List, Optional

from pydantic import BaseModel, Field


class MemoryClassification(BaseModel):
    """Memory classification schema."""

    primary_domain: str = Field(
        description="Main domain category (e.g., Data Analysis, Software Development)"
    )
    secondary_domains: List[str] = Field(
        default_factory=list, description="Related domain categories"
    )
    task_type: str = Field(description="Specific task type")
    complexity_level: str = Field(description="Simple/Medium/Complex")
    keywords: List[str] = Field(
        default_factory=list, description="Keywords for retrieval"
    )


class MemoryInsights(BaseModel):
    """Memory insights schema."""

    should_store: bool = Field(description="Whether to store as memory")
    reason: str = Field(description="Explanation of storage decision")
    classification: MemoryClassification = Field(description="Task classification")

    execution_insights: Optional[str] = Field(
        default="", description="Analysis of execution"
    )
    failure_analysis: Optional[str] = Field(
        default="", description="Root cause analysis"
    )
    success_factors: Optional[str] = Field(
        default="", description="Key success factors"
    )
    learned_patterns: Optional[str] = Field(default="", description="Learned patterns")
    improvement_suggestions: Optional[str] = Field(
        default="", description="Improvement suggestions"
    )
    user_preferences: Optional[str] = Field(default="", description="User preferences")
    behavioral_patterns: Optional[str] = Field(
        default="", description="Behavioral patterns"
    )


class GoalCheckResponse(BaseModel):
    """Goal achievement check response schema."""

    achieved: bool = Field(description="Whether the goal was achieved")
    reason: str = Field(description="Explanation of goal achievement status")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level (0.0-1.0)")
    final_answer: str = Field(
        description="Comprehensive final answer addressing the original goal"
    )
    memory_insights: MemoryInsights = Field(description="Memory storage insights")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "achieved": True,
                    "reason": "The goal was successfully completed with all requirements met",
                    "confidence": 0.95,
                    "final_answer": "The task is complete with the following results...",
                    "memory_insights": {
                        "should_store": False,
                        "reason": "Routine task execution without unique insights",
                        "classification": {
                            "primary_domain": "Task Execution",
                            "secondary_domains": [],
                            "task_type": "Simple",
                            "complexity_level": "Simple",
                            "keywords": ["execution", "completion"],
                        },
                    },
                }
            ]
        }
    }
