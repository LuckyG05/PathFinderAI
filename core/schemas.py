from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class TopicItem(BaseModel):
    """Represents a single topic inside a learning roadmap."""

    order_index: int = Field(ge=1, description="1-based step number in the roadmap")
    title: str = Field(min_length=1, description="Short, clear title of the topic")
    description: str = Field(
        min_length=1, description="1-2 sentence overview of what the student will learn"
    )


class RoadmapResponse(BaseModel):
    """Structured output returned by the LLM when generating a roadmap."""

    subject: str = Field(min_length=1, description="Normalized name of the subject")
    topics: List[TopicItem] = Field(
        description="Ordered list of 5 to 7 topics from beginner to advanced",
        min_length=3,
        max_length=8,
    )


class MCQQuestion(BaseModel):
    """Structured output for a single Multiple Choice Question."""

    question: str = Field(
        min_length=1, description="Clear multiple-choice question testing conceptual understanding"
    )
    option_a: str = Field(min_length=1, description="Text for Option A (without the 'A)' prefix)")
    option_b: str = Field(min_length=1, description="Text for Option B (without the 'B)' prefix)")
    option_c: str = Field(min_length=1, description="Text for Option C (without the 'C)' prefix)")
    option_d: str = Field(min_length=1, description="Text for Option D (without the 'D)' prefix)")
    correct_option: Literal["A", "B", "C", "D"] = Field(
        description="The exact single letter ('A', 'B', 'C', or 'D') of the correct option"
    )
    explanation: str = Field(
        min_length=1, description="Concise explanation of why the correct option is right"
    )

    @field_validator("correct_option", mode="before")
    @classmethod
    def _normalize_correct_option(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value


class QuizBatchResponse(BaseModel):
    """A batch of unique MCQ questions generated in a single API call."""

    questions: List[MCQQuestion] = Field(
        description="List of unique multiple-choice questions",
        min_length=1,
    )


def grade_mcq_answer(user_answer: object, correct_option: object) -> bool:
    """
    Safely grades a user's MCQ answer against the correct option ('A', 'B', 'C', or 'D').
    Prevents empty-string or substring bugs (like '' in 'A').
    """
    if not isinstance(user_answer, str) or not isinstance(correct_option, str):
        return False

    cleaned_user = user_answer.strip().upper()
    cleaned_correct = correct_option.strip().upper()

    if cleaned_user not in {"A", "B", "C", "D"} or cleaned_correct not in {"A", "B", "C", "D"}:
        return False

    return cleaned_user == cleaned_correct
