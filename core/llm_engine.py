import json
import logging
import os
from typing import Dict, List, Optional, Type, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel

from core.rag_pipeline import retrieve_relevant_context
from core.schemas import MCQQuestion, QuizBatchResponse, RoadmapResponse

load_dotenv(override=True)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

DEFAULT_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

DEFAULT_GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
]


def _build_deduplicated_models(primary_override: Optional[str], defaults: List[str]) -> List[str]:
    """Builds an ordered, deduplicated model list with optional env override first."""
    ordered: List[str] = []
    if primary_override and primary_override.strip():
        ordered.append(primary_override.strip())
    for model in defaults:
        if model not in ordered:
            ordered.append(model)
    return ordered


def _build_gemini_client(api_key: str) -> genai.Client:
    """Creates a Gemini client with fast failover (no 30s internal retry stalls on 503)."""
    retry_cls = getattr(types, "HttpRetryOptions", None)
    if retry_cls is not None:
        http_options = types.HttpOptions(
            timeout=20_000,
            retry_options=retry_cls(attempts=1),
        )
        return genai.Client(api_key=api_key, http_options=http_options)
    return genai.Client(api_key=api_key)


class LLMEngine:
    """
    100% Free High-Availability Multi-Model / Multi-Provider AI Engine:
    1. Primary: Groq Cloud Free Tier (openai/gpt-oss-120b -> qwen/qwen3.8-27b -> openai/gpt-oss-20b)
       for fast (~1s), reliable generation.
    2. Automatic Failover: Google Gemini Free Tier
       (gemini-3.5-flash -> gemini-3.5-flash-lite -> gemini-3.6-flash)
       if Groq hits rate/token limits or is unavailable.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        # Reload .env with override=True so local .env always wins over stale OS env vars
        load_dotenv(override=True)

        self.groq_key = (
            groq_api_key if groq_api_key is not None else os.getenv("GROQ_API_KEY", "")
        ).strip() or None
        self.gemini_key = (
            gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")
        ).strip() or None

        if not self.groq_key and not self.gemini_key:
            raise ValueError(
                "No API keys found. Set GROQ_API_KEY (primary) and/or GEMINI_API_KEY (backup) "
                "in your .env file."
            )

        self.groq_client = (
            OpenAI(
                api_key=self.groq_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=25.0,
                max_retries=1,
            )
            if self.groq_key
            else None
        )
        self.gemini_client = _build_gemini_client(self.gemini_key) if self.gemini_key else None

        self.groq_models = _build_deduplicated_models(
            os.getenv("GROQ_MODEL"), DEFAULT_GROQ_MODELS
        )
        self.gemini_models = _build_deduplicated_models(
            os.getenv("GEMINI_MODEL"), DEFAULT_GEMINI_MODELS
        )
        self.last_provider_used: Optional[str] = None

    def _generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generates text with automatic multi-model & multi-provider failover (Groq -> Gemini).
        Supports optional multi-turn `messages` (list of {'role': 'user'|'assistant', 'content': str}).
        """
        errors: List[str] = []
        turn_messages = (
            messages if messages is not None else [{"role": "user", "content": user_prompt}]
        )

        # 1. Primary: Groq Cloud (fast free tier, multi-model failover)
        if self.groq_client:
            for model in self.groq_models:
                try:
                    resp = self.groq_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system_prompt}, *turn_messages],
                        temperature=temperature,
                    )
                    content = resp.choices[0].message.content
                    if content and content.strip():
                        self.last_provider_used = f"groq/{model}"
                        return content.strip()
                    errors.append(f"groq/{model}: empty response")
                except Exception as exc:
                    short_err = str(exc).splitlines()[0][:180]
                    logger.warning("Groq text call failed on %s: %s", model, short_err)
                    errors.append(f"groq/{model}: {short_err}")

        # 2. Automatic Failover: Google Gemini Free Tier
        if self.gemini_client:
            gemini_contents = [
                types.Content(
                    role="model" if m["role"] == "assistant" else "user",
                    parts=[types.Part.from_text(text=m["content"])],
                )
                for m in turn_messages
            ]
            for model in self.gemini_models:
                try:
                    resp = self.gemini_client.models.generate_content(
                        model=model,
                        contents=gemini_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                        ),
                    )
                    if resp.text and resp.text.strip():
                        self.last_provider_used = f"gemini/{model}"
                        return resp.text.strip()
                    errors.append(f"gemini/{model}: empty response")
                except Exception as exc:
                    short_err = str(exc).splitlines()[0][:180]
                    logger.warning("Gemini text call failed on %s: %s", model, short_err)
                    errors.append(f"gemini/{model}: {short_err}")

        raise RuntimeError(
            "All configured LLM providers failed. Details:\n- " + "\n- ".join(errors)
        )

    def _generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        temperature: float = 0.2,
    ) -> T:
        """Generates schema-validated Pydantic output with automatic Groq -> Gemini failover."""
        errors: List[str] = []

        # 1. Primary: Groq Cloud (JSON Object mode + strict Pydantic validation)
        if self.groq_client:
            schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
            structured_system = (
                f"{system_prompt}\n\n"
                "You MUST respond with ONLY a valid JSON object that strictly conforms to this "
                f"JSON Schema (no markdown code fences, no commentary):\n{schema_json}"
            )
            for model in self.groq_models:
                try:
                    resp = self.groq_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": structured_system},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=temperature,
                    )
                    raw_json = (resp.choices[0].message.content or "").strip()
                    if not raw_json:
                        errors.append(f"groq/{model}: empty JSON response")
                        continue
                    parsed = response_schema.model_validate_json(raw_json)
                    self.last_provider_used = f"groq/{model}"
                    return parsed
                except Exception as exc:
                    short_err = str(exc).splitlines()[0][:180]
                    logger.warning("Groq structured call failed on %s: %s", model, short_err)
                    errors.append(f"groq/{model}: {short_err}")

        # 2. Automatic Failover: Google Gemini Native Structured Output
        if self.gemini_client:
            for model in self.gemini_models:
                try:
                    response = self.gemini_client.models.generate_content(
                        model=model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=response_schema,
                            temperature=temperature,
                        ),
                    )
                    if not response.text:
                        errors.append(f"gemini/{model}: empty JSON response")
                        continue
                    parsed = response_schema.model_validate_json(response.text)
                    self.last_provider_used = f"gemini/{model}"
                    return parsed
                except Exception as exc:
                    short_err = str(exc).splitlines()[0][:180]
                    logger.warning("Gemini structured call failed on %s: %s", model, short_err)
                    errors.append(f"gemini/{model}: {short_err}")

        raise RuntimeError(
            "All configured LLM providers failed for structured generation. Details:\n- "
            + "\n- ".join(errors)
        )

    def generate_roadmap(self, subject: str, source_context: str = "") -> RoadmapResponse:
        """Generates a structured 5-7 step learning roadmap, optionally grounded in an uploaded PDF."""
        cleaned_subject = subject.strip()
        if not cleaned_subject:
            raise ValueError("subject must not be empty")

        system_prompt = (
            "You are an expert curriculum designer. Create a clear, logical, step-by-step "
            "learning roadmap ordered from foundational concepts to advanced topics."
        )
        rag_snippet = retrieve_relevant_context(source_context, cleaned_subject, top_k=4)
        context_block = (
            f"\n\nGround the roadmap strictly in this uploaded syllabus/reference document:\n{rag_snippet}"
            if rag_snippet
            else ""
        )
        user_prompt = (
            f"Create a 5 to 7 topic learning roadmap for mastering: '{cleaned_subject}'. "
            f"Number the topics sequentially starting at 1.{context_block}"
        )
        return self._generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=RoadmapResponse,
            temperature=0.2,
        )

    def generate_topic_notes(
        self, subject: str, topic_title: str, topic_description: str, source_context: str = ""
    ) -> str:
        """Generates comprehensive Markdown study notes for a topic, grounded in RAG context if available."""
        rag_snippet = retrieve_relevant_context(
            source_context, f"{subject} {topic_title} {topic_description}", top_k=3
        )
        context_block = (
            "\n\nUse the following uploaded reference material as your primary source "
            f"(cite [Page X] when applicable):\n{rag_snippet}"
            if rag_snippet
            else ""
        )
        system_prompt = (
            "You are an engaging, clear technical tutor. Write structured Markdown study notes "
            "with: 1) Core Intuition, 2) Key Concepts & Definitions, 3) Practical Example/Code, "
            "and 4) Quick Summary Takeaways."
        )
        user_prompt = (
            f"Subject: {subject}\n"
            f"Topic: {topic_title}\n"
            f"Overview: {topic_description}"
            f"{context_block}"
        )
        return self._generate_text(system_prompt, user_prompt, temperature=0.4)

    def answer_doubt(
        self,
        subject: str,
        topic_title: str,
        topic_notes: str,
        chat_history: List[Dict[str, str]],
        user_question: str,
    ) -> str:
        """Stateful multi-turn tutor chat using the last 8 turns and topic study notes."""
        cleaned_question = user_question.strip()
        if not cleaned_question:
            raise ValueError("user_question must not be empty")

        system_prompt = (
            f"You are a helpful personal tutor teaching '{topic_title}' in the subject '{subject}'. "
            f"Answer the student's follow-up questions clearly and concisely based on these study notes:\n\n"
            f"{topic_notes[:3500]}"
        )
        recent_turns = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in chat_history[-8:]
            if msg.get("role") in {"user", "assistant"} and msg.get("content")
        ]
        recent_turns.append({"role": "user", "content": cleaned_question})

        return self._generate_text(
            system_prompt=system_prompt,
            user_prompt=cleaned_question,
            temperature=0.4,
            messages=recent_turns,
        )

    def generate_quiz(
        self,
        subject: str,
        topic_titles: List[str],
        num_questions: int = 3,
        source_context: str = "",
    ) -> List[MCQQuestion]:
        """Generates a batch of unique MCQ questions in a single structured API call."""
        cleaned_topics = [t.strip() for t in topic_titles if t and t.strip()]
        if not cleaned_topics:
            raise ValueError("topic_titles must contain at least one topic")

        count = max(1, int(num_questions))
        topics_str = ", ".join(cleaned_topics)
        rag_snippet = retrieve_relevant_context(source_context, f"{subject} {topics_str}", top_k=3)
        context_block = (
            f"\n\nBase the questions on this reference material:\n{rag_snippet}"
            if rag_snippet
            else ""
        )
        system_prompt = (
            "You are a strict academic examiner. Generate distinct, conceptual multiple-choice "
            "questions with 4 plausible options (A, B, C, D) and a clear explanation."
        )
        user_prompt = (
            f"Subject: {subject}\n"
            f"Topics to test: {topics_str}\n"
            f"Generate {count} unique multiple-choice questions.{context_block}"
        )
        batch = self._generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=QuizBatchResponse,
            temperature=0.2,
        )
        return batch.questions[:count]
