import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from core.database import DatabaseManager
from core.llm_engine import LLMEngine, _build_deduplicated_models
from core.rag_pipeline import chunk_text, retrieve_relevant_context
from core.schemas import (
    MCQQuestion,
    QuizBatchResponse,
    RoadmapResponse,
    TopicItem,
    grade_mcq_answer,
)


# ---------------------------------------------------------------------------
# 1. Schema Validation & Deterministic MCQ Grading Tests
# ---------------------------------------------------------------------------
class TestSchemasAndGrading:
    @pytest.mark.parametrize(
        "user_ans,correct_opt,expected",
        [
            ("A", "A", True),
            ("b", "B", True),
            ("  c  ", "C", True),
            ("", "A", False),  # Prevents classic '' in 'A' Python bug
            ("   ", "A", False),
            ("Z", "A", False),
            ("AB", "A", False),
            (None, "A", False),
            ("A", None, False),
            ("B", "A", False),
        ],
    )
    def test_grade_mcq_answer(self, user_ans, correct_opt, expected):
        assert grade_mcq_answer(user_ans, correct_opt) is expected

    def test_mcq_question_normalizes_lowercase_option(self):
        q = MCQQuestion(
            question="What is a primary key?",
            option_a="Unique row identifier",
            option_b="Duplicate key",
            option_c="Temporary index",
            option_d="Foreign reference",
            correct_option=" a ",
            explanation="A primary key uniquely identifies each record.",
        )
        assert q.correct_option == "A"

    def test_mcq_question_rejects_invalid_option(self):
        with pytest.raises(ValidationError):
            MCQQuestion(
                question="Invalid?",
                option_a="1",
                option_b="2",
                option_c="3",
                option_d="4",
                correct_option="E",
                explanation="Should fail.",
            )

    def test_roadmap_response_enforces_topic_bounds(self):
        with pytest.raises(ValidationError):
            RoadmapResponse(
                subject="SQL",
                topics=[TopicItem(order_index=1, title="Only One", description="Too few topics")],
            )


# ---------------------------------------------------------------------------
# 2. SQLite Database & Relational Integrity Tests
# ---------------------------------------------------------------------------
def _sample_roadmap(subject: str = "Operating Systems") -> RoadmapResponse:
    return RoadmapResponse(
        subject=subject,
        topics=[
            TopicItem(order_index=1, title="Processes & Threads", description="Concurrency basics"),
            TopicItem(order_index=2, title="CPU Scheduling", description="FCFS, SJF, Round Robin"),
            TopicItem(order_index=3, title="Deadlocks", description="Prevention and Banker's algorithm"),
        ],
    )


class TestDatabaseManager:
    def test_create_list_and_progress_aggregation(self, tmp_path):
        db = DatabaseManager(str(tmp_path / "test.db"))
        rm_id = db.create_roadmap(_sample_roadmap(), source_context="[Page 1] OS Syllabus")

        roadmaps = db.list_roadmaps()
        assert len(roadmaps) == 1
        assert roadmaps[0]["subject"] == "Operating Systems"
        assert roadmaps[0]["total_topics"] == 3
        assert roadmaps[0]["completed_topics"] == 0

        topics = db.get_topics_for_roadmap(rm_id)
        assert [t["order_index"] for t in topics] == [1, 2, 3]

        # Caching notes transitions status from not_started -> in_progress
        db.save_topic_notes(topics[0]["id"], "# Notes on Processes")
        updated = db.get_topic(topics[0]["id"])
        assert updated["notes"] == "# Notes on Processes"
        assert updated["status"] == "in_progress"

        # Completing a topic updates aggregate count, and re-saving notes keeps completed status
        db.update_topic_status(topics[0]["id"], "completed")
        db.save_topic_notes(topics[0]["id"], "# Updated Notes")
        assert db.get_topic(topics[0]["id"])["status"] == "completed"
        assert db.list_roadmaps()[0]["completed_topics"] == 1

    def test_check_constraint_rejects_invalid_status(self, tmp_path):
        db = DatabaseManager(str(tmp_path / "test.db"))
        rm_id = db.create_roadmap(_sample_roadmap())
        topic_id = db.get_topics_for_roadmap(rm_id)[0]["id"]

        with pytest.raises(sqlite3.IntegrityError):
            db.update_topic_status(topic_id, "finished_maybe")

    def test_quiz_history_and_cascade_delete(self, tmp_path):
        db = DatabaseManager(str(tmp_path / "test.db"))
        rm_id = db.create_roadmap(_sample_roadmap())
        topic_id = db.get_topics_for_roadmap(rm_id)[0]["id"]

        db.record_quiz_attempt(rm_id, "topic_quiz", score=2, total_questions=3, topic_id=topic_id)
        db.record_quiz_attempt(rm_id, "final_test", score=5, total_questions=6, topic_id=None)

        history = db.get_quiz_history(rm_id)
        assert len(history) == 2
        assert history[0]["topic_title"] == "Cumulative Final Test"
        assert history[1]["topic_title"] == "Processes & Threads"

        # Deleting roadmap must cascade to topics and quiz_attempts
        assert db.delete_roadmap(rm_id) is True
        assert db.get_roadmap(rm_id) is None
        assert db.get_topics_for_roadmap(rm_id) == []
        assert db.get_quiz_history(rm_id) == []


# ---------------------------------------------------------------------------
# 3. RAG Chunking & Keyword-Overlap Retrieval Tests
# ---------------------------------------------------------------------------
class TestRAGPipeline:
    def test_chunk_text_sliding_window_overlap(self):
        text = "A" * 100 + "B" * 100 + "C" * 50  # 250 chars -> [0:100], [80:180], [160:250]
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) == 3
        # Verify overlap continuity between consecutive chunks
        assert chunks[0][-20:] == chunks[1][:20]
        assert chunks[1][-20:] == chunks[2][:20]

    def test_chunk_text_invalid_params(self):
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=100, overlap=100)
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=0, overlap=0)
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=100, overlap=-5)

    def test_retrieve_relevant_context_ranks_matching_chunk_first(self):
        doc = (
            "[Page 1] Gardening tips for tomatoes and soil pH balance.\n"
            + ("filler " * 60)
            + "\n[Page 2] Deadlocks occur in operating systems when circular wait and mutual exclusion hold.\n"
            + ("padding " * 60)
            + "\n[Page 3] History of classical music and symphonies."
        )
        result = retrieve_relevant_context(doc, query="deadlocks circular wait", top_k=1, chunk_size=300, overlap=30)
        assert "Deadlocks" in result
        assert "circular wait" in result

    def test_retrieve_relevant_context_handles_empty_input(self):
        assert retrieve_relevant_context("", "SQL") == ""
        assert retrieve_relevant_context("Some text", "SQL", top_k=0) == ""


# ---------------------------------------------------------------------------
# 4. LLM Engine Multi-Model & Multi-Provider Failover Tests (Mocked / Offline)
# ---------------------------------------------------------------------------
class TestLLMEngineFailover:
    def test_deduplicated_model_order(self):
        models = _build_deduplicated_models("qwen/qwen3.8-27b", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
        assert models == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]

    def test_raises_value_error_when_no_keys_configured(self, monkeypatch):
        monkeypatch.setattr("core.llm_engine.load_dotenv", lambda **kwargs: None)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="No API keys found"):
            LLMEngine(groq_api_key="", gemini_api_key="")

    def test_groq_primary_multi_model_failover_on_429(self, monkeypatch):
        monkeypatch.setattr("core.llm_engine.load_dotenv", lambda **kwargs: None)
        engine = LLMEngine(groq_api_key="fake-groq", gemini_api_key="")
        engine.groq_models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = [
            RuntimeError("429 rate_limit_exceeded"),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="# Study Notes from Qwen"))]
            ),
        ]
        engine.groq_client = mock_groq

        result = engine.generate_topic_notes("SQL", "Joins", "Inner and outer joins")
        assert result == "# Study Notes from Qwen"
        assert engine.last_provider_used == "groq/qwen/qwen3.8-27b"
        assert mock_groq.chat.completions.create.call_count == 2

    def test_failover_from_groq_to_gemini_structured_output(self, monkeypatch):
        monkeypatch.setattr("core.llm_engine.load_dotenv", lambda **kwargs: None)
        engine = LLMEngine(groq_api_key="fake-groq", gemini_api_key="fake-gemini")
        engine.groq_models = ["openai/gpt-oss-120b"]
        engine.gemini_models = ["gemini-3.5-flash"]

        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = RuntimeError("429 daily token limit")
        engine.groq_client = mock_groq

        expected_roadmap = _sample_roadmap("Python")
        mock_gemini = MagicMock()
        mock_gemini.models.generate_content.return_value = SimpleNamespace(
            text=expected_roadmap.model_dump_json()
        )
        engine.gemini_client = mock_gemini

        result = engine.generate_roadmap("Python")
        assert result.subject == "Python"
        assert len(result.topics) == 3
        assert engine.last_provider_used == "gemini/gemini-3.5-flash"

    def test_doubt_tutor_slices_last_8_history_messages(self, monkeypatch):
        monkeypatch.setattr("core.llm_engine.load_dotenv", lambda **kwargs: None)
        engine = LLMEngine(groq_api_key="fake-groq", gemini_api_key="")
        engine.groq_models = ["openai/gpt-oss-120b"]

        captured = {}

        def fake_create(model, messages, temperature):
            captured["messages"] = messages
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Clear explanation."))]
            )

        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = fake_create
        engine.groq_client = mock_groq

        long_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"} for i in range(12)]
        reply = engine.answer_doubt("SQL", "Indexes", "B-Tree notes", long_history, "Why B-Trees?")

        assert reply == "Clear explanation."
        # 1 system prompt + 8 historical turns + 1 new user question = 10 messages total
        assert len(captured["messages"]) == 10
        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][-1]["content"] == "Why B-Trees?"

    def test_generate_quiz_returns_requested_count(self, monkeypatch):
        monkeypatch.setattr("core.llm_engine.load_dotenv", lambda **kwargs: None)
        engine = LLMEngine(groq_api_key="fake-groq", gemini_api_key="")
        engine.groq_models = ["openai/gpt-oss-120b"]

        batch = QuizBatchResponse(
            questions=[
                MCQQuestion(
                    question=f"Q{i}",
                    option_a="A",
                    option_b="B",
                    option_c="C",
                    option_d="D",
                    correct_option="A",
                    explanation="Because A.",
                )
                for i in range(5)
            ]
        )
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=batch.model_dump_json()))]
        )
        engine.groq_client = mock_groq

        questions = engine.generate_quiz("SQL", ["Joins"], num_questions=3)
        assert len(questions) == 3
        assert engine.last_provider_used == "groq/openai/gpt-oss-120b"
