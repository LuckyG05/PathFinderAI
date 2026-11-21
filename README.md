# PathFinder AI

A structured AI Study Companion built with **Python**, **Streamlit**, **Pydantic**, **SQLite**, and a lightweight **PDF RAG Pipeline**. Enter any subject (optionally uploading a syllabus or lecture notes PDF) to generate a step-by-step learning roadmap, study cached AI notes, ask follow-up questions in a stateful Doubt Tutor, pass mastery mini-quizzes, and track progress persistently.

## Architecture

```text
PathFinderAI/
├── app.py                     # Streamlit interactive UI (Roadmap, Notes, Doubt Tutor, Quizzes, History)
├── core/
│   ├── schemas.py             # Pydantic v2 structured output models & deterministic MCQ grader
│   ├── database.py            # Normalized SQLite schema (FKs, ON DELETE CASCADE, CHECK constraints, SQL aggregations)
│   ├── rag_pipeline.py        # pypdf extraction ([Page N] tags), sliding-window chunking, keyword-overlap retrieval
│   └── llm_engine.py          # 100% free high-availability Groq (Primary) -> Gemini (Failover) router
├── tests/
│   └── test_pathfinder.py     # Offline pytest unit test suite (schemas, SQLite integrity, RAG, mocked failover)
├── .env.example               # Environment variable template
├── pytest.ini
├── requirements.txt
└── README.md
```

## Key Engineering Highlights

- **100% Free High-Availability LLM Failover (`core/llm_engine.py`):** Routes primary traffic to **Groq Cloud** (`openai/gpt-oss-120b` → `qwen/qwen3.8-27b` → `openai/gpt-oss-20b`) for sub-second responses, with automatic failover to **Google Gemini Free Tier** (`gemini-3.5-flash` → `gemini-3.5-flash-lite` → `gemini-3.6-flash`) if Groq hits rate/token limits.
- **Strict Structured Outputs & Deterministic Grading (`core/schemas.py`):** Uses Pydantic v2 (`RoadmapResponse`, `MCQQuestion`, `QuizBatchResponse`) to eliminate brittle string parsing, generate full quiz batches in a single API call, and prevent edge-case grading bugs (such as empty input `"" in "A"` evaluating to `True`).
- **Relational SQLite Persistence & Caching (`core/database.py`):** Normalized 3-table schema (`roadmaps`, `topics`, `quiz_attempts`) with parameterized SQL queries, `PRAGMA foreign_keys = ON`, `ON DELETE CASCADE`, `CHECK` constraints, `LEFT JOIN + GROUP BY` progress aggregations, and SQLite note caching (`topics.notes`) to avoid redundant LLM calls.
- **PDF RAG Grounding (`core/rag_pipeline.py`):** Extracts PDF text with `[Page N]` citations, splits documents using sliding-window overlapping chunks (`chunk_size=1200, overlap=200`), and retrieves top-`k` relevant chunks per topic.
- **Stateful Multi-Turn Doubt Tutor:** Retains topic notes and the last 8 conversation turns for context-aware Q&A.
- **Automated Testing (`tests/test_pathfinder.py`):** Fast, deterministic `pytest` unit tests covering Pydantic validation, MCQ grading edge cases, SQLite constraints & cascade deletes, RAG retrieval, and mocked Groq → Gemini failover.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (100% Free)
Copy `.env.example` to `.env` and add your free keys:
```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the App
https://github.com/LuckyG05/PathFinderAI.git```bash
python -m streamlit run app.py
```

### 4. Run Automated Tests
```bash
python -m pytest -v
```
