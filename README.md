# PathFinder AI

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://pathfinderai-personalized-aitutor.streamlit.app/)

**Live Application:** https://pathfinderai-personalized-aitutor.streamlit.app/

> A structured AI Study Companion that turns any subject—and your own PDF syllabus or lecture notes—into a step-by-step learning roadmap, cached study notes, stateful doubt-clearing chat, mastery quizzes, and a cumulative final test.

---

## The Problem (Why I Built This)

While preparing for my college exams, I constantly struggled to build clear, exam-ready answers for every topic. I had to jump back and forth across scattered resources—college PDFs, lecture notes, study websites, and AI chatbots—just to piece together one solid explanation.

Even when I used general AI chatbots like ChatGPT:
- Responses rarely followed a consistent, structured study format.
- There was no guided learning order from fundamentals to advanced topics.
- Nothing checked whether I actually understood a topic before moving on, or tracked my progress across study sessions.

---

## What I Built

I built **PathFinder AI** to solve this end-to-end: an interactive web application that acts like a structured personal tutor.

Instead of giving scattered chat replies, PathFinder AI walks you through a complete learning loop for any subject:
1. **Generates a 5–7 step Learning Roadmap** (optionally grounded in your uploaded PDF syllabus or lecture notes).
2. **Teaches each topic in a consistent format** (Core Intuition, Key Concepts & Definitions, Practical Example/Code, and Summary Takeaways).
3. **Answers follow-up doubts** inside a stateful Doubt Tutor that remembers your notes and recent conversation.
4. **Tests your understanding** with a 3-question Topic Mini-Quiz (score 2/3 to mark the topic `Completed`).
5. **Evaluates overall mastery** with a Cumulative Final Test across all completed topics, saving your entire history in SQLite.

---

## Features

- **Structured Learning Roadmaps:** Enter any subject or skill (e.g., *Operating Systems*, *SQL*, *React.js*) to generate an ordered beginner-to-advanced curriculum.
- **PDF Notes & Syllabus Grounding (RAG):** Upload your own college notes or syllabus PDF. PathFinder AI extracts text with `[Page N]` tags, chunks it with a sliding window, and retrieves the most relevant sections so your roadmap, notes, and quizzes stay aligned with your actual coursework.
- **Consistent, Cached Study Notes:** Notes are generated once in clean Markdown and cached in SQLite (`topics.notes`), so revisiting a topic loads instantly with zero redundant API calls.
- **Stateful Interactive Doubt Tutor:** Ask follow-up questions per topic while the tutor retains both the topic notes and your recent conversation turns.
- **Topic Mini-Quizzes & Mastery Tracking:** Pass a 3-question conceptual MCQ quiz (≥ 2/3) with instant explanations to mark a topic `Completed` and update your live progress bar.
- **Cumulative Final Test & Quiz History:** Unlock a multi-topic final assessment once you complete topics, and review all past scores and accuracy percentages in the SQLite-backed Quiz History tab.
- **Saved Roadmaps Management:** Switch between multiple saved roadmaps anytime or delete old ones (with automatic cascading cleanup in SQLite).

---

## Engineering & Architecture Highlights

- **100% Free High-Availability LLM Router (`core/llm_engine.py`):** Routes primary traffic to **Groq Cloud** (`openai/gpt-oss-120b` → `qwen/qwen3.8-27b` → `openai/gpt-oss-20b`) for fast responses, with automatic failover to **Google Gemini Free Tier** (`gemini-3.5-flash` → `gemini-3.5-flash-lite` → `gemini-3.6-flash`) if rate limits or traffic spikes occur.
- **Strict Structured Outputs & Safe Grading (`core/schemas.py`):** Uses Pydantic v2 schemas (`RoadmapResponse`, `MCQQuestion`, `QuizBatchResponse`) to eliminate brittle string parsing, generate full quiz batches in a single API call, and grade MCQs deterministically (preventing edge-case bugs like empty input `"" in "A"` evaluating to `True`).
- **Normalized SQLite Persistence (`core/database.py`):** 3 relational tables (`roadmaps`, `topics`, `quiz_attempts`) with parameterized queries, `PRAGMA foreign_keys = ON`, `ON DELETE CASCADE`, `CHECK` constraints, and `LEFT JOIN + GROUP BY` progress aggregations.
- **Lightweight PDF RAG Pipeline (`core/rag_pipeline.py`):** `pypdf` extraction with `[Page N]` citations, overlapping sliding-window chunking (`chunk_size=1200, overlap=200`), and lexical keyword-overlap retrieval.
- **Automated Unit Testing (`tests/test_pathfinder.py`):** 26 offline `pytest` unit tests covering Pydantic validation, MCQ grading edge cases, SQLite constraints & cascade deletes, RAG retrieval, and mocked Groq → Gemini failover.

---

## Project Structure

```text
PathFinderAI/
├── app.py                     # Streamlit Web UI (Roadmaps, Notes, Doubt Tutor, Quizzes, History)
├── core/
│   ├── schemas.py             # Pydantic v2 structured output schemas & deterministic MCQ grader
│   ├── database.py            # SQLite manager (normalized tables, caching, SQL aggregations)
│   ├── rag_pipeline.py        # PDF extraction, overlapping chunking, and context retrieval
│   └── llm_engine.py          # High-availability Groq (Primary) -> Gemini (Failover) engine
├── tests/
│   └── test_pathfinder.py     # 26 offline pytest unit tests
├── .env.example               # API key configuration template
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Language & UI:** Python 3.12, Streamlit
- **LLMs & Validation:** Groq API (OpenAI-compatible SDK), Google GenAI SDK, Pydantic v2
- **Database:** SQLite (`sqlite3`)
- **Document Processing (RAG):** `pypdf`
- **Testing:** `pytest`

---

## Getting Started (Run Locally)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/LuckyG05/PathFinderAI.git
cd PathFinderAI
pip install -r requirements.txt
```

### 2. Configure API Keys (100% Free)
Copy `.env.example` to `.env` and add your free keys:
```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```
- Get a free Groq key: https://console.groq.com/keys
- Get a free Gemini key: https://aistudio.google.com/apikey

### 3. Run the Application
```bash
python -m streamlit run app.py
```

### 4. Run Automated Tests
```bash
python -m pytest -v
```
