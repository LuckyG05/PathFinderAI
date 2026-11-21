from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from core.database import DatabaseManager
from core.llm_engine import LLMEngine
from core.rag_pipeline import extract_text_from_pdf
from core.schemas import grade_mcq_answer

load_dotenv(override=True)

DB_PATH = Path(__file__).resolve().parent / "pathfinder.db"

st.set_page_config(
    page_title="PathFinder AI · Study Companion",
    layout="wide",
)

# Initialize SQLite Database Manager
if "db" not in st.session_state:
    st.session_state.db = DatabaseManager(str(DB_PATH))
db: DatabaseManager = st.session_state.db

# Initialize Session State keys
defaults = {
    "active_roadmap_id": None,
    "active_topic_id": None,
    "doubt_chats": {},
    "current_quiz": {},
    "quiz_versions": {},
    "last_topic_quiz_feedback": {},
    "final_quiz": None,
    "final_quiz_version": 0,
    "last_final_quiz_feedback": None,
    "last_provider_used": None,
}
for key, default_val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


def get_llm_engine() -> LLMEngine:
    """Initializes the High-Availability LLM Engine (Groq primary + Gemini failover) from .env."""
    return LLMEngine()


# --- SIDEBAR: Configuration, New Roadmap + PDF RAG, and Saved Roadmaps ---
with st.sidebar:
    st.title("📝 PathFinder AI")
    st.caption("Structured AI Study Companion with RAG & SQLite Persistence")

    if st.button("How it works", use_container_width=True):
        st.session_state.active_roadmap_id = None
        st.session_state.active_topic_id = None
        st.rerun()

    st.divider()

    st.subheader("➕ Create Roadmap")
    new_subject = st.text_input("Subject or Skill", placeholder="e.g., Operating Systems, SQL, React.js")
    uploaded_pdf = st.file_uploader(
        "Optional: Upload Syllabus / Notes PDF",
        type=["pdf"],
        help="Grounds roadmap, notes, and quizzes in your uploaded PDF.",
    )

    if st.button("Generate Roadmap", use_container_width=True, type="primary"):
        if not new_subject.strip():
            st.warning("Please enter a subject name.")
        else:
            try:
                engine = get_llm_engine()
                with st.spinner("Extracting document context & generating structured roadmap..."):
                    pdf_context = ""
                    if uploaded_pdf is not None:
                        pdf_context = extract_text_from_pdf(uploaded_pdf)
                        if not pdf_context:
                            st.warning(
                                "Uploaded PDF contained no extractable text (it may be scanned images). "
                                "Generating roadmap without PDF context."
                            )
                    roadmap_resp = engine.generate_roadmap(new_subject.strip(), source_context=pdf_context)
                    st.session_state.last_provider_used = engine.last_provider_used
                    new_id = db.create_roadmap(roadmap_resp, source_context=pdf_context)
                    st.session_state.active_roadmap_id = new_id
                    st.session_state.active_topic_id = None
                    st.session_state.final_quiz = None
                    st.session_state.last_final_quiz_feedback = None
                    st.rerun()
            except Exception as e:
                st.error(f"Error generating roadmap: {e}")

    st.divider()
    st.subheader("Saved Roadmaps")
    saved_roadmaps = db.list_roadmaps()
    if not saved_roadmaps:
        st.info("No roadmaps yet. Create one above!")
    else:
        for rm in saved_roadmaps:
            done = rm["completed_topics"] or 0
            total = rm["total_topics"] or 0
            is_active = rm["id"] == st.session_state.active_roadmap_id
            label = f"{'👉 ' if is_active else ''}{rm['subject']} ({done}/{total})"

            col_rm, col_del = st.columns([5, 1])
            with col_rm:
                if st.button(label, key=f"rm_{rm['id']}", use_container_width=True):
                    st.session_state.active_roadmap_id = rm["id"]
                    st.session_state.active_topic_id = None
                    st.session_state.final_quiz = None
                    st.session_state.last_final_quiz_feedback = None
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_rm_{rm['id']}", help=f"Delete '{rm['subject']}'"):
                    db.delete_roadmap(rm["id"])
                    if st.session_state.active_roadmap_id == rm["id"]:
                        st.session_state.active_roadmap_id = None
                        st.session_state.active_topic_id = None
                        st.session_state.final_quiz = None
                        st.session_state.last_final_quiz_feedback = None
                    st.rerun()

    if st.session_state.last_provider_used:
        st.divider()
        st.caption(f"Last AI model: `{st.session_state.last_provider_used}`")


# --- MAIN WORKSPACE ---
if not st.session_state.active_roadmap_id:
    st.header("Welcome to PathFinder AI")
    st.markdown(
        """
        **How it works:**
        1. **Enter any subject** in the left sidebar (and optionally attach a **PDF syllabus or lecture notes** for RAG).
        2. **Study step-by-step:** Read structured AI notes and ask follow-up questions in the **stateful Doubt Tutor**.
        3. **Verify mastery:** Pass a **3-question Topic Quiz** (2/3 required) to mark the topic `completed`.
        4. **Take the Final Test:** Once topics are completed, test yourself across the entire roadmap!
        """
    )
    st.stop()

roadmap = db.get_roadmap(st.session_state.active_roadmap_id)
topics = db.get_topics_for_roadmap(st.session_state.active_roadmap_id)

if not roadmap or not topics:
    st.warning("Roadmap not found.")
    st.stop()

completed_topics = [t for t in topics if t["status"] == "completed"]
progress_ratio = len(completed_topics) / len(topics) if topics else 0.0

col_title, col_prog = st.columns([3, 1])
with col_title:
    st.header(f"{roadmap['subject']}")
    if roadmap.get("source_context"):
        st.caption("📄 **RAG Active:** Grounded in uploaded PDF document")
with col_prog:
    st.metric("Topics Completed", f"{len(completed_topics)} / {len(topics)}")
st.progress(progress_ratio)

topic_options = {
    t["id"]: (
        f"{t['order_index']}. {t['title']} "
        f"({'✅ Completed' if t['status'] == 'completed' else '🟡 In Progress' if t['status'] == 'in_progress' else '⚪ Not Started'})"
    )
    for t in topics
}
if st.session_state.active_topic_id not in topic_options:
    st.session_state.active_topic_id = topics[0]["id"]

selected_topic_id = st.selectbox(
    "Select a Roadmap Topic to Study:",
    options=list(topic_options.keys()),
    format_func=lambda tid: topic_options[tid],
    index=list(topic_options.keys()).index(st.session_state.active_topic_id),
)
st.session_state.active_topic_id = selected_topic_id
current_topic = db.get_topic(selected_topic_id)

tab_learn, tab_quiz, tab_final, tab_history = st.tabs(
    ["Study Notes & Doubt Tutor", "Topic Mini-Quiz", "Cumulative Final Test", "Quiz History"]
)

# --- TAB 1: STUDY NOTES & STATEFUL DOUBT TUTOR ---
with tab_learn:
    st.subheader(f"{current_topic['order_index']}. {current_topic['title']}")
    st.caption(current_topic["description"])

    if not current_topic["notes"]:
        if st.button("Generate Notes", type="primary"):
            try:
                engine = get_llm_engine()
                with st.spinner("Generating structured study notes..."):
                    notes_md = engine.generate_topic_notes(
                        subject=roadmap["subject"],
                        topic_title=current_topic["title"],
                        topic_description=current_topic["description"],
                        source_context=roadmap.get("source_context", ""),
                    )
                    st.session_state.last_provider_used = engine.last_provider_used
                    db.save_topic_notes(current_topic["id"], notes_md)
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to generate notes: {e}")
    else:
        st.markdown(current_topic["notes"])
        st.divider()
        st.subheader("Interactive Doubt Tutor (Stateful Chat)")
        tid = current_topic["id"]
        if tid not in st.session_state.doubt_chats:
            st.session_state.doubt_chats[tid] = []

        for msg in st.session_state.doubt_chats[tid]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_doubt = st.chat_input(f"Ask a question about '{current_topic['title']}'...")
        if user_doubt:
            with st.chat_message("user"):
                st.markdown(user_doubt)
            try:
                engine = get_llm_engine()
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        reply = engine.answer_doubt(
                            subject=roadmap["subject"],
                            topic_title=current_topic["title"],
                            topic_notes=current_topic["notes"],
                            chat_history=st.session_state.doubt_chats[tid],
                            user_question=user_doubt,
                        )
                        st.session_state.last_provider_used = engine.last_provider_used
                        st.markdown(reply)
                st.session_state.doubt_chats[tid].append({"role": "user", "content": user_doubt})
                st.session_state.doubt_chats[tid].append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 2: TOPIC MINI-QUIZ ---
with tab_quiz:
    st.subheader(f"Mini-Quiz: {current_topic['title']}")
    st.write("Score at least **2 out of 3** to mark this topic as **Completed**!")

    tid = current_topic["id"]
    if st.button("Generate Quiz"):
        try:
            engine = get_llm_engine()
            with st.spinner("Creating multiple-choice questions..."):
                st.session_state.current_quiz[tid] = engine.generate_quiz(
                    subject=roadmap["subject"],
                    topic_titles=[current_topic["title"]],
                    num_questions=3,
                    source_context=roadmap.get("source_context", ""),
                )
                st.session_state.last_provider_used = engine.last_provider_used
                st.session_state.quiz_versions[tid] = st.session_state.quiz_versions.get(tid, 0) + 1
                st.session_state.last_topic_quiz_feedback.pop(tid, None)
        except Exception as e:
            st.error(f"Error generating quiz: {e}")

    quiz_questions = st.session_state.current_quiz.get(tid)
    q_ver = st.session_state.quiz_versions.get(tid, 0)

    if quiz_questions:
        with st.form(key=f"quiz_form_{tid}_{q_ver}"):
            user_answers = {}
            for idx, q in enumerate(quiz_questions):
                st.markdown(f"**Q{idx + 1}. {q.question}**")
                options_map = {
                    "A": f"A) {q.option_a}",
                    "B": f"B) {q.option_b}",
                    "C": f"C) {q.option_c}",
                    "D": f"D) {q.option_d}",
                }
                user_answers[idx] = st.radio(
                    "Choose one:",
                    options=["A", "B", "C", "D"],
                    format_func=lambda x, om=options_map: om[x],
                    key=f"q_{tid}_{q_ver}_{idx}",
                    index=None,
                )
                st.write("")

            submitted = st.form_submit_button("Submit Topic Quiz", type="primary")
            if submitted:
                if any(ans is None for ans in user_answers.values()):
                    st.warning(
                        f"Please select an answer for all {len(quiz_questions)} questions before submitting."
                    )
                else:
                    score = 0
                    feedback_items = []
                    for idx, q in enumerate(quiz_questions):
                        chosen = user_answers[idx] or ""
                        is_correct = grade_mcq_answer(chosen, q.correct_option)
                        if is_correct:
                            score += 1
                            feedback_items.append(
                                ("success", f"**Q{idx + 1}: Correct! ({chosen})** — {q.explanation}")
                            )
                        else:
                            feedback_items.append(
                                (
                                    "error",
                                    f"**Q{idx + 1}: Incorrect (You chose {chosen}; Correct: {q.correct_option})** — {q.explanation}",
                                )
                            )

                    db.record_quiz_attempt(
                        roadmap_id=roadmap["id"],
                        topic_id=tid,
                        quiz_type="topic_quiz",
                        score=score,
                        total_questions=len(quiz_questions),
                    )
                    passed = score >= min(2, len(quiz_questions))
                    db.update_topic_status(tid, "completed" if passed else "in_progress")
                    st.session_state.last_topic_quiz_feedback[tid] = {
                        "items": feedback_items,
                        "score": score,
                        "total": len(quiz_questions),
                        "passed": passed,
                    }
                    st.rerun()

    saved_feedback = st.session_state.last_topic_quiz_feedback.get(tid)
    if saved_feedback:
        for level, text in saved_feedback["items"]:
            if level == "success":
                st.success(text)
            else:
                st.error(text)
        if saved_feedback["passed"]:
            st.success(
                f"You scored {saved_feedback['score']}/{saved_feedback['total']}! Topic marked as **Completed**."
            )
        else:
            st.warning(
                f"You scored {saved_feedback['score']}/{saved_feedback['total']}. "
                "Review the notes and try again to complete this topic!"
            )

# --- TAB 3: CUMULATIVE FINAL TEST ---
with tab_final:
    st.subheader("Cumulative Final Test")
    if not completed_topics:
        st.info("Complete at least one topic quiz first to unlock the Cumulative Final Test!")
    else:
        st.write(f"Testing across **{len(completed_topics)} completed topic(s)**.")
        num_final_q = min(len(completed_topics) * 2, 8)
        if st.button(f"Generate {num_final_q}-Question Final Test"):
            try:
                engine = get_llm_engine()
                with st.spinner("Generating cumulative final test..."):
                    st.session_state.final_quiz = engine.generate_quiz(
                        subject=roadmap["subject"],
                        topic_titles=[t["title"] for t in completed_topics],
                        num_questions=num_final_q,
                        source_context=roadmap.get("source_context", ""),
                    )
                    st.session_state.last_provider_used = engine.last_provider_used
                    st.session_state.final_quiz_version += 1
                    st.session_state.last_final_quiz_feedback = None
            except Exception as e:
                st.error(f"Error: {e}")

        if st.session_state.final_quiz:
            f_ver = st.session_state.final_quiz_version
            with st.form(f"final_quiz_form_{f_ver}"):
                final_answers = {}
                for idx, q in enumerate(st.session_state.final_quiz):
                    st.markdown(f"**Q{idx + 1}. {q.question}**")
                    options_map = {
                        "A": f"A) {q.option_a}",
                        "B": f"B) {q.option_b}",
                        "C": f"C) {q.option_c}",
                        "D": f"D) {q.option_d}",
                    }
                    final_answers[idx] = st.radio(
                        "Select option:",
                        options=["A", "B", "C", "D"],
                        format_func=lambda x, om=options_map: om[x],
                        key=f"final_q_{f_ver}_{idx}",
                        index=None,
                    )
                if st.form_submit_button("Submit Final Test", type="primary"):
                    if any(a is None for a in final_answers.values()):
                        st.warning("Please answer all questions before submitting.")
                    else:
                        final_score = 0
                        feedback_items = []
                        for idx, q in enumerate(st.session_state.final_quiz):
                            chosen = final_answers[idx] or ""
                            if grade_mcq_answer(chosen, q.correct_option):
                                final_score += 1
                                feedback_items.append(
                                    ("success", f"**Q{idx + 1}: Correct!** — {q.explanation}")
                                )
                            else:
                                feedback_items.append(
                                    (
                                        "error",
                                        f"**Q{idx + 1}: Incorrect (Correct: {q.correct_option})** — {q.explanation}",
                                    )
                                )
                        db.record_quiz_attempt(
                            roadmap_id=roadmap["id"],
                            topic_id=None,
                            quiz_type="final_test",
                            score=final_score,
                            total_questions=len(st.session_state.final_quiz),
                        )
                        st.session_state.last_final_quiz_feedback = {
                            "items": feedback_items,
                            "score": final_score,
                            "total": len(st.session_state.final_quiz),
                        }
                        st.rerun()

        if st.session_state.last_final_quiz_feedback:
            fb = st.session_state.last_final_quiz_feedback
            for level, text in fb["items"]:
                if level == "success":
                    st.success(text)
                else:
                    st.error(text)
            st.info(f"### Final Score: {fb['score']} / {fb['total']}")

# --- TAB 4: SQL QUIZ HISTORY ---
with tab_history:
    st.subheader("Quiz Attempt History (Stored in SQLite)")
    history = db.get_quiz_history(roadmap["id"])
    if not history:
        st.info("No quiz attempts recorded for this roadmap yet.")
    else:
        formatted_history = [
            {
                "Topic": row["topic_title"],
                "Quiz Type": "Topic Mini-Quiz" if row["quiz_type"] == "topic_quiz" else "Cumulative Final Test",
                "Score": f"{row['score']} / {row['total_questions']}",
                "Accuracy": f"{round((row['score'] / row['total_questions']) * 100)}%"
                if row["total_questions"]
                else "0%",
                "Attempted At": row["attempted_at"],
            }
            for row in history
        ]
        st.dataframe(formatted_history, use_container_width=True, hide_index=True)
