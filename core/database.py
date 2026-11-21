import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from core.schemas import RoadmapResponse


class DatabaseManager:
    """Handles all SQLite database connections, schema setup, and SQL queries."""

    def __init__(self, db_path: str = "pathfinder.db"):
        self.db_path = str(db_path)
        self.init_db()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Opens a SQLite connection with Foreign Keys enabled and always closes it cleanly."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initializes the normalized relational schema (roadmaps, topics, quiz_attempts)."""
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS roadmaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    source_context TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    roadmap_id INTEGER NOT NULL,
                    order_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'not_started'
                        CHECK(status IN ('not_started', 'in_progress', 'completed')),
                    FOREIGN KEY (roadmap_id) REFERENCES roadmaps(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS quiz_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    roadmap_id INTEGER NOT NULL,
                    topic_id INTEGER,
                    quiz_type TEXT NOT NULL CHECK(quiz_type IN ('topic_quiz', 'final_test')),
                    score INTEGER NOT NULL CHECK(score >= 0),
                    total_questions INTEGER NOT NULL CHECK(total_questions > 0),
                    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (roadmap_id) REFERENCES roadmaps(id) ON DELETE CASCADE,
                    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
                );
                """
            )

    def create_roadmap(self, roadmap_data: RoadmapResponse, source_context: str = "") -> int:
        """Inserts a new roadmap and all its topics in a single database transaction."""
        with self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO roadmaps (subject, source_context) VALUES (?, ?)",
                (roadmap_data.subject, source_context),
            )
            roadmap_id = int(cursor.lastrowid)

            for topic in roadmap_data.topics:
                conn.execute(
                    """
                    INSERT INTO topics (roadmap_id, order_index, title, description, status)
                    VALUES (?, ?, ?, ?, 'not_started')
                    """,
                    (roadmap_id, topic.order_index, topic.title, topic.description),
                )
            return roadmap_id

    def delete_roadmap(self, roadmap_id: int) -> bool:
        """Deletes a roadmap and cascades removal to its topics and quiz attempts."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM roadmaps WHERE id = ?", (roadmap_id,))
            return cursor.rowcount > 0

    def list_roadmaps(self) -> List[Dict[str, Any]]:
        """Returns all saved roadmaps along with topic completion counts using SQL JOIN & GROUP BY."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    r.id,
                    r.subject,
                    r.created_at,
                    COUNT(t.id) AS total_topics,
                    COALESCE(SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END), 0) AS completed_topics
                FROM roadmaps r
                LEFT JOIN topics t ON r.id = t.roadmap_id
                GROUP BY r.id
                ORDER BY r.created_at DESC, r.id DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def get_roadmap(self, roadmap_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a single roadmap by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM roadmaps WHERE id = ?", (roadmap_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_topics_for_roadmap(self, roadmap_id: int) -> List[Dict[str, Any]]:
        """Fetches all topics for a roadmap ordered by their step number."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM topics WHERE roadmap_id = ? ORDER BY order_index ASC, id ASC",
                (roadmap_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_topic(self, topic_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a single topic by its ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM topics WHERE id = ?", (topic_id,)
            ).fetchone()
            return dict(row) if row else None

    def save_topic_notes(self, topic_id: int, notes: str) -> None:
        """Caches generated AI notes in the database and marks the topic as 'in_progress' if not completed."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE topics
                SET notes = ?,
                    status = CASE WHEN status = 'completed' THEN 'completed' ELSE 'in_progress' END
                WHERE id = ?
                """,
                (notes, topic_id),
            )

    def update_topic_status(self, topic_id: int, status: str) -> None:
        """Updates a topic's status ('not_started', 'in_progress', or 'completed')."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE topics SET status = ? WHERE id = ?",
                (status, topic_id),
            )

    def record_quiz_attempt(
        self,
        roadmap_id: int,
        quiz_type: str,
        score: int,
        total_questions: int,
        topic_id: Optional[int] = None,
    ) -> None:
        """Records a quiz or final test score in the quiz_attempts table."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO quiz_attempts (roadmap_id, topic_id, quiz_type, score, total_questions)
                VALUES (?, ?, ?, ?, ?)
                """,
                (roadmap_id, topic_id, quiz_type, score, total_questions),
            )

    def get_quiz_history(self, roadmap_id: int) -> List[Dict[str, Any]]:
        """Returns quiz attempt history for a roadmap, joining topic titles when available."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    qa.id,
                    qa.quiz_type,
                    qa.score,
                    qa.total_questions,
                    qa.attempted_at,
                    COALESCE(t.title, 'Cumulative Final Test') AS topic_title
                FROM quiz_attempts qa
                LEFT JOIN topics t ON qa.topic_id = t.id
                WHERE qa.roadmap_id = ?
                ORDER BY qa.attempted_at DESC, qa.id DESC
                """,
                (roadmap_id,),
            ).fetchall()
            return [dict(row) for row in rows]
