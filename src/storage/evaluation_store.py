"""Durable store for candidate evaluations produced by the Candidate Agent.

Two things are persisted for every agent run:

1. A row in a SQLite table (data/kira.db) holding the candidate's name, the
   match score, the recommendation, and — crucially — the path to the original
   resume, so a recruiter reading a summary can jump straight to the source CV.
2. A human-readable markdown file in data/evaluations/, for an audit trail that
   can be opened, shared or diffed without touching the database.
"""

import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.exception import CustomException
from src.logger import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "kira.db")
EVALUATION_DIR = os.path.join(DATA_DIR, "evaluations")

# The system only does resume screening; the task_type column is kept in the
# schema so existing rows stay readable, and every new row records this value.
TASK_TYPE = "resume_screening"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evaluations (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_name       TEXT    NOT NULL,
    task_type            TEXT    NOT NULL,
    match_score          INTEGER,
    recommendation       TEXT,
    summary_markdown     TEXT    NOT NULL,
    resume_path          TEXT,
    job_description_path TEXT,
    markdown_path        TEXT,
    created_at           TEXT    NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    """Create the evaluations table if it does not exist yet."""
    try:
        with _connect() as connection:
            connection.execute(CREATE_TABLE_SQL)
        logging.info(f"Evaluation store ready at {DB_PATH}")
    except Exception as e:
        logging.error(f"Failed to initialise evaluation store: {e}")
        raise CustomException(e, sys)


def parse_match_score(summary_markdown: str) -> Optional[int]:
    """Pull the '72 / 100' style score out of the agent's markdown, if present."""
    match = re.search(r"(\d{1,3})\s*/\s*100", summary_markdown)
    if not match:
        return None
    score = int(match.group(1))
    return score if 0 <= score <= 100 else None


def parse_recommendation(summary_markdown: str) -> Optional[str]:
    """Pull the Hire / No Hire verdict out of the agent's markdown, if present."""
    # 'No Hire' must be checked first — it contains the word 'Hire'.
    if re.search(r"no[\s\-]*hire", summary_markdown, re.IGNORECASE):
        return "No Hire"
    if re.search(r"\bhire\b", summary_markdown, re.IGNORECASE):
        return "Hire"
    return None


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug or "unknown_candidate"


def _write_markdown(
    candidate_name: str,
    summary_markdown: str,
    resume_path: Optional[str],
    timestamp: datetime,
) -> str:
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    filename = f"{_slugify(candidate_name)}_{TASK_TYPE}_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    markdown_path = os.path.join(EVALUATION_DIR, filename)

    header = [
        f"# {candidate_name}",
        "",
        f"- **Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if resume_path:
        # file:// URI so the link is clickable from a markdown preview.
        resume_uri = Path(os.path.abspath(resume_path)).as_uri()
        header.append(f"- **Resume:** [{os.path.basename(resume_path)}]({resume_uri})")
    header.extend(["", "---", ""])

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + summary_markdown + "\n")

    return markdown_path


def save_evaluation(
    candidate_name: str,
    summary_markdown: str,
    resume_path: Optional[str] = None,
    job_description_path: Optional[str] = None,
) -> int:
    """Persist one agent evaluation. Returns the new row's id."""
    try:
        init_db()
        timestamp = datetime.now()

        markdown_path = _write_markdown(
            candidate_name, summary_markdown, resume_path, timestamp
        )

        with _connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO evaluations (
                    candidate_name, task_type, match_score, recommendation,
                    summary_markdown, resume_path, job_description_path,
                    markdown_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_name,
                    TASK_TYPE,
                    parse_match_score(summary_markdown),
                    parse_recommendation(summary_markdown),
                    summary_markdown,
                    os.path.abspath(resume_path) if resume_path else None,
                    os.path.abspath(job_description_path) if job_description_path else None,
                    markdown_path,
                    timestamp.isoformat(timespec="seconds"),
                ),
            )
            evaluation_id = cursor.lastrowid

        logging.info(
            f"Saved evaluation #{evaluation_id} for '{candidate_name}' -> {markdown_path}"
        )
        return evaluation_id

    except Exception as e:
        logging.error(f"Failed to save evaluation for '{candidate_name}': {e}")
        raise CustomException(e, sys)


def list_evaluations(min_score: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return stored evaluations, newest first, optionally filtered by score."""
    try:
        init_db()
        query = (
            "SELECT id, candidate_name, match_score, recommendation, "
            "resume_path, created_at FROM evaluations"
        )
        params: tuple = ()
        if min_score is not None:
            query += " WHERE match_score >= ?"
            params = (min_score,)
        query += " ORDER BY created_at DESC, id DESC"

        with _connect() as connection:
            return [dict(row) for row in connection.execute(query, params)]

    except Exception as e:
        logging.error(f"Failed to list evaluations: {e}")
        raise CustomException(e, sys)


def get_evaluation(evaluation_id: int) -> Optional[Dict[str, Any]]:
    """Return one full evaluation (including the summary markdown) by id."""
    try:
        init_db()
        with _connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
            ).fetchone()
        return dict(row) if row else None

    except Exception as e:
        logging.error(f"Failed to fetch evaluation #{evaluation_id}: {e}")
        raise CustomException(e, sys)


def open_resume(evaluation_id: int) -> str:
    """Open the resume behind an evaluation in the OS default application.

    This is the 'I like this summary, show me the actual CV' step: the row keeps
    the original resume path, so the source document is always one call away.
    Returns the path that was opened.
    """
    try:
        evaluation = get_evaluation(evaluation_id)
        if evaluation is None:
            raise ValueError(f"No evaluation found with id {evaluation_id}")

        resume_path = evaluation["resume_path"]
        if not resume_path:
            raise ValueError(
                f"Evaluation #{evaluation_id} ({evaluation['candidate_name']}) has no resume "
                "on file — it was run from inline resume text rather than a document path."
            )
        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f"Resume for evaluation #{evaluation_id} is recorded at {resume_path}, "
                "but that file no longer exists."
            )

        logging.info(f"Opening resume for evaluation #{evaluation_id}: {resume_path}")

        if sys.platform == "win32":
            os.startfile(resume_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", resume_path], check=True)
        else:
            subprocess.run(["xdg-open", resume_path], check=True)

        return resume_path

    except Exception as e:
        logging.error(f"Failed to open resume for evaluation #{evaluation_id}: {e}")
        raise CustomException(e, sys)
