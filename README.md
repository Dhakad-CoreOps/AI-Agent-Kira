# Kira — AI Recruiter Agent

Kira is a LangGraph-based HR assistant that screens resumes against a job
description and keeps a durable, queryable record of every evaluation.

## What it does

1. **Screens candidates.** Given a resume and a job description, the
   Candidate Agent produces a structured markdown evaluation sheet — a match
   score out of 100, three core strengths, three technical gaps, and a
   Hire / No Hire recommendation — using Groq's `llama-3.1-8b-instant`.
2. **Persists every result.** Each evaluation is saved as a row in a local
   SQLite database (`data/kira.db`) and as a markdown file in
   `data/evaluations/`, so results survive restarts and can be browsed later.
3. **Cleans up duplicates.** Re-running screening on the same resume against
   the same job description happens often (retries, re-tests, etc.). The
   Cleanup Agent removes those exact duplicates automatically and flags
   likely same-person entries (same candidate under a different resume file)
   for manual review, without ever deleting those automatically.

## Architecture

```
review.py (CLI)
   │
   ├─ screen / screen-all ─► src/pipeline/hr_graph.py ─► candidate_agent_node ─► evaluation_store.py ─► data/kira.db + data/evaluations/*.md
   ├─ dedupe             ─► src/agents/cleanup_agent.py ────────────────────────► evaluation_store.py ─► data/kira.db (removes duplicate rows)
   └─ list / show / resume ────────────────────────────────────────────────────► evaluation_store.py (read-only)
```

- **`src/pipeline/hr_graph.py`** — compiles a LangGraph `StateGraph` with the
  Candidate Agent as its single node, checkpointed to `data/checkpoints.db`
  (via `SqliteSaver`) so each candidate's run is a durable, resumable thread.
- **`src/agents/candidate_agent.py`** — the Candidate Agent. Loads the resume
  and job description with the `file_reader` tool, prompts the LLM for a
  structured evaluation, extracts the candidate's name, and saves the result.
- **`src/agents/cleanup_agent.py`** — the Cleanup Agent. Deduplicates exact
  re-runs and flags fuzzy same-person matches using LLM name clustering with
  a text-similarity guard.
- **`src/agents/base_agent.py`** — reserved for shared logic once more agents
  are added (currently empty).
- **`src/storage/evaluation_store.py`** — all reads/writes to `data/kira.db`
  and the markdown reports in `data/evaluations/`.
- **`src/tools/file_reader.py`** — reads `.txt`, `.md`, `.pdf`, and `.docx`
  documents into plain text.
- **`src/logger.py`** / **`src/exception.py`** — shared logging (one log file
  per day under `logs/`) and a `CustomException` wrapper used across modules.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root with a free Groq API key
(get one at https://console.groq.com/keys):

```
GROQ_API_KEY=your-key-here
```

## Usage

```bash
# Screen a single resume against a job description
python review.py screen data/resumes/candidate.pdf data/job_descriptions/backend_engineer_cpp.txt

# Screen every resume in data/resumes/ against a job description
python review.py screen-all data/job_descriptions/backend_engineer_cpp.txt

# List stored evaluations (optionally filter by minimum score)
python review.py list
python review.py list --min-score 60

# View one full evaluation, or open the candidate's original resume
python review.py show <id>
python review.py resume <id>

# Remove duplicate evaluations (exact resume + job-description re-runs)
python review.py dedupe --dry-run   # preview only
python review.py dedupe             # actually delete
```

## Data layout

- `data/resumes/` — candidate resumes (gitignored, contains real PII)
- `data/job_descriptions/` — job description text files
- `data/kira.db` — SQLite store of every evaluation (gitignored)
- `data/evaluations/` — one markdown report per evaluation (gitignored)
- `data/checkpoints.db` — LangGraph's per-candidate run state
