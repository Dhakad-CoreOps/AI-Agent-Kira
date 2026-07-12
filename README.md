# Kira — AI HR Agent System

A multi-agent HR assistant built with **LangGraph** and **Groq** (`llama-3.1-8b-instant`, free tier). Kira screens job candidates against job descriptions, answers employee policy questions from official documents, and keeps its own evaluation store clean.

## Agents

| Agent | File | What it does |
|---|---|---|
| **Candidate Agent** | `src/agents/candidate_agent.py` | Screens a resume against a job description and produces a structured evaluation sheet (match score, strengths, gaps, hire/no-hire). Every evaluation is persisted to SQLite + a markdown report. |
| **FAQ Agent** | `src/agents/faq_agent.py` | Internal HR assistant. Answers employee questions **strictly** from the policy documents in `data/policies/` — if a policy isn't in the documents, it says so and refers the employee to HR instead of guessing. |
| **Cleanup Agent** | `src/agents/cleanup_agent.py` | Deduplicates the evaluation store: deletes exact re-runs, flags fuzzy name matches for manual review. |

All agents are LangGraph node functions sharing common setup (LLM factory, paths, API-key check) from `src/agents/base_agent.py`.

## Setup

```bash
git clone <repo-url>
cd AI-Agent-Kira
pip install -r requirements.txt
```

Create a `.env` file in the project root with your free Groq API key ([get one here](https://console.groq.com/keys)):

```
GROQ_API_KEY=your_key_here
```

## Usage

### FAQ chatbot (web UI)

Put your policy documents (`.md`, `.txt`, `.pdf`, `.docx`) in `data/policies/`, then:

```bash
streamlit run app.py
```

Opens a chat interface in your browser (http://localhost:8501). The assistant answers from **all** documents in the folder, cites the source document for each answer, and supports follow-up questions. PDF text extraction is cached and refreshes automatically when you add or edit a file.

### Candidate screening (CLI)

```bash
python review.py screen <resume> <job_description>   # Screen one candidate
python review.py screen-all <job_description>        # Screen every resume in data/resumes
python review.py list [--min-score N]                # List stored evaluations
python review.py show <id>                           # Print one full evaluation
python review.py resume <id>                         # Open that candidate's CV
python review.py dedupe [--dry-run]                  # Remove duplicate evaluations
```

## Project structure

```
├── app.py                      # Streamlit chatbot for the FAQ Agent
├── review.py                   # Recruiter CLI for the Candidate/Cleanup Agents
├── data/
│   ├── policies/               # Policy documents the FAQ Agent answers from
│   ├── resumes/                # Candidate resumes to screen
│   ├── job_descriptions/       # Job descriptions to screen against
│   ├── evaluations/            # Generated markdown evaluation reports
│   └── kira.db                 # SQLite evaluation store
└── src/
    ├── agents/
    │   ├── base_agent.py       # Shared LLM factory, paths, env/key check
    │   ├── candidate_agent.py  # Agent 1 — resume screening
    │   ├── faq_agent.py        # Agent 2 — policy FAQ
    │   └── cleanup_agent.py    # Evaluation store deduplication
    ├── pipeline/hr_graph.py    # LangGraph wiring + SQLite checkpointing
    ├── storage/evaluation_store.py
    ├── tools/file_reader.py    # .txt/.md/.pdf/.docx text extraction tool
    ├── exception.py            # Custom exception with file/line context
    └── logger.py               # Daily log files under logs/
```

## Tech stack

- **LangGraph** — agent orchestration with durable SQLite checkpointing
- **langchain-groq** — `llama-3.1-8b-instant` via Groq's free API
- **Streamlit** — FAQ chatbot UI
- **SQLite** — evaluation store and graph checkpoints
- **pypdf / python-docx** — document text extraction
