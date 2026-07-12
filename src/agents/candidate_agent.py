"""Agent 1: The Candidate Agent.

An objective AI Technical Recruiter node for the LangGraph multi-agent HR system.
Screens a resume against a job description and produces a structured markdown
evaluation sheet (match score, strengths, gaps, hire/no-hire recommendation).

Uses Groq's free API (GROQ_API_KEY loaded from the project .env file).
"""

import os
import sys
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.exception import CustomException
from src.logger import logging
from src.storage.evaluation_store import save_evaluation
from src.tools.file_reader import file_reader

# ---------------------------------------------------------------------------
# LLM configuration (Groq free tier — key comes from .env, never hardcoded)
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"

# Default document locations, resolved from the project root so they work no
# matter which directory Python is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESUME_DIR = os.path.join(DATA_DIR, "resumes")
JOB_DESCRIPTION_DIR = os.path.join(DATA_DIR, "job_descriptions")

if not os.getenv("GROQ_API_KEY"):
    logging.error("GROQ_API_KEY is missing from the environment/.env file")
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to the .env file in the project root "
        "(get a free key at https://console.groq.com/keys)."
    )

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.2,
)

# The agent is equipped with the file_reader tool. The node below invokes it
# deterministically whenever the state carries a document_path, which is more
# reliable for this fixed workflow than LLM-driven tool calling.
TOOLS = [file_reader]
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------
class CandidateAgentState(TypedDict, total=False):
    user_query: str
    document_path: Optional[str]
    job_description_path: Optional[str]
    job_description: Optional[str]
    resume_text: str
    candidate_name: str
    agent_response: str
    evaluation_id: int


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
BASE_SYSTEM_PROMPT = (
    "You are an objective AI Technical Recruiter for an HR system. "
    "You evaluate candidates strictly on evidence found in their resume and the "
    "job description. You never invent skills or experience that are not "
    "present in the provided documents, and you keep a neutral, professional tone."
)

RESUME_SCREENING_PROMPT = BASE_SYSTEM_PROMPT + """

TASK: RESUME SCREENING
Compare the candidate's resume against the job description and produce a
structured markdown evaluation sheet with EXACTLY these sections:

## Candidate Evaluation Sheet

### Match Score
A single overall score out of 100 (e.g. **72 / 100**) with one sentence of justification.

### Core Strengths
Exactly 3 bullet points. Each must cite concrete evidence from the resume.

### Technical Gaps
Exactly 3 bullet points. Each must name a requirement from the job description
that the resume does not demonstrate.

### Recommendation
A single verdict — **Hire** or **No Hire** — followed by a 2-3 sentence rationale.

Be strict and objective. If the resume text is missing or empty, say so instead
of guessing."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NAME_EXTRACTION_PROMPT = (
    "Extract the candidate's full name from the resume below. "
    "Reply with the name and nothing else — no punctuation, no explanation, no label. "
    "If no name is present, reply exactly: Unknown Candidate"
)


def _fallback_name(resume_text: str, document_path: Optional[str]) -> str:
    """Deterministic name guess used when the LLM extraction is unusable."""
    for line in resume_text.splitlines():
        candidate = line.strip()
        # A name line is short and has no resume-section punctuation.
        if 0 < len(candidate) <= 60 and not any(ch in candidate for ch in ":@|"):
            return candidate
    if document_path:
        return os.path.splitext(os.path.basename(document_path))[0].replace("_", " ").title()
    return "Unknown Candidate"


def _extract_candidate_name(resume_text: str, document_path: Optional[str]) -> str:
    """Ask the LLM for the candidate's name, falling back to the resume's first line.

    The name is what a recruiter scans the stored evaluations by, so it must never
    be the reason a run fails — any extraction problem degrades to the fallback.
    """
    if not resume_text.strip():
        return _fallback_name("", document_path)

    try:
        response = llm.invoke(
            [
                SystemMessage(content=NAME_EXTRACTION_PROMPT),
                HumanMessage(content=resume_text[:2000]),
            ]
        )
        name = response.content.strip().strip(".,'\"")

        # Guard against the model ignoring the instruction and returning prose.
        if name and len(name) <= 60 and "\n" not in name:
            logging.info(f"Extracted candidate name: {name}")
            return name

        logging.warning(f"Name extraction returned an unusable value: {name!r}")

    except Exception as e:
        logging.warning(f"Name extraction failed, falling back to heuristic: {e}")

    return _fallback_name(resume_text, document_path)


def _build_human_message(state: CandidateAgentState, resume_text: str, job_description: str) -> str:
    sections = [f"USER REQUEST:\n{state.get('user_query', '')}"]

    if resume_text:
        sections.append(f"CANDIDATE RESUME:\n{resume_text}")
    else:
        sections.append("CANDIDATE RESUME:\n(No resume text was provided.)")

    if job_description:
        sections.append(f"JOB DESCRIPTION:\n{job_description}")

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------
def candidate_agent_node(state: CandidateAgentState) -> CandidateAgentState:
    """Candidate Agent node. Reads the shared state, optionally loads a document
    via the file_reader tool, screens the resume, and writes the model's answer
    back into the state under 'agent_response'."""
    try:
        user_query = state.get("user_query", "")
        document_path = state.get("document_path")
        resume_text = state.get("resume_text", "")
        job_description_path = state.get("job_description_path")
        job_description = state.get("job_description", "")

        logging.info(f"Candidate Agent invoked. Query: {user_query!r}, document_path: {document_path!r}")

        # Load the documents through the file_reader tool when paths are provided
        # and the text has not already been extracted by an upstream node.
        if document_path and not resume_text:
            logging.info(f"Candidate Agent invoking file_reader tool for resume: {document_path}")
            resume_text = file_reader.invoke({"file_path": document_path})

        if job_description_path and not job_description:
            logging.info(f"Candidate Agent invoking file_reader tool for job description: {job_description_path}")
            job_description = file_reader.invoke({"file_path": job_description_path})

        messages = [
            SystemMessage(content=RESUME_SCREENING_PROMPT),
            HumanMessage(content=_build_human_message(state, resume_text, job_description)),
        ]

        response = llm.invoke(messages)
        summary_markdown = response.content
        logging.info(f"Candidate Agent completed screening. Response length: {len(summary_markdown)} chars")

        candidate_name = state.get("candidate_name") or _extract_candidate_name(
            resume_text, document_path
        )

        # Persist the evaluation so it outlives the process: a SQLite row keyed by
        # candidate name, plus a markdown copy on disk. The row keeps document_path,
        # which is what lets open_resume() pull up the original CV later.
        evaluation_id = save_evaluation(
            candidate_name=candidate_name,
            summary_markdown=summary_markdown,
            resume_path=document_path,
            job_description_path=job_description_path,
        )

        return {
            "resume_text": resume_text,
            "job_description": job_description,
            "candidate_name": candidate_name,
            "agent_response": summary_markdown,
            "evaluation_id": evaluation_id,
        }

    except Exception as e:
        logging.error(f"Candidate Agent failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick smoke test against the sample documents in data/.
    # Requires a valid GROQ_API_KEY in .env.
    sample_state: CandidateAgentState = {
        "user_query": "Screen this candidate for the Backend Engineer (C++) role.",
        "document_path": os.path.join(RESUME_DIR, "sample_candidate.txt"),
        "job_description_path": os.path.join(JOB_DESCRIPTION_DIR, "backend_engineer_cpp.txt"),
    }
    result = candidate_agent_node(sample_state)
    print(result["agent_response"])
