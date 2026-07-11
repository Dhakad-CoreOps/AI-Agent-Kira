"""Agent 1: The Candidate Agent.

An objective AI Technical Recruiter node for the LangGraph multi-agent HR system.
Handles two execution paths driven by the shared graph state:

1. RESUME SCREENING       -> structured markdown evaluation sheet (match score,
                             strengths, gaps, hire/no-hire recommendation).
2. INTERVIEW PREPARATION  -> 3 customized deep-technical interview questions.

Runs fully locally against an Ollama server (no API keys required).
"""

import sys
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.exception import CustomException
from src.logger import logging
from src.tools.file_reader import file_reader

# ---------------------------------------------------------------------------
# LLM configuration (local CPU inference via Ollama — no API keys)
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)

# The agent is equipped with the file_reader tool. The node below invokes it
# deterministically whenever the state carries a document_path, which is far
# more reliable on a small local model than LLM-driven tool calling.
TOOLS = [file_reader]
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------
class CandidateAgentState(TypedDict, total=False):
    user_query: str
    document_path: Optional[str]
    job_description: Optional[str]
    task_type: Literal["resume_screening", "interview_preparation"]
    resume_text: str
    agent_response: str


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

INTERVIEW_PREP_PROMPT = BASE_SYSTEM_PROMPT + """

TASK: INTERVIEW PREPARATION
Generate EXACTLY 3 highly customized technical interview questions for this
candidate, based strictly on the strengths and gaps visible in their background.

Rules:
- Target deep technical concepts relevant to the applicant's stack, such as
  C++ memory management (RAII, smart pointers, move semantics), Data Structures
  and Algorithms (complexity trade-offs, real applications), or backend
  architecture (caching, concurrency, database design).
- Each question must reference something specific from the candidate's resume
  (a project, technology, or claimed skill) — no generic textbook questions.
- Prefer probing questions in areas where the resume shows gaps, to verify depth.

Output markdown with EXACTLY these sections:

## Customized Interview Questions

### Question 1
The question, then a one-line note on *why this question* for this candidate.

### Question 2
Same format.

### Question 3
Same format."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
INTERVIEW_KEYWORDS = ("interview", "question", "prepare", "prep", "ask the candidate")


def _detect_task_type(user_query: str) -> str:
    """Infer the execution path from the user query when the graph has not set one."""
    query = (user_query or "").lower()
    if any(keyword in query for keyword in INTERVIEW_KEYWORDS):
        return "interview_preparation"
    return "resume_screening"


def _build_human_message(state: CandidateAgentState, resume_text: str) -> str:
    sections = [f"USER REQUEST:\n{state.get('user_query', '')}"]

    if resume_text:
        sections.append(f"CANDIDATE RESUME:\n{resume_text}")
    else:
        sections.append("CANDIDATE RESUME:\n(No resume text was provided.)")

    job_description = state.get("job_description")
    if job_description:
        sections.append(f"JOB DESCRIPTION:\n{job_description}")

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------
def candidate_agent_node(state: CandidateAgentState) -> CandidateAgentState:
    """Candidate Agent node. Reads the shared state, optionally loads a document
    via the file_reader tool, routes to the correct execution path, and writes
    the model's answer back into the state under 'agent_response'."""
    try:
        user_query = state.get("user_query", "")
        document_path = state.get("document_path")
        resume_text = state.get("resume_text", "")

        logging.info(f"Candidate Agent invoked. Query: {user_query!r}, document_path: {document_path!r}")

        # Load the document through the file_reader tool when a path is provided
        # and the text has not already been extracted by an upstream node.
        if document_path and not resume_text:
            logging.info(f"Candidate Agent invoking file_reader tool for: {document_path}")
            resume_text = file_reader.invoke({"file_path": document_path})

        task_type = state.get("task_type") or _detect_task_type(user_query)
        logging.info(f"Candidate Agent execution path: {task_type}")

        if task_type == "interview_preparation":
            system_prompt = INTERVIEW_PREP_PROMPT
        else:
            task_type = "resume_screening"
            system_prompt = RESUME_SCREENING_PROMPT

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=_build_human_message(state, resume_text)),
        ]

        response = llm.invoke(messages)
        logging.info(f"Candidate Agent completed {task_type}. Response length: {len(response.content)} chars")

        return {
            "resume_text": resume_text,
            "task_type": task_type,
            "agent_response": response.content,
        }

    except Exception as e:
        logging.error(f"Candidate Agent failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick local smoke test (requires the Ollama server to be running).
    sample_state: CandidateAgentState = {
        "user_query": "Screen this candidate for the Backend Engineer (C++) role.",
        "resume_text": (
            "Manas Dhakad — Software Developer. 2 years experience in C++ and Python. "
            "Built a multithreaded order-matching engine; strong in DSA (LeetCode 500+). "
            "Familiar with REST APIs using FastAPI. No production cloud experience."
        ),
        "job_description": (
            "Backend Engineer (C++): requires modern C++ (14/17), memory management, "
            "concurrency, Linux, and experience with distributed systems on AWS."
        ),
    }
    result = candidate_agent_node(sample_state)
    print(result["agent_response"])
