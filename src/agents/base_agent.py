"""Shared foundation for all Kira HR agents.

Not a class hierarchy — agents are LangGraph node functions. This module owns
the setup every agent would otherwise duplicate: environment loading, the Groq
API key check, project paths, and the LLM factory.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.logger import logging

# ---------------------------------------------------------------------------
# Environment (Groq free tier — key comes from .env, never hardcoded)
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"

# Project paths resolved from the repo root so they work no matter which
# directory Python is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

if not os.getenv("GROQ_API_KEY"):
    logging.error("GROQ_API_KEY is missing from the environment/.env file")
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to the .env file in the project root "
        "(get a free key at https://console.groq.com/keys)."
    )


def get_llm(temperature: float = 0.2) -> ChatGroq:
    """Build the shared Groq chat model. Agents pick their own temperature:
    0.2 for evaluative work, 0.0 for verbatim policy answers."""
    return ChatGroq(model=GROQ_MODEL, temperature=temperature)
