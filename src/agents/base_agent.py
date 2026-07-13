"""Shared foundation for all Kira HR agents.

Not a class hierarchy — agents are LangGraph node functions. This module owns
the setup every agent would otherwise duplicate: environment loading, the Groq
API key check, project paths, and the LLM factory.

The factory returns Groq's hosted model with a local Ollama model as an
automatic fallback: any Groq failure (rate limit, network outage, missing key)
reroutes the call to the local model instead of failing the agent.
"""

import os
from typing import Optional, Sequence

from dotenv import load_dotenv
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from src.logger import logging

# ---------------------------------------------------------------------------
# Environment (Groq free tier — key comes from .env, never hardcoded)
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
# Local fallback served by Ollama. Kept small (3B) so CPU-only machines still
# answer at a usable speed when Groq is unavailable.
OLLAMA_MODEL = "llama3.2:3b"
# Ollama defaults to a 2K context, which is too small for the FAQ agent's
# retrieved excerpts (~3K tokens) plus chat history and answer.
OLLAMA_NUM_CTX = 8192

# Project paths resolved from the repo root so they work no matter which
# directory Python is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

if not os.getenv("GROQ_API_KEY"):
    logging.warning(
        "GROQ_API_KEY is not set - agents will run on the local Ollama model "
        f"({OLLAMA_MODEL}) only. Add the key to .env (get a free one at "
        "https://console.groq.com/keys) to use Groq."
    )


def _log_fallback(value):
    logging.warning(
        f"Groq API call failed - falling back to local Ollama model ({OLLAMA_MODEL})."
    )
    return value


def get_llm(temperature: float = 0.2, tools: Optional[Sequence] = None) -> Runnable:
    """Build the shared chat model. Agents pick their own temperature:
    0.2 for evaluative work, 0.0 for verbatim policy answers.

    Returns Groq with the local Ollama model as fallback, so callers just
    .invoke() and never see a Groq outage. Tools must be bound here (not on
    the returned runnable) because a fallback chain has no .bind_tools()."""
    local = ChatOllama(model=OLLAMA_MODEL, temperature=temperature, num_ctx=OLLAMA_NUM_CTX)
    if tools:
        local = local.bind_tools(list(tools))
    local_with_log = RunnableLambda(_log_fallback) | local

    if not os.getenv("GROQ_API_KEY"):
        return local

    groq = ChatGroq(model=GROQ_MODEL, temperature=temperature)
    if tools:
        groq = groq.bind_tools(list(tools))
    return groq.with_fallbacks([local_with_log])
