"""LangGraph wiring for the HR multi-agent system.

Currently a single node (the Candidate Agent), compiled with a SQLite
checkpointer so every state transition is durable and each candidate can be
resumed as its own conversation thread. Additional agents get added as nodes
here without changing how state is persisted.
"""

import os
import sqlite3
import sys
from typing import Any, Dict, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agents.candidate_agent import CandidateAgentState, candidate_agent_node
from src.exception import CustomException
from src.logger import logging
from src.storage.evaluation_store import DATA_DIR

CHECKPOINT_DB_PATH = os.path.join(DATA_DIR, "checkpoints.db")


def _checkpointer() -> SqliteSaver:
    os.makedirs(DATA_DIR, exist_ok=True)
    # check_same_thread=False so the saver can be used from the graph's threads.
    connection = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    return SqliteSaver(connection)


def build_graph() -> Any:
    """Compile the HR graph with durable checkpointing."""
    try:
        builder = StateGraph(CandidateAgentState)
        builder.add_node("candidate_agent", candidate_agent_node)
        builder.add_edge(START, "candidate_agent")
        builder.add_edge("candidate_agent", END)

        graph = builder.compile(checkpointer=_checkpointer())
        logging.info(f"HR graph compiled with checkpointer at {CHECKPOINT_DB_PATH}")
        return graph

    except Exception as e:
        logging.error(f"Failed to build HR graph: {e}")
        raise CustomException(e, sys)


def run_candidate(
    user_query: str,
    document_path: Optional[str] = None,
    job_description_path: Optional[str] = None,
    thread_id: str = "default",
) -> Dict[str, Any]:
    """Run one candidate through the graph.

    thread_id scopes the checkpoint history — use one per candidate so their
    screening runs share a resumable conversation.
    """
    try:
        graph = build_graph()
        state: CandidateAgentState = {
            "user_query": user_query,
            "document_path": document_path,
            "job_description_path": job_description_path,
        }
        config = {"configurable": {"thread_id": thread_id}}

        logging.info(f"Running HR graph on thread '{thread_id}' for document: {document_path}")
        return graph.invoke(state, config=config)

    except Exception as e:
        logging.error(f"HR graph run failed on thread '{thread_id}': {e}")
        raise CustomException(e, sys)
