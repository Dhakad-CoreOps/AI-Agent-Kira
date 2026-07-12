"""Agent 2: The FAQ Agent.

An internal corporate HR Assistant node for the LangGraph multi-agent HR system.
Answers employee questions strictly from the official policy documents in
data/policies (every .md/.txt/.pdf/.docx in that folder), refusing to invent
policies that are not in the text.

Uses Groq's free API (GROQ_API_KEY loaded from the project .env file).
"""

import os
import sys
from typing import List, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.base_agent import DATA_DIR, get_llm
from src.exception import CustomException
from src.logger import logging
from src.tools.file_reader import SUPPORTED_TEXT_EXTENSIONS, file_reader

# Default policies folder under the shared data directory. Every supported
# document in it (handbook, leave policy PDF, IT policy, ...) is given to the
# agent.
DEFAULT_POLICIES_DIR = os.path.join(DATA_DIR, "policies")

SUPPORTED_POLICY_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | {".pdf", ".docx"}

# Temperature 0 — policy answers must be repeatable quotes of the documents,
# not creative writing.
llm = get_llm(temperature=0.0)


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------
class FAQAgentState(TypedDict, total=False):
    user_query: str
    # Folder of policy documents to answer from (defaults to data/policies).
    policies_dir: Optional[str]
    # Optional single-file override: answer from just this document instead.
    handbook_path: Optional[str]
    # Pre-extracted policy text; when set, no files are read (lets the UI cache
    # PDF extraction across chat turns).
    handbook_text: str
    # Prior conversation turns as (role, content) pairs, role being "user" or
    # "assistant". Lets the chatbot UI carry follow-up context between turns.
    chat_history: List[Tuple[str, str]]
    final_response: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
FALLBACK_ANSWER = (
    "I cannot find this policy in the official handbook. "
    "Please reach out to HR directly."
)

FAQ_SYSTEM_PROMPT = f"""You are an internal corporate HR Assistant for an HR system.
You answer employees' questions about company policy using ONLY the official
policy documents provided in the message below. Each document is delimited by
an "=== POLICY DOCUMENT: <filename> ===" header.

STRICT RULES:
1. Base every statement on the policy document text. Quote or paraphrase it
   faithfully.
2. NEVER invent, assume, or generalise a policy that is not explicitly written
   in the documents — not even if it is a common policy at other companies.
3. If none of the documents contain the answer, reply with exactly:
   "{FALLBACK_ANSWER}"
4. Keep a warm but professional tone, and format the answer as clean markdown
   (short headings or bullet points where they help readability).
5. Name the document (and section, if any) the answer comes from so the
   employee can verify it themselves."""


# ---------------------------------------------------------------------------
# Policy document loading
# ---------------------------------------------------------------------------
def load_policy_documents(policies_dir: str = DEFAULT_POLICIES_DIR) -> List[Tuple[str, str]]:
    """Read every supported policy document in the folder via the file_reader
    tool. Returns (filename, text) pairs, sorted by filename so the prompt is
    deterministic across runs."""
    if not os.path.isdir(policies_dir):
        raise FileNotFoundError(
            f"Policies folder not found: {policies_dir}. Create it and add your "
            "policy documents (.md, .txt, .pdf or .docx)."
        )

    paths = sorted(
        os.path.join(policies_dir, name)
        for name in os.listdir(policies_dir)
        if os.path.splitext(name)[1].lower() in SUPPORTED_POLICY_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(
            f"No policy documents found in {policies_dir}. Add your handbook / "
            "policy files (.md, .txt, .pdf or .docx) to that folder."
        )

    logging.info(f"FAQ Agent loading {len(paths)} policy document(s) from {policies_dir}")
    return [
        (os.path.basename(path), file_reader.invoke({"file_path": path}))
        for path in paths
    ]


def combine_policy_documents(documents: List[Tuple[str, str]]) -> str:
    """Merge (filename, text) pairs into one prompt block, each document under
    the header format the system prompt tells the model to expect."""
    return "\n\n".join(
        f"=== POLICY DOCUMENT: {name} ===\n\n{text}" for name, text in documents
    )


def _build_human_message(user_query: str, handbook_text: str) -> str:
    sections = [f"EMPLOYEE QUESTION:\n{user_query}"]

    if handbook_text:
        sections.append(f"OFFICIAL POLICY DOCUMENTS:\n{handbook_text}")
    else:
        sections.append("OFFICIAL POLICY DOCUMENTS:\n(No policy text was provided.)")

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------
def faq_agent_node(state: FAQAgentState) -> FAQAgentState:
    """FAQ Agent node. Reads the shared state, loads the policy documents via
    the file_reader tool, answers the employee's question strictly from that
    text, and writes the answer back into the state under 'final_response'."""
    try:
        user_query = state.get("user_query", "")
        policies_dir = state.get("policies_dir") or DEFAULT_POLICIES_DIR
        handbook_path = state.get("handbook_path")
        handbook_text = state.get("handbook_text", "")
        chat_history = state.get("chat_history", [])

        logging.info(f"FAQ Agent invoked. Query: {user_query!r}, policies_dir: {policies_dir!r}")

        # Load the policy text unless an upstream node (or the UI cache) has
        # already supplied it: a single file when handbook_path is set,
        # otherwise every supported document in the policies folder. Invoked
        # deterministically, which is more reliable for this fixed workflow
        # than LLM-driven tool calling.
        if not handbook_text:
            if handbook_path:
                logging.info(f"FAQ Agent invoking file_reader tool for single document: {handbook_path}")
                handbook_text = combine_policy_documents(
                    [(os.path.basename(handbook_path), file_reader.invoke({"file_path": handbook_path}))]
                )
            else:
                handbook_text = combine_policy_documents(load_policy_documents(policies_dir))

        # Replay recent turns (capped so the handbook + history stay well inside
        # the model's context window) before the current question. Only the
        # current message carries the handbook text.
        messages = [SystemMessage(content=FAQ_SYSTEM_PROMPT)]
        for role, content in chat_history[-6:]:
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(content=_build_human_message(user_query, handbook_text)))

        response = llm.invoke(messages)
        answer_markdown = response.content
        logging.info(f"FAQ Agent completed. Response length: {len(answer_markdown)} chars")

        return {
            "policies_dir": policies_dir,
            "handbook_text": handbook_text,
            "final_response": answer_markdown,
        }

    except Exception as e:
        logging.error(f"FAQ Agent failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick smoke test against the sample policies in data/policies/.
    # Requires a valid GROQ_API_KEY in .env.
    sample_state: FAQAgentState = {
        "user_query": "How many paid leave days do I get per year?",
    }
    result = faq_agent_node(sample_state)
    print(result["final_response"])
