"""Streamlit chatbot for the Kira HR FAQ Agent.

Employees ask policy questions in a chat UI; every answer comes from the FAQ
Agent node, which reads all official policy documents (handbook, leave policy,
IT policy, ... as .md/.txt/.pdf/.docx files) and refuses to invent policies
that are not in them.

Run with:
    streamlit run app.py
"""

import os
from typing import List, Tuple

import streamlit as st

from src.agents.faq_agent import (
    DEFAULT_POLICIES_DIR,
    SUPPORTED_POLICY_EXTENSIONS,
    faq_agent_node,
    load_policy_documents,
)
from src.retrieval.policy_index import get_policy_index

st.set_page_config(page_title="Kira HR — FAQ Assistant", page_icon="💬")


def _folder_fingerprint(policies_dir: str) -> Tuple[Tuple[str, float], ...]:
    """(filename, mtime) of every supported document — cache key that changes
    whenever a policy file is added, removed, or edited."""
    if not os.path.isdir(policies_dir):
        return ()
    return tuple(
        (name, os.path.getmtime(os.path.join(policies_dir, name)))
        for name in sorted(os.listdir(policies_dir))
        if os.path.splitext(name)[1].lower() in SUPPORTED_POLICY_EXTENSIONS
    )


@st.cache_data(show_spinner="Reading policy documents…")
def _load_policies(policies_dir: str, fingerprint) -> List[Tuple[str, str]]:
    """Extract text from all policy documents once, not on every chat turn.
    `fingerprint` only busts the cache when the folder contents change."""
    return load_policy_documents(policies_dir)


@st.cache_resource(show_spinner="Indexing policy documents…")
def _warm_index(policies_dir: str, fingerprint) -> None:
    """Build the semantic index up-front (embeddings are cached on disk), so
    the first question doesn't pay the indexing cost."""
    get_policy_index(_load_policies(policies_dir, fingerprint))


# ---------------------------------------------------------------------------
# Sidebar — policy folder and chat controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💬 Kira HR")
    st.caption("Internal HR FAQ Assistant")

    policies_dir = st.text_input(
        "Policies folder",
        value=DEFAULT_POLICIES_DIR,
        help="Folder containing the company's policy documents "
        "(.md, .txt, .pdf or .docx). All of them are used to answer.",
    )

    documents: List[Tuple[str, str]] = []
    try:
        fingerprint = _folder_fingerprint(policies_dir)
        documents = _load_policies(policies_dir, fingerprint)
        _warm_index(policies_dir, fingerprint)
        st.success(f"{len(documents)} policy document(s) loaded")
        for name, _ in documents:
            st.caption(f"📄 {name}")
    except Exception as e:
        st.error(str(e))

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Answers are based strictly on the official policy documents. "
        "Anything not covered there is referred to HR."
    )

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("HR FAQ Assistant")

if not st.session_state.messages:
    st.info(
        "Ask me anything about company policy — leave, working hours, payroll, "
        "notice periods…"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Chat input -> FAQ Agent node
# ---------------------------------------------------------------------------
if user_query := st.chat_input("e.g. How many paid leave days do I get?"):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Checking the policy documents…"):
            try:
                if not documents:
                    raise FileNotFoundError(
                        f"No policy documents loaded from {policies_dir} — "
                        "fix the folder in the sidebar first."
                    )
                result = faq_agent_node(
                    {
                        "user_query": user_query,
                        "policies_dir": policies_dir,
                        # Pass the cached extraction so PDFs are not re-parsed
                        # on every question; the agent retrieves only the
                        # chunks relevant to this question from them.
                        "policy_documents": documents,
                        "chat_history": [
                            (m["role"], m["content"]) for m in st.session_state.messages
                        ],
                    }
                )
                answer = result["final_response"]
            except Exception as e:
                answer = (
                    "Sorry, I could not process that question.\n\n"
                    f"```\n{e}\n```"
                )
        st.markdown(answer)

    # Persist the turn only after the agent replied, so a failed call can
    # simply be retried by re-sending the question.
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": answer})
