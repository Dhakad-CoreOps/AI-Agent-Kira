"""Agent 2: The Evaluation Cleanup Agent.

Removes duplicate rows from the evaluation store (data/kira.db). Re-running the
screening commands on the same documents inserts a fresh row each time, so the
store fills up with near-identical evaluations of the same candidate.

Two-step behaviour:

1. EXACT RE-RUNS  -> rows that screened the same resume against the same job
                     description are duplicates; only the newest row (and its
                     markdown report) is kept, the rest are deleted.
2. FUZZY MATCHES  -> the LLM clusters the surviving candidate names, and
                     entries that look like the same person under different
                     resume files are flagged for manual review — they are
                     never deleted automatically.
"""

import difflib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import get_llm
from src.exception import CustomException
from src.logger import logging
from src.storage.evaluation_store import delete_evaluations, list_evaluations

llm = get_llm(temperature=0.2)


class CleanupAgentState(TypedDict, total=False):
    dry_run: bool
    deleted_ids: List[int]
    agent_response: str


NAME_CLUSTERING_PROMPT = (
    "You are a data-cleaning assistant for an HR system. You will receive a JSON list of "
    "candidate names extracted from resumes. Group together names that clearly refer to the "
    "same person (case differences, initials, honorifics, obvious typos). Reply with ONLY a "
    'JSON list of lists, e.g. [["Jane Doe", "JANE DOE"], ["John Smith"]]. Every input name '
    "must appear in exactly one group. Different people must stay in separate groups — most "
    "groups should contain exactly one name. When in doubt, keep names separate."
)

# Two names may only share an LLM cluster if they are also this textually
# similar — a guard against the model lumping unrelated people together.
NAME_SIMILARITY_THRESHOLD = 0.65


def _norm_path(path: Optional[str]) -> Optional[str]:
    """Normalize a stored path so 'c:\\...' and 'C:\\...' compare equal on Windows."""
    return os.path.normcase(os.path.abspath(path)) if path else None


def _basename(path: Optional[str]) -> str:
    return os.path.basename(path) if path else "(no resume file)"


def _plausibly_same(a: str, b: str) -> bool:
    ratio = difflib.SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()
    return ratio >= NAME_SIMILARITY_THRESHOLD


def _split_implausible(cluster: List[str]) -> List[List[str]]:
    """Split an LLM cluster so a name only stays grouped with names it actually resembles."""
    subgroups: List[List[str]] = []
    for name in cluster:
        for subgroup in subgroups:
            if any(_plausibly_same(name, other) for other in subgroup):
                subgroup.append(name)
                break
        else:
            subgroups.append([name])
    return subgroups


def _cluster_names(names: List[str]) -> List[List[str]]:
    """Ask the LLM which candidate names refer to the same person.

    The cleanup must never fail because of a bad model reply, so any unusable
    answer degrades to case-insensitive exact-name grouping.
    """
    if len(names) > 1:
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=NAME_CLUSTERING_PROMPT),
                    HumanMessage(content=json.dumps(names)),
                ]
            )
            match = re.search(r"\[.*\]", response.content, re.DOTALL)
            groups = json.loads(match.group(0)) if match else None
            if (
                isinstance(groups, list)
                and all(
                    isinstance(group, list) and all(isinstance(name, str) for name in group)
                    for group in groups
                )
                and sorted(name for group in groups for name in group) == sorted(names)
            ):
                groups = [subgroup for group in groups for subgroup in _split_implausible(group)]
                logging.info(f"Name clustering grouped {len(names)} name(s) into {len(groups)} cluster(s)")
                return groups
            logging.warning(f"Name clustering reply was unusable: {response.content!r}")
        except Exception as e:
            logging.warning(f"Name clustering failed, falling back to exact matching: {e}")

    grouped: Dict[str, List[str]] = {}
    for name in names:
        grouped.setdefault(name.strip().lower(), []).append(name)
    return list(grouped.values())


def cleanup_agent_node(state: CleanupAgentState) -> CleanupAgentState:
    """Cleanup Agent node. Deletes exact re-run evaluations (unless dry_run) and
    reports fuzzy same-person matches, writing a markdown report to the state
    under 'agent_response'."""
    try:
        dry_run = bool(state.get("dry_run"))
        rows = list_evaluations()  # newest first
        logging.info(f"Cleanup Agent invoked over {len(rows)} evaluation(s) (dry_run={dry_run})")

        # 1. Exact re-runs: same resume screened against the same job description.
        #    Rows without a resume path fall back to the candidate's name as identity.
        groups: Dict[Tuple[Optional[str], Optional[str]], List[Dict[str, Any]]] = {}
        for row in rows:
            resume_key = _norm_path(row["resume_path"]) or f"name:{row['candidate_name'].strip().lower()}"
            key = (resume_key, _norm_path(row["job_description_path"]))
            groups.setdefault(key, []).append(row)

        duplicate_groups = [group for group in groups.values() if len(group) > 1]
        delete_ids = [row["id"] for group in duplicate_groups for row in group[1:]]

        # 2. Fuzzy matches among the survivors: same person (per the LLM) screened
        #    against the same job description but from different resume files.
        kept = [row for row in rows if row["id"] not in set(delete_ids)]
        clusters = _cluster_names(sorted({row["candidate_name"] for row in kept}))
        flagged: List[List[Dict[str, Any]]] = []
        for cluster in clusters:
            cluster_rows = [row for row in kept if row["candidate_name"] in cluster]
            by_jd: Dict[Optional[str], List[Dict[str, Any]]] = {}
            for row in cluster_rows:
                by_jd.setdefault(_norm_path(row["job_description_path"]), []).append(row)
            flagged.extend(jd_rows for jd_rows in by_jd.values() if len(jd_rows) > 1)

        verb = "Would delete" if dry_run else "Deleted"
        lines = ["## Evaluation Cleanup Report", ""]
        if duplicate_groups:
            lines.append(f"### {verb} {len(delete_ids)} exact re-run(s)")
            for group in duplicate_groups:
                keep = group[0]
                removed = ", ".join(f"#{row['id']}" for row in group[1:])
                lines.append(
                    f"- {keep['candidate_name']} ({_basename(keep['resume_path'])}): "
                    f"kept newest #{keep['id']}, removed {removed}"
                )
        else:
            lines.append("### No exact re-runs found")

        lines.append("")
        if flagged:
            lines.append("### Possible same-person entries (review manually, nothing deleted)")
            for cluster_rows in flagged:
                entries = ", ".join(
                    f"#{row['id']} {row['candidate_name']} ({_basename(row['resume_path'])})"
                    for row in cluster_rows
                )
                lines.append(f"- {entries}")
        else:
            lines.append("### No fuzzy same-person matches flagged")

        if not dry_run and delete_ids:
            delete_evaluations(delete_ids)

        logging.info(f"Cleanup Agent finished. {verb} ids: {delete_ids}")
        return {
            "dry_run": dry_run,
            "deleted_ids": delete_ids,
            "agent_response": "\n".join(lines),
        }

    except Exception as e:
        logging.error(f"Cleanup Agent failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Preview what would be cleaned up without touching the database.
    result = cleanup_agent_node({"dry_run": True})
    print(result["agent_response"])
