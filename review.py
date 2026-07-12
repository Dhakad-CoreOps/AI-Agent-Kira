"""Recruiter CLI for the Kira HR system.

    python review.py screen     <resume> <job_description>   Run the Candidate Agent
    python review.py screen-all <job_description>            Screen every resume in data/resumes
    python review.py list       [--min-score N]              List stored evaluations
    python review.py show       <id>                         Print one full evaluation
    python review.py resume     <id>                         Open that candidate's CV
    python review.py dedupe     [--dry-run]                  Remove duplicate evaluations
"""

import argparse
import os
import sys

from src.agents.candidate_agent import RESUME_DIR
from src.agents.cleanup_agent import cleanup_agent_node
from src.pipeline.hr_graph import run_candidate
from src.storage.evaluation_store import get_evaluation, list_evaluations, open_resume
from src.tools.file_reader import SUPPORTED_TEXT_EXTENSIONS

SUPPORTED_RESUME_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | {".pdf", ".docx"}


def _cmd_run(args: argparse.Namespace, user_query: str) -> None:
    result = run_candidate(
        user_query=user_query,
        document_path=args.resume,
        job_description_path=args.job_description,
        thread_id=args.thread or args.resume,
    )
    print(f"\nCandidate: {result['candidate_name']}")
    print(f"Saved as evaluation #{result['evaluation_id']}\n")
    print(result["agent_response"])


def _cmd_screen_all(args: argparse.Namespace) -> None:
    resume_dir = args.dir or RESUME_DIR
    resumes = sorted(
        os.path.join(resume_dir, name)
        for name in os.listdir(resume_dir)
        if os.path.splitext(name)[1].lower() in SUPPORTED_RESUME_EXTENSIONS
    )
    if not resumes:
        print(f"No supported resumes found in {resume_dir}")
        sys.exit(1)

    print(f"Screening {len(resumes)} resume(s) from {resume_dir}\n")
    failed = 0
    for index, resume in enumerate(resumes, 1):
        print(f"[{index}/{len(resumes)}] {os.path.basename(resume)}")
        try:
            result = run_candidate(
                user_query="Screen this candidate against the job description.",
                document_path=resume,
                job_description_path=args.job_description,
                thread_id=resume,
            )
            print(
                f"    {result['candidate_name']} -> saved as evaluation #{result['evaluation_id']}"
            )
        except Exception as e:
            failed += 1
            print(f"    FAILED: {e}")

    print(f"\nDone: {len(resumes) - failed} succeeded, {failed} failed.")
    print("Compare candidates with:  python review.py list")


def _cmd_list(args: argparse.Namespace) -> None:
    rows = list_evaluations(min_score=args.min_score)
    if not rows:
        print("No evaluations stored yet.")
        return

    print(f"{'ID':<4} {'CANDIDATE':<24} {'SCORE':<7} {'VERDICT':<9} DATE")
    print("-" * 66)
    for row in rows:
        score = f"{row['match_score']}/100" if row["match_score"] is not None else "-"
        print(
            f"{row['id']:<4} {row['candidate_name'][:23]:<24} "
            f"{score:<7} {(row['recommendation'] or '-'):<9} {row['created_at']}"
        )
    print("\nOpen a candidate's resume with:  python review.py resume <id>")


def _cmd_show(args: argparse.Namespace) -> None:
    evaluation = get_evaluation(args.id)
    if evaluation is None:
        print(f"No evaluation found with id {args.id}")
        sys.exit(1)

    print(f"\n=== {evaluation['candidate_name']} - evaluation #{evaluation['id']} ===")
    print(f"Created: {evaluation['created_at']}")
    print(f"Resume:  {evaluation['resume_path'] or '(none on file)'}")
    print(f"Report:  {evaluation['markdown_path']}\n")
    print(evaluation["summary_markdown"])
    if evaluation["resume_path"]:
        print(f"\nOpen this resume with:  python review.py resume {evaluation['id']}")


def _cmd_resume(args: argparse.Namespace) -> None:
    path = open_resume(args.id)
    print(f"Opened resume: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kira HR — candidate review CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_screen = sub.add_parser("screen", help="Screen a resume against a job description")
    p_screen.add_argument("resume", help="Path to the candidate's resume")
    p_screen.add_argument("job_description", help="Path to the job description")
    p_screen.add_argument("--thread", help="Checkpoint thread id (defaults to the resume path)")

    p_all = sub.add_parser(
        "screen-all", help="Screen every resume in data/resumes against a job description"
    )
    p_all.add_argument("job_description", help="Path to the job description")
    p_all.add_argument("--dir", help="Folder to scan for resumes (defaults to data/resumes)")

    p_list = sub.add_parser("list", help="List stored evaluations")
    p_list.add_argument("--min-score", type=int, help="Only show candidates scoring at least N")

    p_show = sub.add_parser("show", help="Print one full evaluation")
    p_show.add_argument("id", type=int)

    p_resume = sub.add_parser("resume", help="Open the resume behind an evaluation")
    p_resume.add_argument("id", type=int)

    p_dedupe = sub.add_parser(
        "dedupe", help="Remove duplicate evaluations (same resume + job description re-runs)"
    )
    p_dedupe.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without touching the database",
    )

    args = parser.parse_args()

    if args.command == "screen":
        _cmd_run(args, "Screen this candidate against the job description.")
    elif args.command == "screen-all":
        _cmd_screen_all(args)
    elif args.command == "list":
        _cmd_list(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "resume":
        _cmd_resume(args)
    elif args.command == "dedupe":
        result = cleanup_agent_node({"dry_run": args.dry_run})
        print(result["agent_response"])


if __name__ == "__main__":
    main()
