"""Recruiter CLI for the Kira HR system.

    python review.py screen   <resume> <job_description>   Run the Candidate Agent
    python review.py prep     <resume> <job_description>   Generate interview questions
    python review.py list     [--min-score N]              List stored evaluations
    python review.py show     <id>                         Print one full evaluation
    python review.py resume   <id>                         Open that candidate's CV
"""

import argparse
import sys

from src.pipeline.hr_graph import run_candidate
from src.storage.evaluation_store import get_evaluation, list_evaluations, open_resume


def _cmd_run(args: argparse.Namespace, user_query: str, task_type: str) -> None:
    result = run_candidate(
        user_query=user_query,
        document_path=args.resume,
        job_description_path=args.job_description,
        task_type=task_type,
        thread_id=args.thread or args.resume,
    )
    print(f"\nCandidate: {result['candidate_name']}")
    print(f"Saved as evaluation #{result['evaluation_id']}\n")
    print(result["agent_response"])


def _cmd_list(args: argparse.Namespace) -> None:
    rows = list_evaluations(min_score=args.min_score)
    if not rows:
        print("No evaluations stored yet.")
        return

    print(f"{'ID':<4} {'CANDIDATE':<24} {'TASK':<22} {'SCORE':<7} {'VERDICT':<9} DATE")
    print("-" * 88)
    for row in rows:
        score = f"{row['match_score']}/100" if row["match_score"] is not None else "-"
        print(
            f"{row['id']:<4} {row['candidate_name'][:23]:<24} {row['task_type']:<22} "
            f"{score:<7} {(row['recommendation'] or '-'):<9} {row['created_at']}"
        )
    print("\nOpen a candidate's resume with:  python review.py resume <id>")


def _cmd_show(args: argparse.Namespace) -> None:
    evaluation = get_evaluation(args.id)
    if evaluation is None:
        print(f"No evaluation found with id {args.id}")
        sys.exit(1)

    print(f"\n=== {evaluation['candidate_name']} - evaluation #{evaluation['id']} ===")
    print(f"Task:    {evaluation['task_type']}")
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

    for name, help_text in [
        ("screen", "Screen a resume against a job description"),
        ("prep", "Generate interview questions for a candidate"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("resume", help="Path to the candidate's resume")
        p.add_argument("job_description", help="Path to the job description")
        p.add_argument("--thread", help="Checkpoint thread id (defaults to the resume path)")

    p_list = sub.add_parser("list", help="List stored evaluations")
    p_list.add_argument("--min-score", type=int, help="Only show candidates scoring at least N")

    p_show = sub.add_parser("show", help="Print one full evaluation")
    p_show.add_argument("id", type=int)

    p_resume = sub.add_parser("resume", help="Open the resume behind an evaluation")
    p_resume.add_argument("id", type=int)

    args = parser.parse_args()

    if args.command == "screen":
        _cmd_run(
            args,
            "Screen this candidate against the job description.",
            task_type="resume_screening",
        )
    elif args.command == "prep":
        _cmd_run(
            args,
            "Prepare technical interview questions for this candidate.",
            task_type="interview_preparation",
        )
    elif args.command == "list":
        _cmd_list(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "resume":
        _cmd_resume(args)


if __name__ == "__main__":
    main()
