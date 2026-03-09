"""Local runner for Exercise 1 assets only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_catalog import format_exercise_menu, parse_exercise_tokens


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Exercise 1 tools.")
    parser.add_argument(
        "-e",
        "--exercise",
        nargs="+",
        metavar="ID",
        help=(
            "Exercise ID to run (examples: 1_5, 1.5 or 5). "
            "You can pass multiple IDs, e.g. --exercise 1_3 1_5. "
            "Use 'all' to run all Exercise 1 steps."
        ),
    )
    return parser


def _prompt_exercise_selection() -> list[str]:
    menu = format_exercise_menu("1")
    prompt = (
        f"\n{menu}\n"
        "Select one or more Exercise 1 steps (use spaces or commas): "
    )
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("Enter at least one exercise or 'all'.")
            continue
        tokens = raw.replace(",", " ").split()
        try:
            return parse_exercise_tokens(tokens, chapter_id="1")
        except ValueError as exc:
            print(exc)


def run_selected_exercises(exercise_ids: list[str]) -> None:
    """Load the Exercise 1 core lazily and run the requested steps."""
    try:
        from exercise_1.core import run_exercises
    except ImportError as exc:
        print(f"Exercise 1 dependencies are missing: {exc}")
        print("Install them with: python -m pip install -r exercise_1/requirements.txt")
        raise SystemExit(1) from exc

    run_exercises(exercise_ids)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.exercise:
            selected_ids = parse_exercise_tokens(args.exercise, chapter_id="1")
        else:
            selected_ids = _prompt_exercise_selection()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_selected_exercises(selected_ids)


if __name__ == "__main__":
    main()
