"""Local runner for Exercise 2 assets only."""

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
    parser = argparse.ArgumentParser(description="Run Exercise 2 tools.")
    parser.add_argument(
        "-e",
        "--exercise",
        nargs="+",
        metavar="ID",
        help=(
            "Exercise ID to run (examples: 2_3 or 3). "
            "You can pass multiple IDs, e.g. --exercise 2_1 2_3. "
            "Use 'all' to run all Exercise 2 steps."
        ),
    )
    return parser


def _prompt_exercise_selection() -> list[str]:
    prompt = (
        f"\n{format_exercise_menu('2')}\n"
        "Select one or more Exercise 2 steps (use spaces or commas): "
    )
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("Enter at least one exercise or 'all'.")
            continue
        tokens = raw.replace(",", " ").split()
        try:
            return parse_exercise_tokens(tokens, chapter_id="2")
        except ValueError as exc:
            print(exc)


def run_selected_exercises(exercise_ids: list[str]) -> None:
    """Load the Exercise 2 core lazily and run the requested steps."""
    try:
        from exercise_2.core import run_exercises
    except ImportError as exc:
        print(f"Exercise 2 dependencies are missing: {exc}")
        print("Install them with: python -m pip install -r exercise_2/requirements.txt")
        raise SystemExit(1) from exc

    run_exercises(exercise_ids)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.exercise:
            selected_ids = parse_exercise_tokens(args.exercise, chapter_id="2")
        else:
            selected_ids = _prompt_exercise_selection()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_selected_exercises(selected_ids)


if __name__ == "__main__":
    main()
