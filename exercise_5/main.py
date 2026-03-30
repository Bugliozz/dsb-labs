"""Local runner for Exercise 5 assets only."""

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
    parser = argparse.ArgumentParser(description="Run Exercise 5 tools.")
    parser.add_argument(
        "-e",
        "--exercise",
        nargs="+",
        metavar="ID",
        help=(
            "Exercise ID to run (examples: 5_1, 5.1 or 1). "
            "You can pass multiple IDs, e.g. --exercise 5_1 5_2. "
            "Use 'all' to run all Exercise 5 steps."
        ),
    )
    return parser


def _prompt_exercise_selection() -> list[str]:
    menu = format_exercise_menu("5")
    prompt = (
        f"\n{menu}\n"
        "Select one or more Exercise 5 steps (use spaces or commas): "
    )
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("Enter at least one exercise or 'all'.")
            continue
        tokens = raw.replace(",", " ").split()
        try:
            return parse_exercise_tokens(tokens, chapter_id="5")
        except ValueError as exc:
            print(exc)


def run_selected_exercises(exercise_ids: list[str]) -> None:
    """Load the Exercise 5 core lazily and run the requested steps."""
    try:
        from exercise_5.core import run_exercises
    except ImportError as exc:
        print(f"Exercise 5 dependencies are missing: {exc}")
        print("Install them with: python -m pip install -r exercise_5/requirements.txt")
        raise SystemExit(1) from exc

    run_exercises(exercise_ids)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.exercise:
            selected_ids = parse_exercise_tokens(args.exercise, chapter_id="5")
        else:
            selected_ids = _prompt_exercise_selection()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_selected_exercises(selected_ids)


if __name__ == "__main__":
    main()
