"""Local runner for Exercise 3 assets only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_catalog import get_chapter_commands, parse_exercise_tokens


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Exercise 3 tools.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=get_chapter_commands("3"),
        default="train",
        help="`train` builds artifacts, `serve` starts the Flask inference app.",
    )
    parser.add_argument(
        "--exercise",
        nargs="+",
        default=["3_1"],
        help="Exercise identifier. Only `3_1` is supported in this local runner.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for `serve` mode.")
    parser.add_argument("--port", type=int, default=5000, help="Port for `serve` mode.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        selected_ids = parse_exercise_tokens(args.exercise, chapter_id="3")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if selected_ids != ["3_1"]:
        raise SystemExit("Exercise 3 local runner only supports `3_1`.")

    if args.command == "serve":
        try:
            from exercise_3.step_1.app import create_app
        except ImportError as exc:
            print(f"Exercise 3 dependencies are missing: {exc}")
            print("Install them with: python -m pip install -r exercise_3/requirements.txt")
            raise SystemExit(1) from exc

        app = create_app()
        app.run(host=args.host, port=args.port, debug=False)
        return

    try:
        from exercise_3.step_1.workflow import run_training_workflow
    except ImportError as exc:
        print(f"Exercise 3 dependencies are missing: {exc}")
        print("Install them with: python -m pip install -r exercise_3/requirements.txt")
        raise SystemExit(1) from exc

    run_training_workflow()


if __name__ == "__main__":
    main()
