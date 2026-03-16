"""Repository launcher that dispatches execution to chapter-local runners."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lab_catalog import (
    format_chapter_menu,
    format_exercise_menu,
    get_chapter,
    get_chapter_commands,
    get_exercise_chapter,
    get_runner_path,
    normalize_chapter_token,
    parse_exercise_tokens,
)

REPO_ROOT = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a chapter-local exercise runner.")
    parser.add_argument(
        "--chapter",
        help="Chapter to run (1, 2, 3, or 4). If omitted, an interactive chapter prompt is shown.",
    )
    parser.add_argument(
        "-e",
        "--exercise",
        nargs="+",
        metavar="ID",
        help=(
            "Exercise IDs to run. Examples: --exercise 1_2 2_4 3_1, "
            "--chapter 2 --exercise 1 4, or --chapter 2 --exercise all."
        ),
    )
    parser.add_argument(
        "--command",
        help="Command for Exercise 3 local runner (`train` or `serve`). Required for non-interactive Exercise 3 runs.",
    )
    return parser


def _prompt_chapter_selection() -> str:
    prompt = f"\n{format_chapter_menu()}\nSelect a chapter: "
    while True:
        raw = input(prompt).strip()
        try:
            normalized = normalize_chapter_token(raw)
            get_chapter(normalized)
            return normalized
        except ValueError as exc:
            print(exc)


def _prompt_exercise_selection(chapter_id: str) -> list[str]:
    prompt = (
        f"\n{format_exercise_menu(chapter_id)}\n"
        "Select one or more steps (use spaces or commas): "
    )
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("Enter at least one exercise or 'all'.")
            continue
        tokens = raw.replace(",", " ").split()
        try:
            return parse_exercise_tokens(tokens, chapter_id=chapter_id)
        except ValueError as exc:
            print(exc)


def _prompt_command_selection() -> str:
    commands = get_chapter_commands("3")
    prompt = (
        "\nExercise 3 command options:\n"
        + "\n".join(f"  {command}" for command in commands)
        + "\nSelect the command to run: "
    )
    while True:
        raw = input(prompt).strip().lower()
        if raw in commands:
            return raw
        print(f"Unknown command '{raw}'. Valid values: {', '.join(commands)}")


def _resolve_selection(args: argparse.Namespace) -> tuple[list[str], str | None]:
    chapter_id = None
    prompted = False

    if args.chapter:
        chapter_id = normalize_chapter_token(args.chapter)
        get_chapter(chapter_id)

    if args.exercise:
        selected_ids = parse_exercise_tokens(args.exercise, chapter_id=chapter_id)
    else:
        prompted = True
        if chapter_id is None:
            chapter_id = _prompt_chapter_selection()
        selected_ids = _prompt_exercise_selection(chapter_id)

    selected_chapters = {get_exercise_chapter(exercise_id) for exercise_id in selected_ids}
    if args.command and "3" not in selected_chapters:
        raise ValueError("--command can be used only when Exercise 3 is selected.")

    command = args.command
    if "3" in selected_chapters:
        valid_commands = get_chapter_commands("3")
        if command is None:
            if prompted:
                command = _prompt_command_selection()
            else:
                raise ValueError("Exercise 3 requires --command train|serve in non-interactive mode.")
        elif command not in valid_commands:
            raise ValueError(f"Unknown Exercise 3 command '{command}'. Valid values: {', '.join(valid_commands)}")

    return selected_ids, command


def _build_launch_batches(exercise_ids: list[str]) -> list[tuple[str, list[str]]]:
    batches: list[tuple[str, list[str]]] = []
    current_chapter = None
    current_ids: list[str] = []
    for exercise_id in exercise_ids:
        chapter_id = get_exercise_chapter(exercise_id)
        if chapter_id != current_chapter:
            if current_ids:
                batches.append((current_chapter, current_ids))
            current_chapter = chapter_id
            current_ids = [exercise_id]
        else:
            current_ids.append(exercise_id)
    if current_ids:
        batches.append((current_chapter, current_ids))
    return batches


def _run_runner(chapter_id: str, exercise_ids: list[str], command: str | None) -> int:
    runner_path = get_runner_path(chapter_id)
    if not runner_path.is_file():
        print(f"[Launcher] runner not found for chapter {chapter_id}: {runner_path}")
        return 1

    command_args = [sys.executable, str(runner_path)]
    if chapter_id == "3":
        command_args.append(command or "train")
    command_args.extend(["--exercise", *exercise_ids])

    print(f"\n[Launcher] dispatching chapter {chapter_id}: {' '.join(exercise_ids)}")
    result = subprocess.run(command_args, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(
            f"[Launcher] chapter {chapter_id} runner exited with code {result.returncode}. "
            "Check local dependencies or runner output above."
        )
    return result.returncode


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        selected_ids, command = _resolve_selection(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    exit_code = 0
    for chapter_id, exercise_ids in _build_launch_batches(selected_ids):
        run_exit_code = _run_runner(chapter_id, exercise_ids, command)
        if run_exit_code != 0:
            exit_code = run_exit_code
            break

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
