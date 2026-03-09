"""Shared exercise metadata used by the root launcher and local runners."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

CATALOG: dict[str, dict[str, object]] = {
    "1": {
        "title": "Exercise 1",
        "runner": REPO_ROOT / "exercise_1" / "main.py",
        "description": "Plotting, EDA, and introductory Docker labs.",
        "exercises": {
            "1_2": "multi-panel matplotlib layout",
            "1_3": "composed seaborn charts",
            "1_4": "EDA on Indian Liver Patient Records",
            "1_5": "ETF correlation analysis",
            "1_6": "Docker + DokuWiki scaffold",
            "1_7": "Flask + Docker scaffold",
        },
    },
    "2": {
        "title": "Exercise 2",
        "runner": REPO_ROOT / "exercise_2" / "main.py",
        "description": "Kaggle ingestion plus MySQL, Metabase, Neo4j, and OpenSearch.",
        "exercises": {
            "2_1": "Kaggle -> MySQL ingestion",
            "2_2": "Metabase + MySQL exploration",
            "2_3": "MySQL -> Neo4j import",
            "2_4": "MySQL -> OpenSearch import",
        },
    },
    "3": {
        "title": "Exercise 3",
        "runner": REPO_ROOT / "exercise_3" / "main.py",
        "description": "Ames house-price training workflow and Flask inference API.",
        "commands": ("train", "serve"),
        "exercises": {
            "3_1": "Ames regression workflow and API",
        },
    },
}

ALL_TOKEN_ALIASES = {"all", "tutti", "*"}


def get_chapter(chapter_id: str) -> dict[str, object]:
    """Return a chapter entry or raise a ValueError."""
    normalized = normalize_chapter_token(chapter_id)
    if normalized not in CATALOG:
        valid = ", ".join(sorted(CATALOG))
        raise ValueError(f"Unknown chapter '{chapter_id}'. Valid values: {valid}")
    return CATALOG[normalized]


def normalize_chapter_token(token: str) -> str:
    """Normalize user-provided chapter IDs."""
    normalized = token.strip().lower().replace("exercise_", "").replace("exercise", "")
    if normalized in CATALOG:
        return normalized
    return normalized


def normalize_exercise_token(token: str, default_chapter: str | None = None) -> str:
    """Normalize exercise IDs like 1.2, 1-2, or bare local step numbers."""
    normalized = token.strip().lower().replace(".", "_").replace("-", "_")
    if normalized in ALL_TOKEN_ALIASES:
        return "all"
    if normalized.isdigit() and default_chapter is not None:
        return f"{default_chapter}_{normalized}"
    return normalized


def get_exercise_ids(chapter_id: str) -> list[str]:
    """Return the ordered exercise IDs for a chapter."""
    chapter = get_chapter(chapter_id)
    exercises = chapter["exercises"]
    return list(exercises)  # type: ignore[arg-type]


def get_exercise_description(exercise_id: str) -> str:
    """Return a description for an exercise ID."""
    chapter_id = get_exercise_chapter(exercise_id)
    return get_chapter(chapter_id)["exercises"][exercise_id]  # type: ignore[index]


def get_exercise_chapter(exercise_id: str) -> str:
    """Return the chapter ID for a normalized exercise token."""
    if "_" not in exercise_id:
        raise ValueError(f"Exercise ID '{exercise_id}' must look like <chapter>_<step>.")
    chapter_id, _ = exercise_id.split("_", 1)
    if chapter_id not in CATALOG:
        raise ValueError(f"Unknown chapter in exercise ID '{exercise_id}'.")
    return chapter_id


def parse_exercise_tokens(tokens: list[str], chapter_id: str | None = None) -> list[str]:
    """Validate and normalize a list of exercise tokens."""
    if not tokens:
        raise ValueError("At least one exercise must be provided.")

    if chapter_id is not None:
        available_ids = get_exercise_ids(chapter_id)
        available_set = set(available_ids)
        selected: list[str] = []
        for token in tokens:
            normalized = normalize_exercise_token(token, default_chapter=chapter_id)
            if normalized == "all":
                return available_ids
            if normalized not in available_set:
                valid = ", ".join(available_ids)
                raise ValueError(f"Unknown exercise '{token}'. Valid values: {valid}, all")
            if normalized not in selected:
                selected.append(normalized)
        return selected

    if any(normalize_exercise_token(token) == "all" for token in tokens):
        raise ValueError("Use 'all' only together with --chapter.")

    selected = []
    for token in tokens:
        normalized = normalize_exercise_token(token)
        chapter = get_exercise_chapter(normalized)
        if normalized not in get_exercise_ids(chapter):
            valid = ", ".join(get_exercise_ids(chapter))
            raise ValueError(f"Unknown exercise '{token}'. Valid values for chapter {chapter}: {valid}")
        if normalized not in selected:
            selected.append(normalized)
    return selected


def format_chapter_menu() -> str:
    """Render the interactive chapter menu."""
    lines = ["Available chapters:"]
    for chapter_id, chapter in CATALOG.items():
        lines.append(f"  {chapter_id}: {chapter['title']} - {chapter['description']}")
    return "\n".join(lines)


def format_exercise_menu(chapter_id: str) -> str:
    """Render the interactive exercise menu for a chapter."""
    chapter = get_chapter(chapter_id)
    lines = [f"{chapter['title']} options:"]
    exercises = chapter["exercises"]  # type: ignore[assignment]
    for exercise_id, description in exercises.items():
        lines.append(f"  {exercise_id}: {description}")
    lines.append("  all: run the full chapter")
    return "\n".join(lines)


def get_runner_path(chapter_id: str) -> Path:
    """Return the local runner path for a chapter."""
    return Path(get_chapter(chapter_id)["runner"])  # type: ignore[arg-type]


def get_chapter_commands(chapter_id: str) -> tuple[str, ...]:
    """Return optional command choices for a chapter."""
    chapter = get_chapter(chapter_id)
    return tuple(chapter.get("commands", ()))  # type: ignore[arg-type]
