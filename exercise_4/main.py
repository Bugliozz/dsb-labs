"""Local runner for Exercise 4 assets."""

from __future__ import annotations

import argparse
import json

from app import collect_daily_flights, create_app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Exercise 4 tools.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("serve", "collect"),
        default="serve",
        help="`serve` starts the Flask app, `collect` runs one collection cycle.",
    )
    parser.add_argument(
        "--exercise",
        nargs="+",
        default=["4_1"],
        help="Exercise identifier. Only `4_1` is supported in this local runner.",
    )
    parser.add_argument("--date", help="Target date for `collect` mode in YYYY-MM-DD format.")
    parser.add_argument("--host", default="0.0.0.0", help="Host for `serve` mode.")
    parser.add_argument("--port", type=int, default=8080, help="Port for `serve` mode.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.exercise != ["4_1"]:
        raise SystemExit("Exercise 4 local runner only supports `4_1`.")

    if args.command == "collect":
        payload = collect_daily_flights(raw_date=args.date)
        print(json.dumps(payload, indent=2))
        return

    app = create_app()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
