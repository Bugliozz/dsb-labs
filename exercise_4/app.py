"""Flask app that collects Malpensa flight data locally or in GCS."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from flask import Flask, jsonify, request

try:
    from google.cloud import storage
except ImportError:  # pragma: no cover - depends on local optional dependency state
    storage = None  # type: ignore[assignment]

API_URL = "https://apiextra.seamilano.eu/ols-flights/v1/en/operative/flights/lists"
AIRPORT_REFERENCE_IATA = "mxp"
MOVEMENT_TYPES = ("D", "A")
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "malpensa_flights"

REQUEST_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-GB,en;q=0.9,it-IT;q=0.8,it;q=0.7",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://milanomalpensa-airport.com",
    "Referer": "https://milanomalpensa-airport.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    ),
    "keyId": "6bc034ea-ae66-40ce-891e-3dccf63cb2eb",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
}


def get_flights(movement_type: str, date_from: str, date_to: str, airport_reference_iata: str) -> dict[str, Any]:
    """Request the upstream Malpensa endpoint for one movement type."""
    params = {
        "movementType": movement_type,
        "dateFrom": date_from,
        "dateTo": date_to,
        "loadingType": "P",
        "airportReferenceIata": airport_reference_iata,
        "mfFlightType": "P",
    }
    response = requests.get(API_URL, params=params, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def resolve_target_date(raw_date: str | None) -> str:
    """Use the requested date or default to yesterday in UTC."""
    if raw_date:
        return pd.to_datetime(raw_date).strftime("%Y-%m-%d")
    yesterday = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def resolve_output_dir(raw_output_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the filesystem folder used for local backups."""
    if raw_output_dir:
        return Path(raw_output_dir)
    env_output_dir = os.getenv("FLIGHTS_OUTPUT_DIR")
    return Path(env_output_dir) if env_output_dir else DEFAULT_OUTPUT_DIR


def _serialize_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _upload_to_bucket(bucket_name: str, file_name: str, payload: str) -> str:
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage is not installed. Install exercise_4/requirements.txt "
            "before using BUCKET_NAME uploads."
        )
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(payload, content_type="application/json")
    return f"gs://{bucket_name}/{file_name}"


def _write_to_local_file(output_dir: Path, file_name: str, payload: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / file_name
    target_path.write_text(payload, encoding="utf-8")
    return str(target_path)


def collect_daily_flights(
    raw_date: str | None = None,
    bucket_name: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Collect departure and arrival data for the target date."""
    date = resolve_target_date(raw_date)
    date_from = f"{date} 00:00"
    date_to = f"{date} 23:59"
    bucket_name = bucket_name or os.getenv("BUCKET_NAME")
    resolved_output_dir = resolve_output_dir(output_dir)
    storage_mode = "gcs" if bucket_name else "local"
    file_records: list[dict[str, str]] = []

    for movement_type in MOVEMENT_TYPES:
        flights = get_flights(movement_type, date_from, date_to, AIRPORT_REFERENCE_IATA)
        file_name = f"{date}_{movement_type}.json"
        serialized_payload = _serialize_payload(flights)
        if bucket_name:
            target = _upload_to_bucket(bucket_name, file_name, serialized_payload)
        else:
            target = _write_to_local_file(resolved_output_dir, file_name, serialized_payload)
        file_records.append({"movement_type": movement_type, "target": target})

    return {
        "message": f"{date} Flights data uploaded successfully.",
        "date": date,
        "storage_mode": storage_mode,
        "bucket_name": bucket_name,
        "output_dir": str(resolved_output_dir),
        "files": file_records,
    }


def create_app(
    bucket_name: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
) -> Flask:
    """Create the scraper app with a collection and health endpoint."""
    app = Flask(__name__)

    @app.get("/")
    def home():
        try:
            payload = collect_daily_flights(
                raw_date=request.args.get("date"),
                bucket_name=bucket_name,
                output_dir=output_dir,
            )
            return jsonify(payload), 200
        except Exception as exc:  # pragma: no cover - exercised in deployed runtime
            return jsonify({"error": str(exc)}), 500

    @app.get("/health")
    def health():
        configured_bucket = bucket_name or os.getenv("BUCKET_NAME")
        resolved_output_dir = resolve_output_dir(output_dir)
        return jsonify(
            {
                "ready": True,
                "storage_mode": "gcs" if configured_bucket else "local",
                "bucket_name": configured_bucket,
                "output_dir": str(resolved_output_dir),
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
