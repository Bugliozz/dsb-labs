"""Flask app serving the trained Ames house price model."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

try:
    from .ames import (
        ARTIFACT_DIR,
        build_demo_input_frame,
        build_prediction_frame,
        load_artifacts,
        parse_demo_payload,
        predict_price,
    )
except ImportError:  # pragma: no cover - fallback for `python app.py`
    from ames import (  # type: ignore
        ARTIFACT_DIR,
        build_demo_input_frame,
        build_prediction_frame,
        load_artifacts,
        parse_demo_payload,
        predict_price,
    )


def create_app(artifact_dir: str | os.PathLike[str] | None = None) -> Flask:
    """Create the Flask inference service."""
    base_dir = Path(__file__).resolve().parent
    app = Flask(__name__, template_folder=str(base_dir / "templates"))
    artifacts = load_artifacts(Path(artifact_dir) if artifact_dir else ARTIFACT_DIR)

    @app.get("/")
    def index():
        metrics = artifacts["metrics"]
        model_info = artifacts["model_info"]
        schema = artifacts["input_schema"]
        defaults = artifacts["input_defaults"]
        demo_values = [
            defaults.get("LotArea", 10000),
            defaults.get("YearBuilt", 1975),
            defaults.get("GrLivArea", 1500),
            defaults.get("GarageCars", 2),
            defaults.get("OpenPorchSF", 50),
        ]
        return render_template(
            "index.html",
            metrics=metrics,
            model_info=model_info,
            schema=schema,
            defaults=defaults,
            demo_query=f"/api?data={demo_values}",
            advanced_example={
                "OverallQual": defaults.get("OverallQual"),
                "Neighborhood": defaults.get("Neighborhood"),
                "LotArea": defaults.get("LotArea"),
                "YearBuilt": defaults.get("YearBuilt"),
                "GrLivArea": defaults.get("GrLivArea"),
                "GarageCars": defaults.get("GarageCars"),
                "OpenPorchSF": defaults.get("OpenPorchSF"),
            },
        )

    @app.get("/api")
    def demo_predict():
        try:
            values = parse_demo_payload(request.args.get("data"))
            frame = build_demo_input_frame(values, artifacts["input_schema"], artifacts["input_defaults"])
            prediction = predict_price(artifacts["model"], frame)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(
            {
                "prediction": prediction,
                "model_name": artifacts["model_info"]["selected_model_name"],
                "input_mode": "demo",
            }
        )

    @app.post("/api/predict")
    def advanced_predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400
        try:
            frame = build_prediction_frame(payload, artifacts["input_schema"], artifacts["input_defaults"])
            prediction = predict_price(artifacts["model"], frame)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(
            {
                "prediction": prediction,
                "model_name": artifacts["model_info"]["selected_model_name"],
                "input_mode": "advanced",
            }
        )

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "ready": True,
                "model_name": artifacts["model_info"]["selected_model_name"],
                "training_timestamp": artifacts["model_info"]["training_timestamp"],
                "artifact_dir": artifacts["artifact_dir"],
            }
        )

    return app


def create_missing_artifacts_app(message: str) -> Flask:
    """Create a minimal app that reports missing artifacts."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        return (
            "<h1>Ames House Price API</h1>"
            "<p>Inference artifacts are missing. "
            "Run <code>python exercise_3/main.py train</code> first.</p>"
            f"<pre>{message}</pre>"
        ), 500

    @app.get("/api/health")
    def health():
        return jsonify({"ready": False, "error": message}), 503

    return app


try:
    app = create_app(os.getenv("AMES_ARTIFACT_DIR"))
except FileNotFoundError as exc:  # pragma: no cover - depends on local runtime state
    app = create_missing_artifacts_app(str(exc))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
