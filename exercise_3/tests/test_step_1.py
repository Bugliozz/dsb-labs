"""Tests for the Exercise 3.1 Ames API."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from exercise_3.step_1.app import create_app
from exercise_3.step_1.workflow import run_training_workflow


class DummyModel:
    """Minimal model used to isolate API tests from offline training."""

    def predict(self, frame):
        base = float(frame.iloc[0]["LotArea"])
        return np.array([base + 100000.0])


class AmesApiTestCase(unittest.TestCase):
    """Validate the Flask inference service behaviour."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self.tempdir.name)
        schema = {
            "feature_order": [
                "LotArea",
                "YearBuilt",
                "GrLivArea",
                "GarageCars",
                "OpenPorchSF",
                "Neighborhood",
            ],
            "numeric_features": [
                "LotArea",
                "YearBuilt",
                "GrLivArea",
                "GarageCars",
                "OpenPorchSF",
            ],
            "categorical_features": ["Neighborhood"],
            "demo_feature_order": [
                "LotArea",
                "YearBuilt",
                "GrLivArea",
                "GarageCars",
                "OpenPorchSF",
            ],
            "engineered_features": ["HouseAge", "RemodelAge", "TotalBath", "TotalSF"],
            "fields": {
                "LotArea": {"type": "number"},
                "YearBuilt": {"type": "number"},
                "GrLivArea": {"type": "number"},
                "GarageCars": {"type": "number"},
                "OpenPorchSF": {"type": "number"},
                "Neighborhood": {
                    "type": "categorical",
                    "allowed_values": ["NAmes", "CollgCr"],
                },
            },
        }
        defaults = {
            "LotArea": 9600.0,
            "YearBuilt": 1975.0,
            "GrLivArea": 1500.0,
            "GarageCars": 2.0,
            "OpenPorchSF": 50.0,
            "Neighborhood": "NAmes",
        }
        metrics = {
            "selected_model": {
                "name": "DummyRegressor",
                "cv_rmse": 10000.0,
                "holdout": {
                    "mse": 1.0,
                    "rmse": 1.0,
                    "mae": 1.0,
                    "r2": 0.99,
                },
            }
        }
        model_info = {
            "selected_model_name": "DummyRegressor",
            "training_timestamp": "2026-03-09T00:00:00+00:00",
        }

        joblib.dump(DummyModel(), self.artifact_dir / "best_model.joblib")
        (self.artifact_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (self.artifact_dir / "model_info.json").write_text(json.dumps(model_info), encoding="utf-8")
        (self.artifact_dir / "input_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        (self.artifact_dir / "input_defaults.json").write_text(json.dumps(defaults), encoding="utf-8")

        self.app = create_app(self.artifact_dir)
        self.client = self.app.test_client()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_index_returns_ok(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Ames House Price API", response.data)

    def test_health_returns_ready(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["model_name"], "DummyRegressor")

    def test_demo_prediction_returns_numeric_value(self):
        response = self.client.get("/api?data=[10603,1977,1610,2,68]")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["input_mode"], "demo")
        self.assertIsInstance(payload["prediction"], float)

    def test_advanced_prediction_returns_numeric_value(self):
        response = self.client.post(
            "/api/predict",
            json={
                "LotArea": 9000,
                "YearBuilt": 1980,
                "GrLivArea": 1600,
                "GarageCars": 2,
                "OpenPorchSF": 45,
                "Neighborhood": "CollgCr",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["input_mode"], "advanced")
        self.assertGreater(payload["prediction"], 100000.0)

    def test_demo_rejects_wrong_length(self):
        response = self.client.get("/api?data=[1,2,3]")
        self.assertEqual(response.status_code, 400)
        self.assertIn("exactly 5", response.get_json()["error"])

    def test_advanced_rejects_unknown_fields(self):
        response = self.client.post("/api/predict", json={"UnknownField": 1})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown fields", response.get_json()["error"])

    def test_advanced_rejects_invalid_types(self):
        response = self.client.post("/api/predict", json={"LotArea": "not-a-number"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("LotArea", response.get_json()["error"])


class AmesTrainingWorkflowTestCase(unittest.TestCase):
    """Smoke test the offline workflow when the dataset is locally available."""

    def test_training_workflow_if_dataset_exists(self):
        data_dir = Path("exercise_3/step_1/data")
        if not (data_dir / "train.csv").is_file():
            self.skipTest("Ames dataset not available in exercise_3/step_1/data")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            result = run_training_workflow(
                data_dir=data_dir,
                image_dir=output_root / "images",
                artifact_dir=output_root / "artifacts",
            )

            self.assertTrue((output_root / "artifacts" / "best_model.joblib").is_file())
            self.assertTrue((output_root / "artifacts" / "metrics.json").is_file())
            self.assertTrue((output_root / "artifacts" / "model_info.json").is_file())
            self.assertTrue((output_root / "artifacts" / "input_schema.json").is_file())
            self.assertTrue((output_root / "artifacts" / "input_defaults.json").is_file())
            self.assertTrue((output_root / "artifacts" / "mutual_information.csv").is_file())
            self.assertEqual(
                result["metrics"]["selected_model"]["name"],
                result["model_info"]["selected_model_name"],
            )
