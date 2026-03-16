"""Tests for the Exercise 4 Malpensa collector."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_exercise_4_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "exercise_4" / "app.py"
    spec = importlib.util.spec_from_file_location("exercise_4_app", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Exercise4AppTestCase(unittest.TestCase):
    """Validate the local and GCS-facing app wiring."""

    @classmethod
    def setUpClass(cls):
        cls.exercise_4_app = _load_exercise_4_module()

    def test_collects_into_local_directory_when_bucket_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = self.exercise_4_app.create_app(output_dir=tmpdir)
            client = app.test_client()

            with patch.dict("os.environ", {"BUCKET_NAME": ""}, clear=False):
                with patch.object(
                    self.exercise_4_app,
                    "get_flights",
                    side_effect=[{"movement": "D"}, {"movement": "A"}],
                ):
                    response = client.get("/?date=2026-03-14")

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["storage_mode"], "local")
            self.assertEqual(len(payload["files"]), 2)
            self.assertTrue((Path(tmpdir) / "2026-03-14_D.json").is_file())
            self.assertTrue((Path(tmpdir) / "2026-03-14_A.json").is_file())

    def test_health_reports_bucket_mode(self):
        app = self.exercise_4_app.create_app(bucket_name="demo-bucket", output_dir="local-backup")
        client = app.test_client()

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["storage_mode"], "gcs")
        self.assertEqual(payload["bucket_name"], "demo-bucket")
