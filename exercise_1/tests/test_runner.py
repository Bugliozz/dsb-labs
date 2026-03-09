"""Tests for the Exercise 1 local runner."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from exercise_1 import main


class Exercise1RunnerTestCase(unittest.TestCase):
    """Validate the lightweight Exercise 1 runner."""

    @patch("exercise_1.main.run_selected_exercises")
    def test_cli_runs_selected_exercises(self, run_selected_exercises):
        main.main(["--exercise", "1_2", "1_5"])
        run_selected_exercises.assert_called_once_with(["1_2", "1_5"])

    @patch("exercise_1.main.run_selected_exercises")
    def test_cli_expands_all(self, run_selected_exercises):
        main.main(["--exercise", "all"])
        run_selected_exercises.assert_called_once_with(["1_2", "1_3", "1_4", "1_5", "1_6", "1_7"])

    @patch("exercise_1.main.run_selected_exercises")
    @patch("builtins.input", return_value="1_3, 1_4")
    def test_interactive_prompt_runs_selected_exercises(self, _input, run_selected_exercises):
        main.main([])
        run_selected_exercises.assert_called_once_with(["1_3", "1_4"])

    def test_cli_rejects_unknown_exercise(self):
        with self.assertRaises(SystemExit):
            main.main(["--exercise", "2_1"])
