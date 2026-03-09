"""Tests for the Exercise 2 local runner."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from exercise_2 import main


class Exercise2RunnerTestCase(unittest.TestCase):
    """Validate the lightweight Exercise 2 runner."""

    @patch("exercise_2.main.run_selected_exercises")
    def test_cli_runs_selected_exercises(self, run_selected_exercises):
        main.main(["--exercise", "2_1", "2_4"])
        run_selected_exercises.assert_called_once_with(["2_1", "2_4"])

    @patch("exercise_2.main.run_selected_exercises")
    def test_cli_expands_all(self, run_selected_exercises):
        main.main(["--exercise", "all"])
        run_selected_exercises.assert_called_once_with(["2_1", "2_2", "2_3", "2_4"])

    @patch("exercise_2.main.run_selected_exercises")
    @patch("builtins.input", return_value="2, 4")
    def test_interactive_prompt_runs_selected_exercises(self, _input, run_selected_exercises):
        main.main([])
        run_selected_exercises.assert_called_once_with(["2_2", "2_4"])

    def test_cli_rejects_unknown_exercise(self):
        with self.assertRaises(SystemExit):
            main.main(["--exercise", "1_2"])
