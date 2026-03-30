"""Tests for the Exercise 5 local runner."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from exercise_5 import main


class Exercise5RunnerTestCase(unittest.TestCase):
    """Validate the lightweight Exercise 5 runner."""

    @patch("exercise_5.main.run_selected_exercises")
    def test_cli_runs_selected_exercises(self, run_selected_exercises):
        main.main(["--exercise", "5_1", "5_2"])
        run_selected_exercises.assert_called_once_with(["5_1", "5_2"])

    @patch("exercise_5.main.run_selected_exercises")
    def test_cli_expands_all(self, run_selected_exercises):
        main.main(["--exercise", "all"])
        run_selected_exercises.assert_called_once_with(["5_1", "5_2"])

    @patch("exercise_5.main.run_selected_exercises")
    @patch("builtins.input", return_value="1, 2")
    def test_interactive_prompt_runs_selected_exercises(self, _input, run_selected_exercises):
        main.main([])
        run_selected_exercises.assert_called_once_with(["5_1", "5_2"])

    def test_cli_rejects_unknown_exercise(self):
        with self.assertRaises(SystemExit):
            main.main(["--exercise", "2_1"])
