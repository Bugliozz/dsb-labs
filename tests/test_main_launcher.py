"""Tests for the root launcher."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main as root_main


class RootLauncherTestCase(unittest.TestCase):
    """Validate root launcher parsing and dispatch."""

    @patch("main.subprocess.run")
    def test_noninteractive_mixed_selection_dispatches_local_runners(self, subprocess_run):
        subprocess_run.return_value = Mock(returncode=0)

        root_main.main(["--exercise", "1_2", "2_4", "3_1", "--command", "train"])

        self.assertEqual(subprocess_run.call_count, 3)
        commands = [call.args[0] for call in subprocess_run.call_args_list]
        self.assertEqual(Path(commands[0][1]).parent.name, "exercise_1")
        self.assertEqual(commands[0][-2:], ["--exercise", "1_2"])
        self.assertEqual(Path(commands[1][1]).parent.name, "exercise_2")
        self.assertEqual(commands[1][-2:], ["--exercise", "2_4"])
        self.assertEqual(Path(commands[2][1]).parent.name, "exercise_3")
        self.assertEqual(commands[2][2:5], ["train", "--exercise", "3_1"])

    @patch("main.subprocess.run")
    def test_chapter_all_dispatches_single_runner(self, subprocess_run):
        subprocess_run.return_value = Mock(returncode=0)

        root_main.main(["--chapter", "2", "--exercise", "all"])

        subprocess_run.assert_called_once()
        command = subprocess_run.call_args.args[0]
        self.assertEqual(Path(command[1]).parent.name, "exercise_2")
        self.assertEqual(command[2], "--exercise")
        self.assertEqual(command[3:], ["2_1", "2_2", "2_3", "2_4"])

    @patch("main.subprocess.run")
    @patch("builtins.input", side_effect=["3", "1", "serve"])
    def test_interactive_exercise_three_prompts_for_command(self, _input, subprocess_run):
        subprocess_run.return_value = Mock(returncode=0)

        root_main.main([])

        subprocess_run.assert_called_once()
        command = subprocess_run.call_args.args[0]
        self.assertEqual(Path(command[1]).parent.name, "exercise_3")
        self.assertEqual(command[2:5], ["serve", "--exercise", "3_1"])

    @patch("main.subprocess.run")
    def test_all_without_chapter_is_rejected(self, subprocess_run):
        with self.assertRaises(SystemExit):
            root_main.main(["--exercise", "all"])
        subprocess_run.assert_not_called()

    @patch("main.subprocess.run")
    def test_noninteractive_exercise_four_dispatches_laboratory_runner(self, subprocess_run):
        subprocess_run.return_value = Mock(returncode=0)

        root_main.main(["--exercise", "4_1"])

        subprocess_run.assert_called_once()
        command = subprocess_run.call_args.args[0]
        self.assertEqual(Path(command[1]).parent.name, "exercise_4")
        self.assertEqual(command[-2:], ["--exercise", "4_1"])
