"""CLI scope, partial failures and explicit nominal-light selection."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_pcba_preview as cli


class PreviewCliTests(unittest.TestCase):
    def test_nominal_assumptions_require_explicit_flag(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.main([])
        self.assertEqual(error.exception.code, 2)

    def test_partial_failure_is_recorded_without_losing_successful_board(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "captures", root / "previews"
            for name in ("PCB2", "PCB1"):
                (source / name).mkdir(parents=True)
            with patch.object(cli, "export_preview", side_effect=[{"status": "QUALITATIVE_PREVIEW"}, ValueError("insufficient matches")]) as export:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = cli.main(["--input-root", str(source), "--output-root", str(output), "--nominal-lights"])
            self.assertEqual(code, 1)
            self.assertEqual([call.args[0].name for call in export.call_args_list], ["PCB1", "PCB2"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["counts"], {"requested": 2, "exported": 1, "failed": 1})
            self.assertFalse(summary["calibrated"])
            self.assertFalse(summary["training_ready"])
            self.assertEqual(summary["boards"][1]["error"], "insufficient matches")

    def test_source_destinations_existing_runs_and_board_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "captures", root / "previews"
            (source / "PCB1").mkdir(parents=True)
            output.mkdir()
            summary_path = output / "summary.json"
            summary_path.write_text("original")
            cases = [
                ["--output-root", str(source / "derived")],
                ["--output-root", str(output)],
                ["--output-root", str(root / "new"), "--boards", "../PCB1"],
            ]
            for args in cases:
                with self.subTest(args=args), patch.object(cli, "export_preview") as export:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self.assertEqual(cli.main(["--input-root", str(source), "--nominal-lights", *args]), 1)
                    export.assert_not_called()
            self.assertEqual(summary_path.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
