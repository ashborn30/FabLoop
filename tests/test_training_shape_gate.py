"""Prove that a failed shape gate blocks expensive training preparation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import trainer


class TrainingShapeGateTests(unittest.TestCase):
    def config(self, output: str) -> dict:
        return {
            "dataset": {"name": "Visa"},
            "training": {"batch_size": 1, "optimizer": "Adam"},
            "data": {"image_size": [256, 256]},
            "output_dir": output,
        }

    def test_shape_failure_blocks_teacher_download_data_preparation_and_optimizer(self) -> None:
        for error in (RuntimeError("Student shape mismatch"), ValueError("Invalid image size")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                wrapper = MagicMock()
                wrapper.verify_feature_shapes.side_effect = error
                with patch.object(trainer, "EfficientAdWrapper", return_value=wrapper), \
                        patch.object(trainer, "environment_info", return_value={}), \
                        patch.object(trainer, "seed_everything"), \
                        patch.object(trainer.torch.optim, "Adam") as optimizer:
                    with self.assertRaises(type(error)):
                        trainer.train(self.config(directory), "pcb1", 42, torch.device("cpu"), None, {})
                    wrapper.verify_feature_shapes.assert_called_once_with((256, 256))
                    wrapper.load_pretrained_teacher.assert_not_called()
                    wrapper.lightning_model.prepare_imagenette_data.assert_not_called()
                    wrapper.lightning_model.teacher_channel_mean_std.assert_not_called()
                    wrapper.trainable_parameters.assert_not_called()
                    optimizer.assert_not_called()
                report = json.loads((Path(directory) / "pcb1/checkpoints/shape_check_seed_42.json").read_text())
                self.assertEqual(report, {"status": "FAIL", "error": str(error)})

    def test_success_is_recorded_before_teacher_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrapper = MagicMock()
            report = {"status": "PASS", "input_shape": [1, 3, 256, 256]}
            wrapper.verify_feature_shapes.return_value = report
            wrapper.load_pretrained_teacher.side_effect = RuntimeError("Stop before any download")
            events = MagicMock()
            events.attach_mock(wrapper.verify_feature_shapes, "shape_gate")
            events.attach_mock(wrapper.load_pretrained_teacher, "teacher_loader")
            with patch.object(trainer, "EfficientAdWrapper", return_value=wrapper), \
                    patch.object(trainer, "environment_info", return_value={}), \
                    patch.object(trainer, "seed_everything"), \
                    patch.object(trainer.torch.optim, "Adam") as optimizer:
                with self.assertRaisesRegex(RuntimeError, "Stop before any download"):
                    trainer.train(self.config(directory), "pcb1", 42, torch.device("cpu"), None, {})
                optimizer.assert_not_called()
            self.assertEqual(events.mock_calls, [call.shape_gate((256, 256)), call.teacher_loader()])
            actual = json.loads((Path(directory) / "pcb1/checkpoints/shape_check_seed_42.json").read_text())
            self.assertEqual(actual, report)


if __name__ == "__main__":
    unittest.main()
