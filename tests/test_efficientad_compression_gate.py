"""Check Student cost thresholds and durable CLI decisions without profiling or training."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import profile_efficientad_slim as profiler


class EfficientAdCompressionGateTests(unittest.TestCase):
    @staticmethod
    def profile_record(student_params: int, student_flops: int, ae_cost: int) -> dict:
        def module(parameters: int, flops: int, channels: int) -> dict:
            return {
                "parameters": parameters,
                "supported_flops": flops,
                "convolution_flops": flops,
                "output_shape": [1, channels, 56, 56],
            }

        modules = {
            "teacher": module(100, 100, 384),
            "student": module(student_params, student_flops, 768),
            "autoencoder": module(ae_cost, ae_cost, 384),
        }
        return {
            "execution": {"device": "cpu", "input_shape": [1, 3, 256, 256]},
            "counting_policy": {"multiply_add_is_one_flop": True},
            "modules": modules,
            "deployment_core_teacher_student_autoencoder": {
                metric: sum(record[metric] for record in modules.values())
                for metric in ("parameters", "supported_flops", "convolution_flops")
            },
            "source": {"source_manifest": "src/models/efficientad_slim/SOURCE.json"},
        }

    def compare_costs(self, student_params: int, student_flops: int) -> dict:
        # AE dominates this synthetic pipeline: its large reduction must never
        # rescue a Student that misses either of the independent targets.
        baseline = self.profile_record(1000, 1000, 10000)
        candidate = self.profile_record(student_params, student_flops, 100)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("seed: 42\ndata:\n  image_size: [256, 256]\n", encoding="utf-8")
            with patch.object(profiler, "profile", side_effect=[baseline, candidate]) as profile:
                result = profiler.compare(config_path)
            self.assertEqual(profile.call_count, 2)
            baseline_sample = profile.call_args_list[0].kwargs["sample"]
            candidate_sample = profile.call_args_list[1].kwargs["sample"]
            self.assertIs(baseline_sample, candidate_sample)
            self.assertEqual(list(baseline_sample.shape), [1, 3, 256, 256])
        return result

    def assert_pipeline_exceeds_student_targets(self, result: dict) -> None:
        pipeline = result["reductions"]["deployment_core_teacher_student_autoencoder"]
        self.assertGreater(pipeline["parameters"]["reduction_fraction"], 0.40)
        self.assertGreater(pipeline["supported_flops"]["reduction_fraction"], 0.30)

    def test_student_passes_at_both_inclusive_thresholds(self) -> None:
        result = self.compare_costs(student_params=600, student_flops=700)
        student = result["reductions"]["student_only"]
        self.assertEqual(student["parameters"]["reduction_fraction"], 0.40)
        self.assertEqual(student["supported_flops"]["reduction_fraction"], 0.30)
        gate = result["student_compression_gate"]
        self.assertTrue(gate["parameters_passed"])
        self.assertTrue(gate["supported_flops_passed"])
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["decision"], "PROCEED_TO_TRAINING_INTEGRATION")

    def test_student_params_below_target_fail_despite_pipeline_reduction(self) -> None:
        result = self.compare_costs(student_params=601, student_flops=700)
        self.assert_pipeline_exceeds_student_targets(result)
        gate = result["student_compression_gate"]
        self.assertFalse(gate["parameters_passed"])
        self.assertTrue(gate["supported_flops_passed"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["status"], "FAIL")
        self.assertEqual(gate["decision"], "REVISE_ARCHITECTURE_BEFORE_TRAINING")

    def test_student_flops_below_target_fail_despite_pipeline_reduction(self) -> None:
        result = self.compare_costs(student_params=600, student_flops=701)
        self.assert_pipeline_exceeds_student_targets(result)
        gate = result["student_compression_gate"]
        self.assertTrue(gate["parameters_passed"])
        self.assertFalse(gate["supported_flops_passed"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["status"], "FAIL")
        self.assertEqual(gate["decision"], "REVISE_ARCHITECTURE_BEFORE_TRAINING")

    def run_cli_with_result(self, result: dict, expected_exit: int) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "cost_report.json"
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("data:\n  image_size: [256, 256]\n", encoding="utf-8")
            with patch.object(profiler, "compare", return_value=result) as compare, \
                    redirect_stdout(io.StringIO()):
                exit_code = profiler.main(["--config", str(config_path), "--output", str(output_path)])
            compare.assert_called_once_with(config_path.resolve())
            self.assertEqual(exit_code, expected_exit)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), result)

    def test_failed_cli_gate_persists_report_and_returns_nonzero(self) -> None:
        self.run_cli_with_result(self.compare_costs(601, 700), expected_exit=1)

    def test_passed_cli_gate_persists_report_and_returns_zero(self) -> None:
        self.run_cli_with_result(self.compare_costs(600, 700), expected_exit=0)


if __name__ == "__main__":
    unittest.main()
