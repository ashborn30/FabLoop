"""Pre-training feature contract regressions; synthetic CPU forwards only."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models.efficientad_slim import SlimEfficientAdModel
from src.models.efficientad_slim.shape_check import check_feature_shapes


class _FeatureModule(nn.Module):
    """Small stand-in that also exposes unintended dry-run side effects."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(0.5)
        self.register_buffer("calls", torch.zeros(()), persistent=False)
        self.nonfinite = False

    def forward(self, image: torch.Tensor, image_size=None) -> torch.Tensor:
        # Deliberate mutation verifies that the gate restores values, including
        # buffers absent from state_dict(), on both the success and error paths.
        with torch.no_grad():
            self.calls.add_(1)
            self.weight.add_(1)
        self.weight.requires_grad_(False)
        value = self.dropout(torch.rand((), device=image.device)) + self.weight
        if self.nonfinite:
            value = value * float("nan")
        return value.expand(self.shape)


class _FeatureCore(nn.Module):
    teacher_out_channels = 384

    def __init__(self, *, student_channels=768, ae_shape=(1, 384, 56, 56)) -> None:
        super().__init__()
        self.teacher = _FeatureModule((1, 384, 56, 56))
        self.student = _FeatureModule((1, student_channels, 56, 56))
        self.ae = _FeatureModule(ae_shape)


class FeatureShapeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_real_slim_candidate_has_two_heads_and_three_compatible_differences(self) -> None:
        model = SlimEfficientAdModel()
        report = check_feature_shapes(model)
        feature_shape = [1, 384, 56, 56]
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["input_shape"], [1, 3, 256, 256])
        self.assertEqual(
            report["features"],
            {
                "teacher": feature_shape,
                "student": [1, 768, 56, 56],
                "autoencoder": feature_shape,
                "student_teacher": feature_shape,
                "student_autoencoder": feature_shape,
            },
        )
        self.assertEqual(
            report["differences"],
            {name: feature_shape for name in ("T-S_T", "T-A", "A-S_A")},
        )
        self.assertTrue(report["finite"])
        self.assertEqual(report["device"], "cpu")
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.teacher.parameters()))

    def test_broadcastable_ae_outputs_are_rejected(self) -> None:
        for ae_shape in ((1, 384, 56, 1), (1, 1, 56, 56)):
            with self.subTest(ae_shape=ae_shape):
                # Both would silently pass a subtraction-only sanity check.
                self.assertEqual(torch.broadcast_shapes(ae_shape, (1, 384, 56, 56)), (1, 384, 56, 56))
                with self.assertRaisesRegex(RuntimeError, "autoencoder shape"):
                    check_feature_shapes(_FeatureCore(ae_shape=ae_shape))

    def test_student_without_second_head_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Student shape"):
            check_feature_shapes(_FeatureCore(student_channels=384))

    def test_nonfinite_features_are_rejected(self) -> None:
        model = _FeatureCore()
        model.ae.nonfinite = True
        with self.assertRaisesRegex(RuntimeError, "autoencoder contains non-finite"):
            check_feature_shapes(model)

    def test_modes_values_rng_and_existing_gradients_survive_success_and_failure(self) -> None:
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                model = _FeatureCore(ae_shape=(1, 384, 56, 1) if invalid else (1, 384, 56, 56))
                model.train()
                model.teacher.eval()
                model.student.dropout.eval()
                model.ae.eval()
                model.teacher.weight.requires_grad_(False)
                for index, parameter in enumerate(model.parameters()):
                    parameter.grad = torch.full_like(parameter, index + 0.25)
                modes = [(module, module.training) for module in model.modules()]
                values = [(tensor, tensor.detach().clone()) for tensor in (*model.parameters(), *model.buffers())]
                flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
                gradients = [(parameter, parameter.grad, parameter.grad.clone()) for parameter in model.parameters()]
                rng = torch.get_rng_state().clone()

                if invalid:
                    with self.assertRaises(RuntimeError):
                        check_feature_shapes(model)
                else:
                    self.assertEqual(check_feature_shapes(model)["status"], "PASS")

                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                for module, expected in modes:
                    self.assertEqual(module.training, expected)
                for tensor, expected in values:
                    torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
                for parameter, expected in flags:
                    self.assertEqual(parameter.requires_grad, expected)
                for parameter, original, expected in gradients:
                    self.assertIs(parameter.grad, original)
                    torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
