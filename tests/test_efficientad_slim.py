"""Architecture/gradient regressions using synthetic CPU inputs, no training run."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from src.model import module_checksum
from src.models.efficientad_slim import SlimAutoEncoder, SlimEfficientAdModel, SlimStudent
from src.models.efficientad_slim import torch_model as baseline


ROOT = Path(__file__).resolve().parents[1]


class SlimCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_frozen_source_still_matches_manifest(self) -> None:
        directory = ROOT / "src/models/efficientad_slim"
        manifest = json.loads((directory / "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256((directory / "torch_model.py").read_bytes()).hexdigest(),
            manifest["sha256"],
        )

    def test_only_channel_widths_change(self) -> None:
        expected_channels = {
            "student": [(3, 64), (64, 128), (128, 128), (128, 768)],
            "ae.encoder": [(3, 16), (16, 16), (16, 32), (32, 32), (32, 32), (32, 32)],
            "ae.decoder": [(32, 32)] * 7 + [(32, 384)],
        }
        for padding in (False, True):
            with self.subTest(padding=padding):
                original = baseline.EfficientAdModel(padding=padding)
                slim = SlimEfficientAdModel(padding=padding)
                for component, expected in expected_channels.items():
                    layers = slim.get_submodule(component).modules()
                    actual = [(m.in_channels, m.out_channels) for m in layers if isinstance(m, nn.Conv2d)]
                    self.assertEqual(actual, expected)
                old_modules = dict(original.named_modules())
                new_modules = dict(slim.named_modules())
                self.assertEqual(list(old_modules), list(new_modules))
                for name, old in old_modules.items():
                    new = new_modules[name]
                    if isinstance(old, nn.Conv2d):
                        self.assertIsInstance(new, nn.Conv2d)
                        for attribute in ("kernel_size", "stride", "padding", "dilation", "groups", "padding_mode"):
                            self.assertEqual(getattr(new, attribute), getattr(old, attribute), (name, attribute))
                        self.assertEqual(new.bias is None, old.bias is None, name)
                        if name.startswith("teacher."):
                            self.assertEqual(new.weight.shape, old.weight.shape, name)
                    elif isinstance(old, (nn.AvgPool2d, nn.Dropout)):
                        self.assertEqual(type(new), type(old), name)
                        self.assertEqual(new.extra_repr(), old.extra_repr(), name)
                self.assertIs(SlimStudent.forward, baseline.SmallPatchDescriptionNetwork.forward)
                self.assertIs(SlimAutoEncoder.forward, baseline.AutoEncoder.forward)
                self.assertIs(type(slim.ae.encoder).forward, baseline.Encoder.forward)
                self.assertIs(type(slim.ae.decoder).forward, baseline.Decoder.forward)
                for method in (
                    "forward", "compute_student_teacher_distance", "compute_losses",
                    "compute_maps", "get_maps", "choose_random_aug_image",
                ):
                    self.assertIs(getattr(SlimEfficientAdModel, method), getattr(baseline.EfficientAdModel, method))

    def test_teacher_weights_and_training_modes_are_preserved(self) -> None:
        original = baseline.EfficientAdModel()
        torch.manual_seed(42)
        slim = SlimEfficientAdModel()
        self.assertEqual(module_checksum(original.teacher), module_checksum(slim.teacher))
        # Verified Teacher state can be loaded strictly without touching the slim networks.
        slim.teacher.load_state_dict(original.teacher.state_dict(), strict=True)
        self.assertTrue(all(not parameter.requires_grad for parameter in slim.teacher.parameters()))
        for mode in (True, False, True):
            self.assertIs(slim.train(mode), slim)
            self.assertFalse(slim.teacher.training)
            self.assertEqual(slim.student.training, mode)
            self.assertEqual(slim.ae.training, mode)
            self.assertTrue(all(m.training == mode for m in slim.ae.modules() if isinstance(m, nn.Dropout)))
        self.assertEqual(module_checksum(original.teacher), module_checksum(slim.teacher))

    def test_output_channels_and_spatial_geometry(self) -> None:
        for padding in (False, True):
            for image_size in ((256, 256), (256, 320)):
                with self.subTest(padding=padding, image_size=image_size), torch.no_grad():
                    model = SlimEfficientAdModel(padding=padding).eval()
                    sample = torch.rand(1, 3, *image_size)
                    teacher = model.teacher(sample)
                    student = model.student(sample)
                    ae = model.ae(sample, image_size)
                    expected_spatial = tuple(size // 4 - (0 if padding else 8) for size in image_size)
                    self.assertEqual(tuple(teacher.shape), (1, 384, *expected_spatial))
                    self.assertEqual(tuple(student.shape), (1, 768, *expected_spatial))
                    self.assertEqual(tuple(ae.shape), (1, 384, *expected_spatial))
                    for feature in (teacher, student, ae):
                        self.assertTrue(torch.isfinite(feature).all())

    def test_original_loss_backward_reaches_both_student_heads_and_ae(self) -> None:
        model = SlimEfficientAdModel().train()
        teacher_before = module_checksum(model.teacher)
        sample = torch.rand(1, 3, 256, 256)
        imagenet_sample = torch.rand_like(sample)
        losses = model(sample, imagenet_sample)
        self.assertEqual(len(losses), 3)
        for loss in losses:
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(loss.requires_grad)
        sum(losses).backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.teacher.parameters()))
        self.assertEqual(module_checksum(model.teacher), teacher_before)
        for gradient in (model.student.conv4.weight.grad[:384], model.student.conv4.weight.grad[384:]):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(torch.count_nonzero(gradient).item(), 0)
        for component in (model.student, model.ae):
            for name, parameter in component.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in component.parameters()), 0)
        # No optimizer step: this checks differentiability, not learned quality.

    def test_maps_match_original_core_and_state_roundtrip(self) -> None:
        for pad_maps in (False, True):
            with self.subTest(pad_maps=pad_maps), torch.no_grad():
                slim = SlimEfficientAdModel(pad_maps=pad_maps).eval()
                slim.mean_std["std"].fill_(1)
                slim.quantiles["qa_st"].fill_(0.1)
                slim.quantiles["qb_st"].fill_(1.1)
                slim.quantiles["qa_ae"].fill_(0.2)
                slim.quantiles["qb_ae"].fill_(1.2)
                reference = baseline.EfficientAdModel(pad_maps=pad_maps)
                reference.student = copy.deepcopy(slim.student)
                reference.ae = copy.deepcopy(slim.ae)
                reference.load_state_dict(slim.state_dict(), strict=True)
                reference.eval()
                sample = torch.rand(1, 3, 256, 256)
                for normalized in (False, True):
                    for actual, expected in zip(
                        slim.get_maps(sample, normalize=normalized),
                        reference.get_maps(sample, normalize=normalized), strict=True,
                    ):
                        self.assertEqual(tuple(actual.shape), (1, 1, 256, 256))
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                result = slim(sample)
                expected_result = reference(sample)
                torch.testing.assert_close(result.anomaly_map, expected_result.anomaly_map, rtol=0, atol=0)
                torch.testing.assert_close(result.pred_score, expected_result.pred_score, rtol=0, atol=0)
                saved = io.BytesIO()
                torch.save(slim.state_dict(), saved)
                saved.seek(0)
                restored = SlimEfficientAdModel(pad_maps=pad_maps).eval()
                restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
                torch.testing.assert_close(restored(sample).anomaly_map, result.anomaly_map, rtol=0, atol=0)
                self.assertEqual(module_checksum(restored.teacher), module_checksum(slim.teacher))

    def test_invalid_feature_contract_and_baseline_state_are_rejected(self) -> None:
        for constructor, kwargs in (
            (SlimStudent, {"out_channels": 384}),
            (SlimAutoEncoder, {"out_channels": 192}),
            (SlimEfficientAdModel, {"teacher_out_channels": 192}),
        ):
            with self.subTest(constructor=constructor.__name__), self.assertRaises(ValueError):
                constructor(**kwargs)
        with self.assertRaises(RuntimeError):
            SlimEfficientAdModel().load_state_dict(baseline.EfficientAdModel().state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
