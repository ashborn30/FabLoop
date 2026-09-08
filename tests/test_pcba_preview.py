"""Preview exports remain qualitative, preserve sources, and respect numerical masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.fabloop.photometric_stereo.pcba_preview import LIGHT_ORDER, NOMINAL_LIGHT_DIRECTIONS, export_preview

MODULE = "src.fabloop.photometric_stereo.pcba_preview"
RPS_ROOT = Path(__file__).resolve().parents[1] / "third-party/RobustPhotometricStereo"


class PcbaPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.board = self.root / "PCBA_4Light_edited" / "PCB1"
        self.board.mkdir(parents=True)
        self.output = self.root / "preview" / "PCB1"
        self.normal = np.array([.18, -.12, .976])
        self.normal /= np.linalg.norm(self.normal)
        self.common = np.zeros((12, 16), dtype=bool)
        self.common[1:-1, 1:-1] = True
        self.images, sources = {}, {}
        for key, light in zip(LIGHT_ORDER, NOMINAL_LIGHT_DIRECTIONS):
            signal = .45 * np.dot(light, self.normal)
            srgb = 12.92 * signal if signal <= .0031308 else 1.055 * signal ** (1 / 2.4) - .055
            image = np.full((*self.common.shape, 3), round(srgb * 255), dtype=np.uint8)
            image[2, 2] = 0
            path = self.board / f"light_{key}.png"
            cv2.imwrite(str(path), image[:, :, ::-1])
            sources[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.images[key] = image
        self.metadata = {"stage": "qualitative_preview_alignment", "reference_light": "F", "sources": sources}
        self.source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.board.iterdir()}

    def alignment(self):
        return self.images, self.common, self.metadata

    @unittest.skipUnless((RPS_ROOT / "rps.py").is_file(), "Upstream RPS checkout unavailable")
    def test_real_rps_recovers_synthetic_normal_and_exports_explicit_preview_provenance(self):
        before = {key: value.copy() for key, value in self.images.items()}
        with patch(f"{MODULE}.align_lights", return_value=self.alignment()):
            report = export_preview(self.board, self.output, max_edge=128)
        valid = cv2.imread(str(self.output / "valid_mask.png"), cv2.IMREAD_GRAYSCALE) != 0
        expected_mask = self.common.copy()
        expected_mask[2, 2] = False
        np.testing.assert_array_equal(valid, expected_mask)
        normals = np.load(self.output / "normal_est.npy")
        np.testing.assert_allclose(normals[valid], np.broadcast_to(self.normal, normals[valid].shape), atol=.004)
        np.testing.assert_array_equal(normals[~valid], 0)
        height = np.load(self.output / "relativeheight.npy")
        self.assertTrue(np.isfinite(height[valid]).all())
        self.assertTrue(np.isnan(height[~valid]).all())
        self.assertEqual(report["status"], "QUALITATIVE_PREVIEW")
        for flag in ("calibrated", "registration_verified", "training_ready", "normal_gt_available", "board_mask_available"):
            self.assertIs(report[flag], False)
        self.assertEqual(report["numerical_valid_mask"]["low_signal_excluded"], 1)
        self.assertEqual(report["provenance"]["source_images"], self.metadata["sources"])
        self.assertNotIn("mae", json.dumps(report).lower())
        self.assertFalse(report["display"]["raw_height_rescaled"])
        self.assertLessEqual(max(report["display"]["surface_grid_shape"]), 150)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), sorted([
            "normal_est.npy", "normal_rgb.png", "relativeheight.npy", "height_preview.png",
            "pseudo3d.png", "valid_mask.png", "reference_rgb.png", "alignment_preview.png", "report.json",
        ]))
        for path in self.board.iterdir():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.source_hashes[path.name])
        for key in LIGHT_ORDER:
            np.testing.assert_array_equal(self.images[key], before[key])
        for name in ("height_preview.png", "pseudo3d.png", "normal_rgb.png"):
            self.assertIsNotNone(cv2.imread(str(self.output / name)))

    @unittest.skipUnless((RPS_ROOT / "rps.py").is_file(), "Upstream RPS checkout unavailable")
    def test_zero_and_near_grazing_normals_are_masked_before_height_integration(self):
        normals = np.broadcast_to(np.array([0., 0., 1.]), (*self.common.shape, 3)).copy()
        normals[3, 3] = 0
        normals[4, 4] = [np.sqrt(1 - .05 ** 2), 0, .05]
        with patch(f"{MODULE}.align_lights", return_value=self.alignment()), \
             patch(f"{MODULE}.solve_with_rps", return_value=(normals, 0.0)):
            report = export_preview(self.board, self.output)
        height = np.load(self.output / "relativeheight.npy")
        self.assertTrue(np.isnan(height[3, 3]))
        self.assertTrue(np.isnan(height[4, 4]))
        self.assertEqual(report["numerical_valid_mask"]["invalid_normal_excluded"], 1)
        self.assertEqual(report["numerical_valid_mask"]["unstable_z_excluded"], 1)

    def test_source_tree_and_nonempty_outputs_are_rejected_before_alignment(self):
        for output in (self.board, self.board / "preview", self.board.parent / "other_preview"):
            with self.subTest(output=output), patch(f"{MODULE}.align_lights") as align:
                with self.assertRaisesRegex(ValueError, "outside the source"):
                    export_preview(self.board, output)
                align.assert_not_called()
        self.output.mkdir(parents=True)
        existing = self.output / "normal_est.npy"
        existing.write_bytes(b"existing user data")
        with patch(f"{MODULE}.align_lights") as align:
            with self.assertRaisesRegex(ValueError, "new or empty"):
                export_preview(self.board, self.output)
            align.assert_not_called()
        self.assertEqual(existing.read_bytes(), b"existing user data")

    def test_empty_signal_fails_without_creating_output(self):
        images = {key: np.zeros_like(value) for key, value in self.images.items()}
        with patch(f"{MODULE}.align_lights", return_value=(images, self.common, self.metadata)):
            with self.assertRaisesRegex(ValueError, "enough signal"):
                export_preview(self.board, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
