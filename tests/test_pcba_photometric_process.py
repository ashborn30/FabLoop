"""Calibrated PCBA processing fails closed and recovers analytical normals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.fabloop.photometric_stereo.pcba_process import COORDINATE_FRAME, LIGHT_ORDER, process_capture

RPS_ROOT = Path(__file__).resolve().parents[1] / "third-party/RobustPhotometricStereo"


class PcbaPhotometricProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output, self.capture = self.root / "output", self.root / "capture.json"
        self.mask = np.zeros((10, 12), dtype=bool)
        self.mask[1:-1, 1:-1] = True
        cv2.imwrite(str(self.root / "mask.png"), self.mask.astype(np.uint8) * 255)
        self.normal = np.array([0.18, -0.12, 0.976])
        self.normal /= np.linalg.norm(self.normal)
        self.directions = np.array([[0, -.6, .8], [0, .6, .8], [-.6, 0, .8], [.6, 0, .8]])
        self.intensities = np.array([[.9, .8, .7], [.8, 1., .9], [1., .9, .8], [.7, .8, 1.]])
        self.config = {
            "schema_version": 1, "registration_verified": True, "calibrated": True,
            "coordinate_frame": COORDINATE_FRAME, "light_order": list(LIGHT_ORDER),
            "images": {key: f"{key}.png" for key in LIGHT_ORDER}, "mask": "mask.png",
            "light_directions": self.directions.tolist(), "light_intensities": self.intensities.tolist(),
            "image_linearity": "linear",
        }
        self.write_observations()

    def write_observations(self, srgb=False):
        for index, key in enumerate(LIGHT_ORDER):
            rgb = np.broadcast_to(
                np.dot(self.directions[index], self.normal) * np.array([.55, .4, .25]) * self.intensities[index],
                (*self.mask.shape, 3),
            ).copy()
            rgb[~self.mask] = 0
            if srgb:
                rgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
            encoded = np.rint(rgb * 65535).astype(np.uint16)
            self.assertTrue(cv2.imwrite(str(self.root / f"{key}.png"), encoded[:, :, ::-1]))

    def run_capture(self, config=None):
        self.capture.write_text(json.dumps(config or self.config), encoding="utf-8")
        return process_capture(self.capture, self.output, RPS_ROOT)

    def test_unverified_missing_or_invalid_metadata_blocks_before_solver_and_outputs(self):
        cases = [
            ("registration_verified", False, "registration_verified"),
            ("calibrated", None, "calibrated"),
            ("coordinate_frame", "unknown", "coordinate_frame"),
            ("image_linearity", None, "image_linearity"),
            ("light_order", ["B", "F", "L", "R"], "light_order"),
            ("light_directions", None, "4x3"),
            ("light_directions", [[0, 0, 1]] * 4, "rank 3"),
            ("light_intensities", [[0, 1, 1]] * 4, "strictly positive"),
            ("light_intensities", [[float("nan"), 1, 1]] * 4, "finite"),
            ("ambient_path", "ambient.jpg", "unsupported"),
        ]
        with patch("src.fabloop.photometric_stereo.pcba_process.import_rps_class") as load_solver:
            for key, value, message in cases:
                with self.subTest(key=key, value=value):
                    candidate = dict(self.config)
                    if value is None:
                        candidate.pop(key)
                    else:
                        candidate[key] = value
                    with self.assertRaisesRegex(ValueError, message):
                        self.run_capture(candidate)
                    self.assertFalse(self.output.exists())
            load_solver.assert_not_called()

    def test_mismatched_image_shape_and_empty_mask_block_without_resizing(self):
        cv2.imwrite(str(self.root / "B.png"), np.ones((5, 7, 3), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "match mask shape"):
            self.run_capture()
        self.assertFalse(self.output.exists())
        self.write_observations()
        cv2.imwrite(str(self.root / "mask.png"), np.zeros(self.mask.shape, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.run_capture()
        self.assertFalse(self.output.exists())

    @unittest.skipUnless((RPS_ROOT / "rps.py").is_file(), "Upstream RobustPhotometricStereo checkout unavailable")
    def test_real_rps_recovers_analytic_normals_with_rgb_calibration_and_provenance(self):
        report = self.run_capture()
        estimated = np.load(self.output / "normal_est.npy")
        expected = np.broadcast_to(self.normal, estimated[self.mask].shape)
        np.testing.assert_allclose(estimated[self.mask], expected, atol=5.0e-5)
        np.testing.assert_array_equal(estimated[~self.mask], 0)
        height = np.load(self.output / "relativeheight.npy")
        self.assertTrue(np.all(np.isfinite(height[self.mask])))
        self.assertTrue(np.all(np.isnan(height[~self.mask])))
        self.assertAlmostEqual(float(np.median(height[self.mask])), 0, places=12)
        self.assertEqual(report["ambient_correction"], "not_performed")
        self.assertFalse(report["relative_height"]["metric_measurement"])
        self.assertEqual(report["light_order"], list(LIGHT_ORDER))
        self.assertEqual(report["provenance"]["capture_config"]["sha256"], hashlib.sha256(self.capture.read_bytes()).hexdigest())
        self.assertEqual(report["outputs"]["normal_est.npy"]["sha256"], hashlib.sha256((self.output / "normal_est.npy").read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.output / "report.json").read_text())["status"], "EXPORTED")
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ["normal_est.npy", "normal_rgb.png", "relativeheight.npy", "report.json"])

    @unittest.skipUnless((RPS_ROOT / "rps.py").is_file(), "Upstream RobustPhotometricStereo checkout unavailable")
    def test_srgb_is_linearized_before_per_channel_intensity_correction(self):
        self.write_observations(srgb=True)
        self.config["image_linearity"] = "srgb"
        self.run_capture()
        estimated = np.load(self.output / "normal_est.npy")
        expected = np.broadcast_to(self.normal, estimated[self.mask].shape)
        np.testing.assert_allclose(estimated[self.mask], expected, atol=8.0e-5)


if __name__ == "__main__":
    unittest.main()
