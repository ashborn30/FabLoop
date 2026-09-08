from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fabloop.photometric_stereo import diligent_validate as ps


class PhotometricStereoSafetyTests(unittest.TestCase):
    def test_published_baseline_is_populated_and_protocol_is_required(self) -> None:
        rows = ps.load_baselines(ps.default_baseline_path())
        self.assertEqual(len(rows), 10)
        record = {"mae_deg": 4.2}
        ps.attach_baseline(record, rows, "ball", "l2", 96)
        self.assertEqual(record["baseline_status"], "reference_available")
        self.assertAlmostEqual(record["baseline_mae_deg"], 4.10)
        self.assertEqual(ps.require_matching_baseline(rows, "ball", "l2", 4), "not_applicable")
        with self.assertRaisesRegex(ValueError, "Missing published baseline"):
            ps.require_matching_baseline([], "ball", "l2", 96)
        with self.assertRaisesRegex(ValueError, "Missing published baseline"):
            ps.require_matching_baseline(rows, "ball", "l1", 4, required=True)

    def test_configured_baseline_missing_empty_and_nonfinite_fail(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.csv"
            with self.assertRaises(FileNotFoundError):
                ps.load_baselines(path)
            header = "object,solver,image_count,mae_deg,source_url\n"
            path.write_text(header, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                ps.load_baselines(path)
            path.write_text(header + "ball,l2,96,nan,https://example.com\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid baseline row"):
                ps.load_baselines(path)

    def test_subset_selection_needs_intent_and_unique_valid_indices(self) -> None:
        args = argparse.Namespace(indices=None, image_counts=["4"], start_index=0, selection=None)
        with self.assertRaisesRegex(ValueError, "Subset selection must be explicit"):
            ps.build_image_sets(args, 96)
        args.selection = "sequential"
        self.assertEqual(ps.build_image_sets(args, 96), [("lights_004", [0, 1, 2, 3])])
        args.selection = None
        args.image_counts = ["all"]
        self.assertEqual(len(ps.build_image_sets(args, 96)[0][1]), 96)
        for text in ("0,0,1,2", "0,1", "0,1,96", "-1,0,1"):
            with self.subTest(indices=text), self.assertRaises(ValueError):
                ps.parse_indices(text, 96)

    def test_degenerate_geometry_and_nonfinite_normals_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "degenerate"):
            ps.light_geometry(np.ones((4, 3)))
        with self.assertRaises(ValueError):
            ps.light_geometry(np.eye(3)[:2])
        lights = np.array([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]], dtype=float)
        geometry = ps.light_geometry(lights)
        self.assertEqual(geometry["rank"], 3)
        self.assertTrue(np.isfinite(geometry["condition_number"]))
        normal = np.zeros((2, 2, 3)); normal[..., 2] = 1
        invalid = normal.copy(); invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ps.mean_angular_error(invalid, normal, np.ones((2, 2), dtype=bool))

    def test_scoped_vendor_import_restores_system_psutil_and_avoids_stale_clone(self) -> None:
        system_psutil = importlib.import_module("psutil")
        original_path = list(sys.path)
        original_modules = {name: sys.modules.get(name) for name in ("rps", "rpsnumerics", "psutil")}
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "first", Path(folder) / "second"
            for root, marker in ((first, "FIRST"), (second, "SECOND")):
                root.mkdir()
                (root / "psutil.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")
                (root / "rpsnumerics.py").write_text("VALUE = 123\n", encoding="utf-8")
                (root / "rps.py").write_text(
                    "import psutil\nimport rpsnumerics\nclass RPS:\n"
                    "    marker = psutil.MARKER\n    value = rpsnumerics.VALUE\n", encoding="utf-8"
                )
            self.assertEqual(ps.import_rps_class(first).marker, "FIRST")
            self.assertEqual(ps.import_rps_class(second).marker, "SECOND")
            (second / "rps.py").write_text("raise RuntimeError('broken vendor import')\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "broken vendor import"):
                ps.import_rps_class(second)
        self.assertEqual(sys.path, original_path)
        for name, module in original_modules.items():
            self.assertIs(sys.modules.get(name), module)
        self.assertIs(importlib.import_module("psutil"), system_psutil)
        self.assertTrue(callable(system_psutil.Process))

    def test_baseline_failure_precedes_vendor_import_and_output_creation(self) -> None:
        sample = ps.DiligentSample(
            "ball", Path("unused"), [Path(f"{i}.png") for i in range(96)],
            np.tile(np.eye(3), (32, 1)), np.ones((96, 3)),
            np.ones((2, 2), dtype=bool), np.tile([0, 0, 1], (2, 2, 1)),
        )
        args = ps.parse_args(["--object-dir", "unused", "--image-counts", "all", "--solvers", "l2"])
        with patch.object(ps, "load_diligent_sample", return_value=sample), \
                patch.object(ps, "load_baselines", return_value=[]), \
                patch.object(ps, "import_rps_class") as importer, \
                patch.object(ps, "build_measurement_matrix") as image_loader:
            with self.assertRaisesRegex(ValueError, "Missing published baseline"):
                ps.run_validation(args)
        importer.assert_not_called()
        image_loader.assert_not_called()

    def test_height_conversion_respects_source_y_axis(self) -> None:
        rows, cols = 32, 48
        y, x = np.mgrid[:rows, :cols]
        surface = np.sin(2 * np.pi * y / rows) + np.cos(2 * np.pi * x / cols)
        dx = -(2 * np.pi / cols) * np.sin(2 * np.pi * x / cols)
        drow = (2 * np.pi / rows) * np.cos(2 * np.pi * y / rows)
        down_normals = np.stack([-dx, -drow, np.ones_like(dx)], axis=-1)
        up_normals = down_normals.copy(); up_normals[..., 1] *= -1
        mask = np.ones((rows, cols), dtype=bool)
        expected = surface - np.median(surface)
        np.testing.assert_allclose(ps.normals_to_height_frankot_chellappa(down_normals, mask), expected, atol=1e-12)
        np.testing.assert_allclose(
            ps.normals_to_height_frankot_chellappa(up_normals, mask, normal_y_axis="up"), expected, atol=1e-12
        )

    def test_real_vendor_l2_and_l1_recover_synthetic_normals_and_preserve_psutil(self) -> None:
        vendor_root = ROOT / "third-party" / "RobustPhotometricStereo"
        if not (vendor_root / "rps.py").is_file():
            self.skipTest("Optional external RPS checkout unavailable")
        system_psutil = importlib.import_module("psutil")
        RPS = ps.import_rps_class(vendor_root)
        lights = ps.normalize_normals(np.array([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]]))
        expected = ps.normalize_normals(np.array([[0.1, 0.2, 1.0], [-0.2, 0.1, 1.0]]))
        measurement = (expected * np.array([[2.0], [3.0]])) @ lights.T
        mask = np.ones((1, 2), dtype=bool)
        for solver in ("l2", "l1"):
            normal, _ = ps.solve_with_rps(RPS, measurement, lights, mask, solver)
            np.testing.assert_allclose(normal.reshape(2, 3), expected, atol=1e-8)
        self.assertIs(importlib.import_module("psutil"), system_psutil)


if __name__ == "__main__":
    unittest.main()
