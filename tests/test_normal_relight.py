from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fabloop.photometric_stereo.normal_relight import (
    discover_relight_cases,
    load_albedo,
    light_direction,
    load_relight_inputs,
    render_relight,
)


class NormalRelightTest(unittest.TestCase):
    def test_light_direction_uses_camera_axis_at_ninety_degree_elevation(self) -> None:
        np.testing.assert_allclose(light_direction(123.0, 90.0), [0.0, 0.0, 1.0], atol=1.0e-6)

    def test_render_relight_implements_normal_dot_light_times_albedo(self) -> None:
        normals = np.zeros((2, 2, 3), dtype=np.float32)
        normals[:, :, 2] = 1.0
        albedo = np.full((2, 2), 0.5, dtype=np.float32)

        lit = render_relight(normals, albedo, azimuth_deg=0.0, elevation_deg=90.0)
        grazing = render_relight(normals, albedo, azimuth_deg=0.0, elevation_deg=0.0)

        self.assertTrue(np.all(lit == 127))
        self.assertTrue(np.all(grazing == 0))

    def test_discover_and_load_pcba_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pcb10 = root / "PCB10"
            pcb2 = root / "PCB2"
            pcb2.mkdir()
            pcb10.mkdir()
            normals = np.zeros((3, 4, 3), dtype=np.float32)
            normals[:, :, 2] = 1.0
            albedo = np.ones((3, 4), dtype=np.float32)
            for folder in (pcb10, pcb2):
                np.save(folder / "normal_l2_nominal.npy", normals)
                np.save(folder / "albedo_l2_nominal.npy", albedo)

            cases = discover_relight_cases(root)

            self.assertEqual([case.name for case in cases], ["PCB2", "PCB10"])
            loaded = load_relight_inputs(cases[0])
            self.assertEqual(loaded.normals.shape, (3, 4, 3))
            self.assertEqual(loaded.albedo.shape, (3, 4))

    def test_float_npy_albedo_is_not_treated_as_uint8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "albedo.npy"
            np.save(path, np.array([[0.25, 1.2]], dtype=np.float32))

            albedo = load_albedo(path)

            np.testing.assert_allclose(albedo, [[0.25, 1.0]], atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
