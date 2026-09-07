from __future__ import annotations

import unittest

import numpy as np

from fabloop.photometric_stereo.diligent_validate import (
    coerce_n_by_3_vectors,
    frankot_chellappa,
    mean_angular_error,
    normalize_object_name,
)


class PhotometricStereoUtilityTests(unittest.TestCase):
    def test_coerce_light_matrix_from_3_by_n(self) -> None:
        values = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )

        vectors = coerce_n_by_3_vectors(values, name="light_directions", expected_count=2)

        self.assertEqual(vectors.shape, (2, 3))
        np.testing.assert_allclose(vectors[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(vectors[1], [0.0, 1.0, 0.0])

    def test_mean_angular_error_uses_only_mask(self) -> None:
        estimated = np.array(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float64,
        )
        ground_truth = np.array(
            [
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        mask = np.array([[True, False], [True, False]])

        self.assertAlmostEqual(mean_angular_error(estimated, ground_truth, mask), 0.0)

    def test_frankot_chellappa_recovers_periodic_surface(self) -> None:
        rows, cols = 48, 64
        x = np.arange(cols, dtype=np.float64)
        y = np.arange(rows, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(x, y)
        surface = np.sin(2.0 * np.pi * grid_x / cols) + 0.5 * np.cos(4.0 * np.pi * grid_y / rows)
        dz_dx = (2.0 * np.pi / cols) * np.cos(2.0 * np.pi * grid_x / cols)
        dz_dy = -(2.0 * np.pi / rows) * np.sin(4.0 * np.pi * grid_y / rows)

        reconstructed = frankot_chellappa(dz_dx, dz_dy)

        expected = surface - surface.mean()
        reconstructed = reconstructed - reconstructed.mean()
        np.testing.assert_allclose(reconstructed, expected, atol=1.0e-2)

    def test_normalize_object_name_handles_diligent_folder_suffix(self) -> None:
        self.assertEqual(normalize_object_name("BallPNG"), "ball")
        self.assertEqual(normalize_object_name("diligent_harvest_object"), "harvest")


if __name__ == "__main__":
    unittest.main()
