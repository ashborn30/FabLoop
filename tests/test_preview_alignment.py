"""Synthetic geometry checks for the explicitly unverified preview alignment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from src.fabloop.photometric_stereo.preview_alignment import LIGHT_ORDER, align_lights


class PreviewAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _save(self, direction, image):
        Image.fromarray(image).save(self.root / f"light_{direction}.jpg", quality=98)

    def _textured_captures(self):
        rng = np.random.default_rng(731)
        base = rng.integers(10, 245, (600, 720, 3), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (0, 0), 1.2)
        for _ in range(120):
            x, y = rng.integers(20, 580, size=2)
            colour = tuple(int(value) for value in rng.integers(0, 255, size=3))
            cv2.circle(base, (int(x), int(y)), int(rng.integers(3, 15)), colour, 2)
        self._save("F", base[100:500, 130:650])
        self._save("B", base[85:515, 110:665])
        rotation_l = cv2.getRotationMatrix2D((360, 300), 3.0, 1.0)
        rotation_r = cv2.getRotationMatrix2D((360, 300), -4.0, 1.0)
        self._save("L", cv2.warpAffine(base, rotation_l, (720, 600))[95:505, 120:655])
        self._save("R", cv2.warpAffine(base, rotation_r, (720, 600))[110:510, 115:660])

    def test_unequal_crops_and_rotations_align_on_reference_with_honest_mask(self):
        self._textured_captures()
        before = {direction: hashlib.sha256((self.root / f"light_{direction}.jpg").read_bytes()).hexdigest()
                  for direction in LIGHT_ORDER}
        aligned, mask, metadata = align_lights(self.root, max_edge=384)
        self.assertEqual(list(aligned), list(LIGHT_ORDER))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertGreater(mask.mean(), 0.8)
        self.assertFalse(mask[0].any())
        self.assertFalse(mask[:, 0].any())
        self.assertFalse(mask.all())
        safe = cv2.erode(mask.astype(np.uint8), np.ones((15, 15), dtype=np.uint8)).astype(bool)
        reference = aligned["F"].astype(np.float32)
        for direction, image in aligned.items():
            with self.subTest(light=direction):
                self.assertEqual(image.dtype, np.uint8)
                self.assertEqual(image.shape, (*mask.shape, 3))
                self.assertLess(float(np.mean(np.abs(image.astype(np.float32)[safe] - reference[safe]))), 10.0)
                self.assertEqual(before[direction], hashlib.sha256((self.root / f"light_{direction}.jpg").read_bytes()).hexdigest())
                self.assertEqual(metadata["sources"][direction]["sha256"], before[direction])
                original_h, original_w = metadata["sources"][direction]["original_shape_hw"]
                resized_h, resized_w = metadata["sources"][direction]["resized_shape_hw"]
                self.assertLess(abs(resized_w / resized_h - original_w / original_h), 0.01)
        self.assertEqual(metadata["alignment_status"], "estimated_unverified")
        self.assertFalse(metadata["registration_verified"])
        self.assertFalse(metadata["ready_for_training"])
        json.dumps(metadata, allow_nan=False)

    def test_missing_directional_capture_fails_without_fallback(self):
        for direction in ("F", "B", "L"):
            self._save(direction, np.zeros((80, 120, 3), dtype=np.uint8))
        with self.assertRaisesRegex(FileNotFoundError, "light_R"):
            align_lights(self.root)

    def test_blank_captures_do_not_invent_registration(self):
        for direction in LIGHT_ORDER:
            self._save(direction, np.full((100, 160, 3), 127, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "insufficient image texture"):
            align_lights(self.root)


if __name__ == "__main__":
    unittest.main()
