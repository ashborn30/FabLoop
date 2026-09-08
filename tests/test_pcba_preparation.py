"""PCBA inventory must not silently invent alignment or normal-only labels."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.fabloop.photometric_stereo.pcba_prepare import LIGHT_ORDER, prepare_inventory, validate_labels


class PcbaPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.edited, self.raw = self.root / "edited", self.root / "raw"
        for source in (self.edited, self.raw):
            folder = source / "PCB1"
            folder.mkdir(parents=True)
            for direction in LIGHT_ORDER:
                Image.new("RGB", (12, 10), (30, 60, 90)).save(folder / f"light_{direction}.jpg")
        self.output, self.summary = self.root / "output", self.root / "summary.json"

    def run_inventory(self):
        return prepare_inventory(self.edited, self.raw, self.output, self.summary)

    def test_complete_images_remain_blocked_without_metadata_or_registration(self):
        result = self.run_inventory()
        self.assertEqual(result["counts"]["edited_images"], 4)
        self.assertEqual(result["counts"]["unknown_labels"], 1)
        self.assertFalse(result["ready_for_training"])
        self.assertTrue(result["calibration_missing"])
        board = json.loads((self.output / "inventory.json").read_text())["boards"][0]
        self.assertEqual(list(board["edited"]["lights"]), list(LIGHT_ORDER))
        self.assertIn("registration_not_verified", board["blockers"])
        self.assertEqual(len(board["edited"]["lights"]["F"]["sha256"]), 64)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ["inventory.json", "labels.json"])

    def test_incomplete_lights_and_mismatched_shapes_block_stacking(self):
        (self.edited / "PCB1/light_R.jpg").unlink()
        Image.new("RGB", (9, 10)).save(self.edited / "PCB1/light_B.jpg")
        result = self.run_inventory()
        self.assertEqual(result["status"], "BLOCKED_ALIGNMENT")
        self.assertEqual(result["counts"]["edited_groups_with_all_four_lights"], 0)
        board = json.loads((self.output / "inventory.json").read_text())["boards"][0]
        self.assertEqual(board["edited"]["missing_lights"], ["R"])
        self.assertIn("missing_directional_images", board["blockers"])
        self.assertIn("directional_image_shapes_not_stackable", board["blockers"])

    def test_manual_template_is_not_overwritten_and_odd_filename_is_not_ambient(self):
        Image.new("RGB", (12, 10)).save(self.edited / "PCB1/unknown_capture.jpg")
        self.run_inventory()
        labels_path = self.output / "labels.json"
        payload = json.loads(labels_path.read_text())
        payload["records"][0].update(label="normal", split="train", physical_board_id="board-a", category="type-a")
        exact = json.dumps(payload, indent=3) + "\n"
        labels_path.write_text(exact)
        self.run_inventory()
        self.assertEqual(labels_path.read_text(), exact)
        board = json.loads((self.output / "inventory.json").read_text())["boards"][0]
        self.assertIsNone(board["edited"]["ambient"])
        self.assertEqual(len(board["edited"]["unclassified_images"]), 1)

    def test_anomaly_train_or_calibration_and_physical_board_leakage_rejected(self):
        for split in ("train", "calibration"):
            payload = {"schema_version": 1, "records": [{"board_group_id": "PCB1", "label": "anomaly", "split": split}]}
            with self.assertRaisesRegex(ValueError, "only normal"):
                validate_labels(payload, ["PCB1"])
        payload = {"schema_version": 1, "records": [
            {"board_group_id": "PCB1", "label": "normal", "split": "train", "physical_board_id": "same-board"},
            {"board_group_id": "PCB2", "label": "normal", "split": "test", "physical_board_id": "same-board"},
        ]}
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_labels(payload, ["PCB1", "PCB2"])

    def test_outputs_cannot_modify_capture_trees(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_inventory(self.edited, self.raw, self.edited / "generated", self.summary)


if __name__ == "__main__":
    unittest.main()
