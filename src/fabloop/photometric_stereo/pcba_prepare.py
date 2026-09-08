"""Inventory PCBA captures without inventing labels, calibration, or alignment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

LIGHT_ORDER = ("F", "B", "L", "R")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    record: dict[str, Any] = {
        "path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(), "readable": False,
    }
    try:
        with Image.open(path) as image:
            record.update(width=image.width, height=image.height, mode=image.mode)
            image.load()
        record["readable"] = True
    except (OSError, ValueError, SyntaxError) as error:
        record["error"] = str(error)
    return record


def _scan_group(root: Path, board_id: str) -> dict[str, Any]:
    folder = root / board_id
    records = {path.name: _file_record(path) for path in sorted(folder.glob("*"))
               if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    lights = {direction: records.get(f"light_{direction}.jpg") for direction in LIGHT_ORDER}
    present = [record for record in lights.values() if record is not None]
    missing = [direction for direction, record in lights.items() if record is None]
    shapes = {(record.get("width"), record.get("height")) for record in present}
    known_names = {"ambient.jpg", *(f"light_{direction}.jpg" for direction in LIGHT_ORDER)}
    return {
        "root": str(folder.resolve()), "exists": folder.is_dir(), "lights": lights,
        "ambient": records.get("ambient.jpg"),
        "unclassified_images": [record for name, record in records.items() if name not in known_names],
        "unexpected_files": [str(path.resolve()) for path in sorted(folder.glob("*"))
                             if path.is_file() and path.suffix.lower() not in IMAGE_SUFFIXES],
        "image_count": len(records), "missing_lights": missing,
        "unreadable_image_count": sum(not record["readable"] for record in records.values()),
        "light_shapes_match": not missing and len(shapes) == 1 and all(record["readable"] for record in present),
        "alignment_status": "unverified",
    }


def validate_labels(payload: dict[str, Any], board_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Validate manual annotations, including physical-board split leakage."""
    if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError("Labels require schema_version=1 and a records list")
    records: dict[str, dict[str, Any]] = {}
    physical_assignments: dict[str, tuple[str, str]] = {}
    for record in payload["records"]:
        if not isinstance(record, dict):
            raise ValueError("Each label record must be an object")
        board_id = record.get("board_group_id")
        if board_id not in board_ids or board_id in records:
            raise ValueError(f"Unknown or duplicate board_group_id: {board_id!r}")
        label, split = record.get("label"), record.get("split")
        if label not in {"unknown", "normal", "anomaly"}:
            raise ValueError(f"Invalid label for {board_id}: {label!r}")
        if split not in {"unknown", "train", "calibration", "test"}:
            raise ValueError(f"Invalid split for {board_id}: {split!r}")
        if label == "anomaly" and split in {"train", "calibration"}:
            raise ValueError(f"{board_id}: only normal boards may enter train/calibration")
        physical_id = record.get("physical_board_id")
        if physical_id is not None and (not isinstance(physical_id, str) or not physical_id.strip()):
            raise ValueError(f"Invalid physical_board_id for {board_id}")
        if physical_id and split != "unknown":
            assignment = (split, label)
            if physical_id in physical_assignments and physical_assignments[physical_id] != assignment:
                raise ValueError(f"Physical board {physical_id!r} crosses splits or has conflicting labels")
            physical_assignments[physical_id] = assignment
        records[board_id] = record
    if set(records) != set(board_ids):
        raise ValueError("Labels must have exactly one record for every board group")
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_inventory(
    edited_root: Path, raw_root: Path, output_dir: Path, summary_path: Path,
    labels_path: Path | None = None,
) -> dict[str, Any]:
    """Write inventory and non-overwriting label template; never generate training images."""
    edited_root, raw_root = edited_root.resolve(), raw_root.resolve()
    output_dir, summary_path = output_dir.resolve(), summary_path.resolve()
    labels_path = labels_path.resolve() if labels_path else output_dir / "labels.json"
    inventory_path = output_dir / "inventory.json"
    if not edited_root.is_dir():
        raise FileNotFoundError(f"Edited capture directory not found: {edited_root}")
    if edited_root == raw_root:
        raise ValueError("Raw and edited capture roots must be distinct")
    for destination in (output_dir, summary_path, labels_path):
        if any(destination == root or root in destination.parents for root in (edited_root, raw_root)):
            raise ValueError("Inventory outputs must remain outside the original/edited capture roots")
    if len({inventory_path, summary_path, labels_path}) != 3:
        raise ValueError("Inventory, summary, and labels must use distinct paths")
    board_ids = sorted({path.name for root in (edited_root, raw_root) for path in root.glob("*")
                        if path.is_dir()}, key=lambda name: (int(name[3:]) if name.startswith("PCB") and name[3:].isdigit() else float("inf"), name))
    if not board_ids:
        raise ValueError("No board directories found")
    if not labels_path.exists():
        template = {
            "schema_version": 1,
            "instructions": "Enter verified labels, split, category, and physical_board_id. Keep every related view/crop in one split. Folder names do not establish physical identity.",
            "records": [{"board_group_id": board_id, "physical_board_id": None,
                         "category": None, "label": "unknown", "split": "unknown"}
                        for board_id in board_ids],
        }
        _write_json(labels_path, template)
    labels = validate_labels(json.loads(labels_path.read_text(encoding="utf-8-sig")), board_ids)
    boards = []
    for board_id in board_ids:
        edited, raw = _scan_group(edited_root, board_id), _scan_group(raw_root, board_id)
        annotation = labels[board_id]
        blockers = ["light_calibration_missing", "registration_not_verified", "board_mask_not_provided"]
        if edited["missing_lights"]:
            blockers.append("missing_directional_images")
        if not edited["light_shapes_match"]:
            blockers.append("directional_image_shapes_not_stackable")
        if edited["unreadable_image_count"]:
            blockers.append("unreadable_images")
        if annotation["label"] == "unknown" or annotation["split"] == "unknown":
            blockers.append("labels_or_split_unknown")
        if not annotation.get("physical_board_id"):
            blockers.append("physical_board_identity_unknown")
        if not annotation.get("category"):
            blockers.append("board_category_unknown")
        boards.append({"board_group_id": board_id, "annotation": annotation,
                       "edited": edited, "raw": raw, "blockers": blockers})
    mismatched = [board["board_group_id"] for board in boards if not board["edited"]["light_shapes_match"]]
    counts = {
        "board_groups": len(boards),
        "edited_images": sum(board["edited"]["image_count"] for board in boards),
        "raw_images": sum(board["raw"]["image_count"] for board in boards),
        "edited_groups_with_all_four_lights": sum(not board["edited"]["missing_lights"] for board in boards),
        "edited_groups_with_matching_light_shapes": len(boards) - len(mismatched),
        "unknown_labels": sum(board["annotation"]["label"] == "unknown" for board in boards),
        "unreadable_images": sum(board[source]["unreadable_image_count"] for board in boards for source in ("edited", "raw")),
    }
    summary = {
        "schema_version": 1, "stage": "inventory_only", "inventory_completed": True,
        "status": "BLOCKED_ALIGNMENT" if mismatched else "BLOCKED_METADATA",
        "blocking_statuses": ["BLOCKED_METADATA", "BLOCKED_ALIGNMENT"],
        "ready_for_training": False, "light_order": list(LIGHT_ORDER), "counts": counts,
        "calibration_missing": True, "registration_verified": False,
        "shape_mismatch_groups": mismatched,
        "inventory_path": str(inventory_path), "labels_path": str(labels_path),
        "notes": [
            "No source images copied or modified; hashes link raw and edited captures.",
            "F/B/L/R filenames establish order only, not calibrated light vectors or intensities.",
            "Matching dimensions do not prove image registration; resizing alone is not registration.",
            "Ambient is recognized by filename only; unusually named images remain unclassified.",
            "Physical identity across folders is unknown until supplied; keep related views, edits and patches in one split.",
            "Calibration, board masks and alignment are not supplied to this inventory stage.",
            "No PS features, training split files or dataset loader are generated by this stage.",
        ],
    }
    _write_json(inventory_path, {**summary, "sources": {"edited": str(edited_root), "raw": str(raw_root)}, "boards": boards})
    _write_json(summary_path, summary)
    return summary
