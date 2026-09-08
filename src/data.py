"""Dataset routing and audited split construction for EfficientAD."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from anomalib.data import MVTecAD, MVTecLOCO
from anomalib.data.datasets import AnomalibDataset
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize


PCB_CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
REQUIRED_COLUMNS = ("object", "split", "label", "image", "mask")
SUPPORTED_DATASETS = {"visa": "Visa", "mvtecad": "MVTecAD", "mvtecloco": "MVTecLoco"}
Sample = TypeVar("Sample")


def dataset_name(config: dict[str, Any]) -> str:
    """Return the canonical dataset name used by the routing layer."""

    value = str(config.get("dataset", {}).get("name", "")).strip().lower()
    try:
        return SUPPORTED_DATASETS[value]
    except KeyError as error:
        raise ValueError(
            "dataset.name must be one of: Visa, MVTecAD, MVTecLoco"
        ) from error


def dataset_root(config: dict[str, Any]) -> Path:
    """Resolve the dataset root from the shared or legacy VisA config schema."""

    dataset_config = config.get("dataset", {})
    value = dataset_config.get("root", config.get("data", {}).get("root"))
    if not value:
        raise ValueError("Dataset root is missing (dataset.root or data.root)")
    return Path(value)


def configured_category(config: dict[str, Any]) -> str:
    """Resolve category from the shared or legacy VisA config schema."""

    dataset_config = config.get("dataset", {})
    value = dataset_config.get("category", config.get("category"))
    if not value:
        raise ValueError("Dataset category is missing (dataset.category or category)")
    return str(value)


def report_categories(config: dict[str, Any]) -> tuple[str, ...]:
    """Return all categories represented by one config/report root."""

    return PCB_CATEGORIES if dataset_name(config) == "Visa" else (configured_category(config),)


class VisaCsvDataset(AnomalibDataset):
    """Anomalib dataset backed by rows from the official VisA split CSV."""

    def __init__(self, samples: pd.DataFrame, image_size: tuple[int, int]) -> None:
        super().__init__(augmentations=Resize(image_size, antialias=True))
        samples.attrs["task"] = "segmentation"
        self.samples = samples


@dataclass(frozen=True)
class SplitBundle:
    """Immutable train/calibration/test datasets and their audit manifest."""

    train: AnomalibDataset
    calibration: AnomalibDataset
    test: AnomalibDataset
    manifest: dict[str, Any]

    def train_loader(self, num_workers: int = 0) -> DataLoader:
        return _loader(self.train, shuffle=True, num_workers=num_workers)

    def calibration_loader(self, num_workers: int = 0) -> DataLoader:
        return _loader(self.calibration, shuffle=False, num_workers=num_workers)

    def test_loader(self, num_workers: int = 0) -> DataLoader:
        return _loader(self.test, shuffle=False, num_workers=num_workers)


def _loader(dataset: AnomalibDataset, *, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(csv_path: Path, category: str) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(
                f"Unexpected VisA CSV columns {reader.fieldnames}; expected {list(REQUIRED_COLUMNS)}"
            )
        rows = [dict(row) for row in reader if row["object"] == category]

    if not rows:
        raise ValueError(f"Category {category!r} is absent from {csv_path}")
    invalid_split = sorted({row["split"] for row in rows} - {"train", "test"})
    invalid_label = sorted({row["label"] for row in rows} - {"normal", "anomaly"})
    if invalid_split or invalid_label:
        raise ValueError(f"Invalid split/label values: split={invalid_split}, label={invalid_label}")
    return rows


def _records_to_frame(rows: list[dict[str, str]], root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        image_path = (root / Path(row["image"])).resolve()
        mask_value = row["mask"].strip()
        # Keep empty strings for normal rows; pandas may coerce ``None`` to NaN,
        # which is not a valid Anomalib mask path value.
        mask_path = str((root / Path(mask_value)).resolve()) if mask_value else ""
        records.append(
            {
                "image_path": str(image_path),
                "mask_path": mask_path,
                "label_index": 0 if row["label"] == "normal" else 1,
                "label": row["label"],
                "split": row["split"],
                "csv_image": row["image"],
                "csv_mask": mask_value,
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame.attrs["task"] = "segmentation"
    return frame


def _manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"image": row["image"], "label": row["label"], "mask": row["mask"]}
        for row in sorted(rows, key=lambda item: item["image"])
    ]


def split_calibration(
    samples: list[Sample], calibration_ratio: float, split_seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split a deterministic calibration-normal holdout from train-normal samples."""

    if not 0.0 < calibration_ratio < 1.0:
        raise ValueError("data.calibration_ratio must be strictly between 0 and 1")
    if len(samples) < 2:
        raise ValueError("At least two train-normal samples are required for calibration holdout")
    indices = list(range(len(samples)))
    random.Random(split_seed).shuffle(indices)
    calibration_count = max(1, min(len(indices) - 1, math.ceil(len(indices) * calibration_ratio)))
    calibration_indices = set(indices[:calibration_count])
    calibration = [sample for index, sample in enumerate(samples) if index in calibration_indices]
    train = [sample for index, sample in enumerate(samples) if index not in calibration_indices]
    return train, calibration


def _train_calibration_settings(data_config: dict[str, Any]) -> tuple[float, int]:
    if data_config.get("calibration_source") != "train_normal":
        raise ValueError("Visa and MVTecAD require calibration_source: train_normal")
    return float(data_config["calibration_ratio"]), int(data_config["split_seed"])


def _image_size(data_config: dict[str, Any]) -> tuple[int, int]:
    image_size = tuple(int(value) for value in data_config["image_size"])
    if len(image_size) != 2:
        raise ValueError("data.image_size must contain [height, width]")
    return image_size


def _relative_path(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _mask_paths(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(item) for item in values if item]


def _mvtec_manifest_rows(records: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["image_path"])):
        masks = [_relative_path(path, root) for path in _mask_paths(record.get("mask_path"))]
        rows.append(
            {
                "image": _relative_path(record["image_path"], root),
                "label": "normal" if int(record["label_index"]) == 0 else "anomaly",
                "mask": "" if not masks else masks[0] if len(masks) == 1 else masks,
            }
        )
    return rows


def _clone_with_records(
    dataset: AnomalibDataset, records: list[dict[str, Any]]
) -> AnomalibDataset:
    clone = copy.copy(dataset)
    frame = pd.DataFrame.from_records(records, columns=dataset.samples.columns)
    frame.attrs.update(dataset.samples.attrs)
    frame.attrs["task"] = "segmentation"
    clone.samples = frame.reset_index(drop=True)
    return clone


def manifest_path_for_category(config: dict[str, Any], category: str) -> Path:
    """Return the configured manifest path, adjusted only for a CLI category override."""

    configured = Path(config["data"]["split_manifest"])
    config_category = configured_category(config)
    if category == config_category:
        return configured
    return configured.with_name(f"{category}.json")


def _build_visa_splits(
    config: dict[str, Any], category: str, *, write_manifest: bool
) -> SplitBundle:
    """Build VisA splits without changing the official one-class test rows."""

    if category not in PCB_CATEGORIES:
        raise ValueError(f"EfficientAD PCB category must be one of {PCB_CATEGORIES}, got {category!r}")
    data_config = config["data"]
    if data_config.get("use_official_split") is not True:
        raise ValueError("data.use_official_split must remain true for comparable VisA results")
    if data_config.get("split_type") != "1cls":
        raise ValueError("Only the official VisA one-class (1cls) split is supported")
    ratio, seed = _train_calibration_settings(data_config)

    csv_path = Path(data_config["split_csv"])
    root = dataset_root(config)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Official split CSV not found: {csv_path}")
    if not root.is_dir():
        raise FileNotFoundError(f"VisA root not found: {root}")

    rows = _read_rows(csv_path, category)
    official_train = [row for row in rows if row["split"] == "train"]
    official_test = [row for row in rows if row["split"] == "test"]
    if any(row["label"] != "normal" for row in official_train):
        raise ValueError("Official one-class training split contains anomaly rows")
    if not any(row["label"] == "normal" for row in official_test) or not any(
        row["label"] == "anomaly" for row in official_test
    ):
        raise ValueError("Official test split must retain both normal and anomaly samples")

    train_rows, calibration_rows = split_calibration(official_train, ratio, seed)
    image_size = _image_size(data_config)

    missing_images = [str(root / row["image"]) for row in rows if not (root / row["image"]).is_file()]
    missing_masks = [
        str(root / row["mask"])
        for row in official_test
        if row["label"] == "anomaly" and (not row["mask"] or not (root / row["mask"]).is_file())
    ]
    if missing_images or missing_masks:
        preview = (missing_images + missing_masks)[:10]
        raise FileNotFoundError(f"VisA CSV references missing files (first 10): {preview}")

    manifest = {
        "schema_version": 1,
        "dataset_name": "Visa",
        "category": category,
        "split_type": data_config["split_type"],
        "use_official_split": True,
        "official_csv": str(csv_path),
        "official_csv_sha256": _sha256(csv_path),
        "calibration_source": "train_normal",
        "calibration_ratio": ratio,
        "split_seed": seed,
        "counts": {
            "official_train_normal": len(official_train),
            "train_normal": len(train_rows),
            "calibration_normal": len(calibration_rows),
            "test_normal": sum(row["label"] == "normal" for row in official_test),
            "test_anomaly": sum(row["label"] == "anomaly" for row in official_test),
        },
        "train": _manifest_rows(train_rows),
        "calibration": _manifest_rows(calibration_rows),
        "test": _manifest_rows(official_test),
    }

    if write_manifest:
        manifest_path = manifest_path_for_category(config, category)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return SplitBundle(
        train=VisaCsvDataset(_records_to_frame(train_rows, root), image_size),
        calibration=VisaCsvDataset(_records_to_frame(calibration_rows, root), image_size),
        test=VisaCsvDataset(_records_to_frame(official_test, root), image_size),
        manifest=manifest,
    )


def _build_mvtec_splits(
    config: dict[str, Any], category: str, *, write_manifest: bool
) -> SplitBundle:
    """Use Anomalib's native MVTec datamodule and preserve its official splits."""

    name = dataset_name(config)
    data_config = config["data"]
    image_size = _image_size(data_config)
    root = dataset_root(config)
    if not root.is_dir():
        raise FileNotFoundError(f"{name} root not found: {root}")

    datamodule_class = MVTecAD if name == "MVTecAD" else MVTecLOCO
    datamodule_seed = int(data_config.get("split_seed", config.get("seed", 42)))
    datamodule = datamodule_class(
        root=root,
        category=category,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        augmentations=Resize(image_size, antialias=True),
        seed=datamodule_seed,
    )
    # Anomalib 2.6 stores multiple LOCO masks as lists in one dataframe column.
    # Pandas' new string inference rejects that assignment unless legacy object
    # inference is scoped to the native datamodule setup call.
    with pd.option_context("future.infer_string", False):
        datamodule.setup()
    official_train_records = datamodule.train_data.samples.to_dict("records")
    official_test_records = datamodule.test_data.samples.to_dict("records")
    if any(int(record["label_index"]) != 0 for record in official_train_records):
        raise ValueError(f"{name} official training split contains anomaly samples")
    test_labels = {int(record["label_index"]) for record in official_test_records}
    if test_labels != {0, 1}:
        raise ValueError(f"{name} official test split must contain normal and anomaly samples")

    if name == "MVTecLoco":
        if data_config.get("calibration_source") != "official_validation":
            raise ValueError("MVTecLoco requires calibration_source: official_validation")
        calibration_records = datamodule.val_data.samples.to_dict("records")
        if not calibration_records:
            raise ValueError("MVTecLoco official validation split is empty")
        if any(int(record["label_index"]) != 0 for record in calibration_records):
            raise ValueError("MVTecLoco official validation split contains anomaly samples")
        train_records = official_train_records
        calibration_dataset = datamodule.val_data
        calibration_source = "official_validation"
        calibration_ratio = None
        manifest_split_seed = None
    else:
        ratio, split_seed = _train_calibration_settings(data_config)
        train_records, calibration_records = split_calibration(
            official_train_records, ratio, split_seed
        )
        calibration_dataset = _clone_with_records(datamodule.train_data, calibration_records)
        calibration_source = "train_normal"
        calibration_ratio = ratio
        manifest_split_seed = split_seed
    train_manifest = _mvtec_manifest_rows(train_records, root)
    calibration_manifest = _mvtec_manifest_rows(calibration_records, root)
    test_manifest = _mvtec_manifest_rows(official_test_records, root)
    manifest = {
        "schema_version": 1,
        "dataset_name": name,
        "category": category,
        "split_type": "official_directory",
        "use_official_split": True,
        "official_csv": None,
        "official_csv_sha256": None,
        "dataset_root": str(root),
        "calibration_source": calibration_source,
        "calibration_ratio": calibration_ratio,
        "split_seed": manifest_split_seed,
        "counts": {
            "official_train_normal": len(official_train_records),
            "train_normal": len(train_records),
            "calibration_normal": len(calibration_records),
            "test_normal": sum(int(record["label_index"]) == 0 for record in official_test_records),
            "test_anomaly": sum(int(record["label_index"]) == 1 for record in official_test_records),
        },
        "train": train_manifest,
        "calibration": calibration_manifest,
        "test": test_manifest,
    }
    if write_manifest:
        manifest_path = manifest_path_for_category(config, category)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return SplitBundle(
        train=_clone_with_records(datamodule.train_data, train_records),
        calibration=calibration_dataset,
        test=datamodule.test_data,
        manifest=manifest,
    )


def build_official_splits(
    config: dict[str, Any], category: str, *, write_manifest: bool = True
) -> SplitBundle:
    """Route to the official dataset and its dataset-specific calibration source."""

    name = dataset_name(config)
    if name == "Visa":
        return _build_visa_splits(config, category, write_manifest=write_manifest)
    return _build_mvtec_splits(config, category, write_manifest=write_manifest)
