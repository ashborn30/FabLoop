"""Single CLI entrypoint that reads config and orchestrates EfficientAD stages."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

from benchmarker import benchmark
from calibrator import calibrate
from data import (
    PCB_CATEGORIES,
    build_official_splits,
    configured_category,
    dataset_name,
    dataset_root,
    manifest_path_for_category,
)
from evaluator import evaluate
from trainer import train
from validator import validate


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def resolved_run_config(base: dict[str, Any], category: str, seed: int) -> dict[str, Any]:
    config = copy.deepcopy(base)
    original_category = configured_category(config)
    original_manifest = Path(config["data"]["split_manifest"])
    if "category" in config:
        config["category"] = category
    if "category" in config.get("dataset", {}):
        config["dataset"]["category"] = category
    config["seed"] = seed
    if category != original_category:
        config["data"]["split_manifest"] = str(original_manifest.with_name(f"{category}.json"))
    return config


def build_reports_from_current_source(config: dict[str, Any]) -> tuple[Path, bool]:
    """Load reporter at report time so a long training run cannot use a stale module."""
    importlib.invalidate_caches()
    reporter_module = sys.modules.get("reporter")
    if reporter_module is None:
        reporter_module = importlib.import_module("reporter")
    else:
        reporter_module = importlib.reload(reporter_module)
    return reporter_module.build_reports(config)


def preflight(config: dict[str, Any], category: str, device: torch.device) -> dict[str, Any]:
    name = dataset_name(config)
    if name == "Visa":
        split_csv = Path(config["data"]["split_csv"])
        if not split_csv.is_file():
            local_copy = dataset_root(config) / "split_csv" / "1cls.csv"
            hint = (
                f" An existing candidate is at {local_copy}; verify/copy it explicitly."
                if local_copy.is_file()
                else ""
            )
            raise FileNotFoundError(f"Configured official split CSV is missing: {split_csv}.{hint}")
    sam_config = config.get("sam") or {}
    sam_enabled = bool(sam_config.get("enabled", False))
    if sam_enabled:
        if name != "Visa":
            raise ValueError("SAM refinement is enabled only for the VisA PCB baseline")
        sam_checkpoint = Path(sam_config["checkpoint"])
        if not sam_checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
    splits = build_official_splits(config, category, write_manifest=True)
    manifest_path = manifest_path_for_category(config, category)
    result = {
        "status": "PASS",
        "dataset": name,
        "category": category,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "anomalib_version": __import__("anomalib").__version__,
        "split_manifest": str(manifest_path),
        "counts": splits.manifest["counts"],
        "imagenette": {
            "path": config["imagenette"]["root"],
            "status": "READY" if Path(config["imagenette"]["root"]).is_dir() else "DOWNLOAD_ON_TRAIN",
        },
    }
    if sam_enabled:
        result["sam"] = {
            "enabled": True,
            "checkpoint": str(sam_checkpoint),
            "checkpoint_size_bytes": sam_checkpoint.stat().st_size,
            "threshold_source": sam_config["threshold_source"],
        }
    print(json.dumps(result, indent=2))
    return result


def run_stage(stage: str, config: dict[str, Any], category: str, seed: int, device: torch.device) -> None:
    if stage == "preflight":
        preflight(config, category, device)
        return
    if stage == "report":
        path, passed = build_reports_from_current_source(config)
        print(f"reproduction_gate={'PASS' if passed else 'FAIL'}: {path}")
        return

    splits = build_official_splits(config, category, write_manifest=True)
    num_workers = int(config["data"]["num_workers"])
    train_loader = splits.train_loader(num_workers)
    calibration_loader = splits.calibration_loader(num_workers)
    test_loader = splits.test_loader(num_workers)
    calibration_ids = [row["image"] for row in splits.manifest["calibration"]]
    if stage == "train":
        print(train(config, category, seed, device, train_loader, splits.manifest))
    elif stage == "calibrate":
        print(calibrate(config, category, seed, device, calibration_loader, calibration_ids))
    elif stage == "evaluate":
        print(evaluate(config, category, seed, device, test_loader))
    elif stage == "benchmark":
        print(benchmark(config, category, seed, device, calibration_loader))
    elif stage == "validate":
        print(validate(config, category, seed, device, splits))
    elif stage == "run":
        preflight(config, category, device)
        train(config, category, seed, device, train_loader, splits.manifest)
        calibrate(config, category, seed, device, calibration_loader, calibration_ids)
        evaluate(config, category, seed, device, test_loader)
        benchmark(config, category, seed, device, calibration_loader)
        print(validate(config, category, seed, device, splits))
    else:
        raise ValueError(f"Unknown stage: {stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Anomalib EfficientAD pipeline for VisA PCB")
    parser.add_argument(
        "stage",
        choices=("preflight", "train", "calibrate", "evaluate", "benchmark", "validate", "report", "run"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/efficientad.yaml"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    seed = int(args.seed if args.seed is not None else base["seed"])
    requested = args.category or configured_category(base)
    if requested == "all" and dataset_name(base) != "Visa":
        raise ValueError("--category all is supported only by the multi-category VisA config")
    categories = PCB_CATEGORIES if requested == "all" else (requested,)
    if args.stage == "report":
        path, passed = build_reports_from_current_source(base)
        print(f"reproduction_gate={'PASS' if passed else 'FAIL'}: {path}")
        return
    device = resolve_device(str(args.device or base["device"]))
    for category in categories:
        config = resolved_run_config(base, category, seed)
        run_stage(args.stage, config, category, seed, device)
    if args.stage == "run":
        path, passed = build_reports_from_current_source(base)
        print(f"reproduction_gate={'PASS' if passed else 'FAIL'}: {path}")


if __name__ == "__main__":
    main()
