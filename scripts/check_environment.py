"""Inspect dependencies and configured inputs without running the ML pipeline.

Run from any directory with the project's virtual-environment Python. Only the
JSON report is written; no models, datasets, loaders, or tensors are constructed.
Exit 0 means the checks passed, 1 means an environment error or input blocker.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import importlib.metadata
import inspect
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
PCB_CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
CSV_COLUMNS = ("object", "split", "label", "image", "mask")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEPENDENCIES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "anomalib": "anomalib",
    "lightning": "lightning",
    "torchmetrics": "torchmetrics",
    "numpy": "numpy",
    "pandas": "pandas",
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "timm": "timm",
    "fvcore.nn": "fvcore",
    "pycocotools.mask": "pycocotools",
    "triton": "triton-windows" if sys.platform == "win32" else "triton",
    "PySide6.QtWidgets": "PySide6",
    "pyqtgraph": "pyqtgraph",
    "OpenGL.GL": "PyOpenGL",
}


class Preflight:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.modules: dict[str, Any] = {}
        self.sources: dict[str, ast.Module] = {}
        self.configs: dict[str, dict[str, Any]] = {}

    def record(self, name: str, status: str, **details: Any) -> None:
        self.checks.append({"name": name, "status": status, **details})
        print(f"[{status}] {name}", flush=True)

    def attempt(self, name: str, operation: Callable[[], Any]) -> Any:
        try:
            result = operation()
        except Exception as error:
            self.record(name, "ERROR", error=f"{type(error).__name__}: {error}")
            return None
        self.record(name, "PASS", details=result)
        return result

    def check_sources(self) -> None:
        paths = sorted((ROOT / "src").glob("*.py"))
        if not paths:
            self.record("src.syntax", "ERROR", error="No Python source files found")
        for path in paths:
            def check(path: Path = path) -> dict[str, Any]:
                source = path.read_text(encoding="utf-8-sig")
                compile(source, str(path), "exec")
                self.sources[path.stem] = ast.parse(source, filename=str(path))
                return {"file": str(path.relative_to(ROOT))}
            self.attempt(f"src.syntax.{path.stem}", check)

    def import_module(self, name: str, distribution: str | None = None) -> None:
        def check() -> dict[str, Any]:
            module = importlib.import_module(name)
            self.modules[name] = module
            result = {"file": getattr(module, "__file__", None)}
            if distribution:
                result["version"] = importlib.metadata.version(distribution)
            return result
        self.attempt(f"import.{name}", check)

    def check_imports(self) -> None:
        for name, distribution in DEPENDENCIES.items():
            self.import_module(name, distribution)
        # Match `python src/anomaly.py`: src is importable, vendor paths are not
        # injected. SAM3 must be genuinely registered in this environment.
        for name in sorted(self.sources):
            self.import_module(name)
        self.import_module("sam3", "sam3")

        def registration() -> dict[str, Any]:
            distribution = importlib.metadata.distribution("sam3")
            direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
            module_path = Path(self.modules["sam3"].__file__).resolve()
            expected = (ROOT / "third-party" / "SAM3" / "sam3").resolve()
            if not direct_url.get("dir_info", {}).get("editable"):
                raise ValueError("SAM3 is not installed in editable mode")
            if not module_path.is_relative_to(expected):
                raise ValueError(f"SAM3 imports from {module_path}, expected {expected}")
            return {"module": str(module_path), "direct_url": direct_url}
        self.attempt("sam3.editable_registration", registration)

    def check_contracts(self) -> None:
        targets = {
            "EfficientAd": ("anomalib.models", "EfficientAd", "model", False),
            "build_sam3_image_model": (
                "sam3.model_builder", "build_sam3_image_model", "sam_refiner", False
            ),
            "Sam3Processor": (
                "sam3.model.sam3_image_processor", "Sam3Processor", "sam_refiner", False
            ),
            "predict_inst": (
                "sam3.model.sam3_image", "Sam3Image.predict_inst", "sam_refiner", True
            ),
        }
        for call_name, (module_name, attribute, source_name, method) in targets.items():
            def check(
                call_name: str = call_name, module_name: str = module_name,
                attribute: str = attribute, source_name: str = source_name,
                method: bool = method,
            ) -> dict[str, Any]:
                target = importlib.import_module(module_name)
                for part in attribute.split("."):
                    target = getattr(target, part)
                signature = inspect.signature(target)
                calls = [
                    node for node in ast.walk(self.sources[source_name])
                    if isinstance(node, ast.Call)
                    and (
                        isinstance(node.func, ast.Name) and node.func.id == call_name
                        or isinstance(node.func, ast.Attribute) and node.func.attr == call_name
                    )
                ]
                if not calls:
                    raise ValueError(f"No {call_name} call found in src/{source_name}.py")
                for call in calls:
                    if any(keyword.arg is None for keyword in call.keywords):
                        raise ValueError("Cannot statically check unpacked keyword arguments")
                    args = [None] * (len(call.args) + int(method))
                    kwargs = {keyword.arg: None for keyword in call.keywords}
                    signature.bind(*args, **kwargs)
                    if call_name == "predict_inst":
                        # Sam3Image accepts **kwargs and forwards them to predict.
                        predictor = importlib.import_module("sam3.model.sam1_task_predictor")
                        inspect.signature(predictor.SAM3InteractiveImagePredictor.predict).bind(
                            None, **kwargs
                        )
                result = {"signature": str(signature), "source_calls_checked": len(calls)}
                if call_name == "predict_inst":
                    result["forwarded_signature"] = str(inspect.signature(
                        predictor.SAM3InteractiveImagePredictor.predict
                    ))
                return result
            self.attempt(f"contract.{call_name}", check)

    def check_configs(self) -> None:
        yaml = self.modules.get("yaml")
        if yaml is None:
            self.record("configs.yaml", "ERROR", error="PyYAML import failed")
            return
        paths = sorted(set((ROOT / "configs").rglob("*.yaml"))
                       | set((ROOT / "configs").rglob("*.yml")))
        if not paths:
            self.record("configs.yaml", "ERROR", error="No YAML config files found")
        for path in paths:
            def check(path: Path = path) -> dict[str, Any]:
                config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
                if not isinstance(config, dict):
                    raise ValueError("Config must contain a YAML mapping")
                self.configs[str(path.relative_to(ROOT))] = config
                return {"file": str(path.relative_to(ROOT))}
            self.attempt(f"config.{path.stem}", check)

    def check_cuda(self) -> None:
        torch = self.modules.get("torch")
        if torch is None:
            return
        def inspect_cuda() -> dict[str, Any]:
            available = torch.cuda.is_available()
            devices = []
            if available:
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    devices.append({
                        "index": index, "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "compute_capability": [properties.major, properties.minor],
                    })
            return {"available": available, "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda, "devices": devices}
        result = self.attempt("cuda.inspect", inspect_cuda)
        if result is not None:
            requested = [name for name, config in self.configs.items()
                         if str(config.get("device", "cpu")).startswith("cuda")]
            if requested and not result["available"]:
                self.record("cuda.configured_device", "BLOCKED_DEVICE", configs=requested,
                            error="Configs request CUDA but CUDA is unavailable")

    def required_path(self, name: str, path: Path, *, file: bool = False,
                      status: str = "BLOCKED_DATA") -> bool:
        exists = path.is_file() if file else path.is_dir()
        details: dict[str, Any] = {"path": str(path), "exists": exists}
        if exists and file:
            details["size_bytes"] = path.stat().st_size
            exists = details["size_bytes"] > 0
        self.record(name, "PASS" if exists else status, **details)
        return exists

    def check_inputs(self) -> None:
        for name, config in self.configs.items():
            try:
                data = self.modules.get("data")
                if data is not None:
                    dataset = data.dataset_name(config)
                    root = ROOT / data.dataset_root(config)
                    category = data.configured_category(config)
                else:
                    dataset = str(config["dataset"]["name"])
                    root = ROOT / Path(config["dataset"].get("root", config["data"].get("root")))
                    category = config["dataset"].get("category", config.get("category"))
                self.required_path(f"{name}.dataset_root", root)
                imagenette = ROOT / Path(config["imagenette"]["root"])
                self.record(f"{name}.imagenette", "PASS" if imagenette.is_dir() else "INFO",
                            path=str(imagenette), exists=imagenette.is_dir(),
                            availability="READY" if imagenette.is_dir() else "DOWNLOAD_ON_TRAIN")
                sam = config.get("sam", {})
                if sam.get("enabled", False):
                    self.required_path(f"{name}.sam_checkpoint", ROOT / Path(sam["checkpoint"]),
                                       file=True, status="BLOCKED_CHECKPOINT")
                if dataset.lower() == "visa":
                    csv_path = ROOT / Path(config["data"]["split_csv"])
                    if self.required_path(f"{name}.split_csv", csv_path, file=True):
                        try:
                            self.check_visa_csv(name, csv_path, root)
                        except (OSError, ValueError, csv.Error) as error:
                            self.record(f"{name}.csv_integrity", "BLOCKED_DATA",
                                        error=f"{type(error).__name__}: {error}")
                else:
                    required_dirs = ["train/good", "test/good", "ground_truth"]
                    if dataset.lower() == "mvtecloco":
                        required_dirs += ["validation/good", "test/logical_anomalies",
                                          "test/structural_anomalies"]
                    for relative in required_dirs:
                        path = root / str(category) / relative
                        if self.required_path(f"{name}.{relative}", path):
                            count = sum(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                                        for p in path.rglob("*"))
                            self.record(f"{name}.{relative}.images",
                                        "PASS" if count else "BLOCKED_DATA", count=count)
            except Exception as error:
                self.record(f"{name}.input_configuration", "ERROR",
                            error=f"{type(error).__name__}: {error}")

    def check_visa_csv(self, name: str, csv_path: Path, root: Path) -> None:
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                raise ValueError(f"CSV columns must be {CSV_COLUMNS}, got {reader.fieldnames}")
            rows = list(reader)
        malformed = [index for index, row in enumerate(rows, 2)
                     if None in row or any(row.get(column) is None for column in CSV_COLUMNS)]
        if malformed:
            raise ValueError(f"Malformed CSV rows (first 10): {malformed[:10]}")
        unsafe, duplicates, invalid = [], [], []
        seen: set[str] = set()
        missing_images, missing_masks = [], []
        counts: Counter[tuple[str, str, str]] = Counter()
        for line, row in enumerate(rows, 2):
            valid_paths = True
            for column in ("image", "mask"):
                value = row[column].strip().replace("\\", "/")
                parts = PurePosixPath(value).parts
                if (column == "image" and not value) or (value and (
                    PurePosixPath(value).is_absolute() or PureWindowsPath(value).anchor
                    or ".." in parts or ":" in value
                    or not (root / value).resolve().is_relative_to(root.resolve())
                )):
                    unsafe.append({"line": line, "column": column, "value": value})
                    valid_paths = False
            image_key = row["image"].replace("\\", "/").casefold()
            if image_key in seen:
                duplicates.append({"line": line, "image": row["image"]})
            seen.add(image_key)
            if row["object"] not in PCB_CATEGORIES:
                continue
            counts[row["object"], row["split"], row["label"]] += 1
            if (row["split"] not in {"train", "test"}
                or row["label"] not in {"normal", "anomaly"}
                or row["split"] == "train" and row["label"] != "normal"):
                invalid.append(line)
            if valid_paths:
                if not (root / row["image"]).is_file():
                    missing_images.append(row["image"])
                if row["label"] == "anomaly" and (
                    not row["mask"].strip() or not (root / row["mask"]).is_file()
                ):
                    missing_masks.append(row["mask"] or f"line {line}: empty mask")
        counts_by_category = {
            category: {f"{split}_{label}": counts[category, split, label]
                       for split, label in (("train", "normal"), ("test", "normal"),
                                            ("test", "anomaly"))}
            for category in PCB_CATEGORIES
        }
        missing_groups = [f"{category}.{group}" for category, groups in counts_by_category.items()
                          for group, count in groups.items() if count == 0]
        bad = bool(unsafe or duplicates or invalid or missing_groups)
        self.record(f"{name}.csv_integrity", "BLOCKED_DATA" if bad else "PASS",
                    total_rows=len(rows), pcb_counts=counts_by_category,
                    unsafe_count=len(unsafe), unsafe_preview=unsafe[:10],
                    duplicate_count=len(duplicates), duplicate_preview=duplicates[:10],
                    invalid_label_split_rows=invalid[:10], missing_groups=missing_groups)
        self.record(f"{name}.csv_referenced_files",
                    "BLOCKED_DATA" if missing_images or missing_masks else "PASS",
                    missing_images=len(missing_images), missing_masks=len(missing_masks),
                    missing_image_preview=missing_images[:10], missing_mask_preview=missing_masks[:10])

    def report(self) -> dict[str, Any]:
        counts = Counter(check["status"] for check in self.checks)
        packages_valid = all(check["status"] == "PASS" for check in self.checks
                             if check["name"].startswith(("import.", "contract.", "sam3.")))
        environment_valid = counts["ERROR"] == 0
        pipeline_ready = environment_valid and not any(
            check["status"].startswith("BLOCKED_") for check in self.checks
        )
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "root": str(ROOT), "python": sys.version, "executable": sys.executable,
            "venv": sys.prefix != sys.base_prefix, "packages_valid": packages_valid,
            "environment_valid": environment_valid,
            "pipeline_ready": pipeline_ready, "status_counts": dict(counts),
            "scope": "Import/signature/config/file checks only; no training or inference",
            "limitations": [
                "pipeline_ready covers configured inputs; Imagenette may download when training later runs. No pipeline stage runs during this check.",
                "Model construction, checkpoint contents, GPU kernels and model outputs are untested.",
                "Dataset files are checked by path, not decoded or loaded into datamodules.",
                "EfficientAD teacher download/cache and existing trained outputs are not validated.",
            ],
            "checks": self.checks,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/preflight/environment_check.json"),
                        help="JSON report path, relative to the project root by default")
    args = parser.parse_args()
    previous_cwd, previous_path = Path.cwd(), sys.path[:]
    previous_bytecode = sys.dont_write_bytecode
    try:
        os.chdir(ROOT)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(ROOT / "src"))
        preflight = Preflight()
        preflight.check_sources()
        preflight.check_imports()
        preflight.check_contracts()
        preflight.check_configs()
        preflight.check_cuda()
        preflight.check_inputs()
        report = preflight.report()
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in
                          ("packages_valid", "environment_valid", "pipeline_ready", "status_counts")}, indent=2))
        print(f"Report: {output}")
        return 0 if report["pipeline_ready"] else 1
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        os.chdir(previous_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
