"""Run the mandatory EfficientAD feature-shape gate with random weights only.

    .venv/Scripts/python.exe scripts/check_efficientad_shapes.py

No dataset, checkpoint, training, or anomaly-quality evaluation is involved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("slim-0.5", "baseline"), default="slim-0.5")
    parser.add_argument("--config", default="configs/efficientad.yaml")
    parser.add_argument("--device", default="cpu", help="Dummy execution device, e.g. cpu or cuda:0")
    parser.add_argument("--output", help="JSON report; defaults to docs/preflight/<variant>_shapes.json")
    args = parser.parse_args()
    name = "efficientad_slim_05" if args.variant == "slim-0.5" else "efficientad_baseline"
    output = resolve_path(args.output or f"docs/preflight/{name}_shapes.json")
    config_path = resolve_path(args.config)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL",
        "variant": args.variant,
        "candidate": args.variant == "slim-0.5",
        "trained": False,
        "pretrained_teacher_loaded": False,
        "purpose": "Mandatory synthetic feature-shape gate before training; random initial weights",
        "config": str(config_path),
        "device": args.device,
        "input_shape": [1, 3, 256, 256],
    }
    try:
        import torch
        import yaml

        from src.models.efficientad_slim import EfficientAdModel, EfficientAdModelSize, SlimEfficientAdModel
        from src.models.efficientad_slim.shape_check import check_feature_shapes

        report["versions"] = {"torch": torch.__version__, "anomalib": importlib.metadata.version("anomalib")}
        folder = ROOT / "src/models/efficientad_slim"
        manifest = json.loads((folder / "SOURCE.json").read_text(encoding="utf-8"))
        installed = Path(importlib.metadata.distribution("anomalib").locate_file(manifest["source_file"]))
        report["source"] = {
            "source_manifest_sha256": sha256(folder / "SOURCE.json"),
            "baseline_expected_sha256": manifest["sha256"],
            "baseline_actual_sha256": sha256(folder / "torch_model.py"),
            "installed_baseline_sha256": sha256(installed),
            "candidate_actual_sha256": sha256(folder / "slim_model.py"),
        }
        if (report["source"]["baseline_actual_sha256"] != manifest["sha256"]
                or report["source"]["installed_baseline_sha256"] != manifest["sha256"]
                or report["versions"]["anomalib"] != manifest["version"]):
            raise ValueError("Frozen baseline, installed Anomalib and SOURCE.json must match.")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
        report["config_sha256"] = sha256(config_path)
        settings = config.get("model", {})
        image_size = config.get("data", {}).get("image_size", [256, 256])
        report["configured_image_size"] = image_size
        if image_size != [256, 256] or any(type(value) is not int for value in image_size):
            raise ValueError("This gate requires data.image_size=[256, 256].")
        if settings.get("size", "small") != "small" or settings.get("teacher_out_channels", 384) != 384:
            raise ValueError("This gate requires EfficientAD-S and teacher_out_channels=384.")
        kwargs = {"teacher_out_channels": 384, "padding": settings.get("padding", False),
                  "pad_maps": settings.get("pad_maps", True)}
        if any(type(kwargs[key]) is not bool for key in ("padding", "pad_maps")):
            raise ValueError("model.padding and model.pad_maps must be YAML booleans.")
        report["model_settings"] = kwargs
        report["configured_device"] = config.get("device")
        report["seed"] = int(config.get("seed", 42))
        torch.manual_seed(report["seed"])
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(min(previous_threads, 4))
            core = (SlimEfficientAdModel(**kwargs) if args.variant == "slim-0.5" else
                    EfficientAdModel(**kwargs, model_size=EfficientAdModelSize.S))
            core = core.to(device=torch.device(args.device), dtype=torch.float32).eval()
            core.teacher.requires_grad_(False)
            report["shape_check"] = check_feature_shapes(core, image_size=(256, 256))
        finally:
            torch.set_num_threads(previous_threads)
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report["status"] == "PASS":
        check = report["shape_check"]
        print(f"Input: {check['input_shape']}")
        student_label = "Slim Student" if report["candidate"] else "Student"
        for label, key in (("Teacher", "teacher"), (student_label, "student")):
            print(f"{label}: {check['features'][key]}")
        for label, key in (("Student [:384]", "student_teacher"), ("Student [384:]", "student_autoencoder"),
                           ("Autoencoder", "autoencoder")):
            print(f"{label}: {check['features'][key]}")
        for name, shape in check["differences"].items():
            print(f"{name}: {shape} PASS")
    else:
        print(report["error"])
    print(f"{report['status']}: {output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
