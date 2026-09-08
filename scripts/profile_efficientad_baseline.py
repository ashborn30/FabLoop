"""Profile the frozen EfficientAD-S source using synthetic inputs on CPU.

This is an architecture preflight, not training or a quality/latency benchmark.
No checkpoints or datasets are loaded. The copied source must still match the
installed Anomalib source byte for byte. Run with the project virtual environment:

    .venv/Scripts/python.exe scripts/profile_efficientad_baseline.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def profile(
    config_path: Path,
    *,
    model_factory: Callable[..., Any] | None = None,
    sample: Any = None,
) -> dict[str, Any]:
    """Count one model using the frozen-source policy shared by slim comparisons.

    With no keyword arguments this profiles the unchanged baseline. A factory
    receives ``teacher_out_channels``, ``padding`` and ``pad_maps``; a supplied
    sample lets architecture comparisons reuse exactly the same input tensor.
    """
    import torch
    from torch import nn
    from fvcore.nn import FlopCountAnalysis
    import yaml

    sys.path.insert(0, str(ROOT))
    from src.models.efficientad_slim import torch_model as copied_source

    source_path = Path(copied_source.__file__).resolve()
    installed_source = Path(importlib.metadata.distribution("anomalib").locate_file(
        "anomalib/models/image/efficient_ad/torch_model.py",
    )).resolve()
    source_hash = sha256(source_path)
    installed_hash = sha256(installed_source)
    manifest_path = source_path.with_name("SOURCE.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    installed_version = importlib.metadata.version("anomalib")
    if (source_hash != installed_hash or source_hash != manifest["sha256"]
            or installed_version != manifest["version"]):
        raise ValueError(
            "Copied source, installed Anomalib, and SOURCE.json must match. "
            "This script profiles only the unchanged source baseline.",
        )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    model_config = config.get("model", {})
    size = model_config.get("size", "small")
    channels = model_config.get("teacher_out_channels", 384)
    if size != "small" or channels != 384:
        raise ValueError("This baseline requires model.size=small and teacher_out_channels=384.")
    image_size = config.get("data", {}).get("image_size", [256, 256])
    if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
            or any(type(value) is not int or value <= 0 for value in image_size)):
        raise ValueError("data.image_size must contain two positive integer dimensions.")
    image_size = tuple(image_size)
    padding = model_config.get("padding", False)
    pad_maps = model_config.get("pad_maps", True)
    if not isinstance(padding, bool) or not isinstance(pad_maps, bool):
        raise ValueError("model.padding and model.pad_maps must be YAML booleans.")

    seed = int(config.get("seed", 42))
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model_kwargs = {
        "teacher_out_channels": channels,
        "padding": padding,
        "pad_maps": pad_maps,
    }
    if model_factory is None:
        model = copied_source.EfficientAdModel(
            **model_kwargs,
            model_size=copied_source.EfficientAdModelSize.S,
        )
    else:
        model = model_factory(**model_kwargs)
    model = model.cpu().float().eval()
    model.teacher.requires_grad_(False)
    if sample is None:
        sample = torch.rand((1, 3, *image_size), dtype=torch.float32, device="cpu")
    elif (not isinstance(sample, torch.Tensor)
          or tuple(sample.shape) != (1, 3, *image_size)
          or sample.dtype != torch.float32 or sample.device.type != "cpu"):
        raise ValueError("sample must be a CPU float32 tensor of shape (1, 3, *data.image_size).")

    class AutoencoderAdapter(nn.Module):
        """Supply the image-size argument without changing the copied AE."""

        def __init__(self, autoencoder: nn.Module) -> None:
            super().__init__()
            self.autoencoder = autoencoder

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.autoencoder(image, image_size=image_size)

    networks = {
        "teacher": model.teacher,
        "student": model.student,
        "autoencoder": AutoencoderAdapter(model.ae).eval(),
    }
    records: dict[str, Any] = {}
    ignored_policy: list[str] = []
    expected_channels = {"teacher": 384, "student": 768, "autoencoder": 384}
    spatial_shapes = set()
    with torch.no_grad():
        for name, network in networks.items():
            output = network(sample)
            shape = list(output.shape)
            if len(shape) != 4 or shape[0] != 1 or shape[1] != expected_channels[name]:
                raise AssertionError(f"Unexpected {name} output shape: {shape}")
            spatial_shapes.add(tuple(shape[2:]))
            analysis = FlopCountAnalysis(network, sample)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            supported_flops = int(analysis.total())
            by_operator = dict(sorted(analysis.by_operator().items()))
            # Expose the installed fvcore policy for reproducibility. This is
            # diagnostic metadata, not an assertion that every listed op ran.
            ignored_policy = sorted(analysis._ignored_ops)
            records[name] = {
                "parameters": sum(parameter.numel() for parameter in network.parameters()),
                "output_shape": shape,
                "supported_flops": supported_flops,
                "convolution_flops": int(by_operator.get("conv", 0)),
                "supported_flops_by_operator": by_operator,
                "supported_flops_by_module": dict(sorted(analysis.by_module().items())),
                "unsupported_operator_occurrences": dict(sorted(analysis.unsupported_ops().items())),
                "uncalled_modules": sorted(analysis.uncalled_modules()),
            }
            del output
    if len(spatial_shapes) != 1:
        raise AssertionError(f"Teacher/Student/AE spatial sizes differ: {records}")

    component_parameters = sum(record["parameters"] for record in records.values())
    calibration_parameters = {
        "mean_std": sum(parameter.numel() for parameter in model.mean_std.parameters()),
        "quantiles": sum(parameter.numel() for parameter in model.quantiles.parameters()),
    }
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    if model_parameters != component_parameters + sum(calibration_parameters.values()):
        raise AssertionError("EfficientAD model contains unexpected parameters outside T/S/AE and calibration.")
    aggregate_operators: Counter[str] = Counter()
    aggregate_unsupported: Counter[str] = Counter()
    for record in records.values():
        aggregate_operators.update(record["supported_flops_by_operator"])
        aggregate_unsupported.update(record["unsupported_operator_occurrences"])

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "EfficientAD-S architecture baseline; random weights and synthetic input only"
            if model_factory is None
            else "EfficientAD-S candidate architecture; random weights and synthetic input only"
        ),
        "source": {
            "copied_file": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "copied_sha256": source_hash,
            "installed_file": str(installed_source),
            "installed_sha256": installed_hash,
            "copied_source_matches_installed": True,
            "source_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "source_manifest_verified": True,
            "config_file": str(config_path),
            "config_sha256": sha256(config_path),
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "anomalib": importlib.metadata.version("anomalib"),
            "fvcore": importlib.metadata.version("fvcore"),
        },
        "execution": {
            "device": "cpu",
            "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
            "seed": seed,
            "input_shape": list(sample.shape),
            "input_distribution": "uniform [0, 1); each original module applies its ImageNet normalization",
            "model_size": size,
            "padding": padding,
            "pad_maps": pad_maps,
            "eval_mode": True,
            "grad_enabled_during_profile": False,
            "teacher_frozen": all(not parameter.requires_grad for parameter in model.teacher.parameters()),
            "pretrained_checkpoint_loaded": False,
            "training_or_dataset_inference_run": False,
        },
        "counting_policy": {
            "tool": "fvcore.nn.FlopCountAnalysis with unmodified default operator handlers",
            "multiply_add_flops": 1,
            "supported_flops_are_not_all_operation_flops": True,
            "convolution_flops": "Convolution multiply-add count; bias additions excluded by fvcore",
            "unsupported_operator_occurrences": "Counts of traced operator nodes, not element counts or FLOPs",
            "ignored_operators": (
                "fvcore deliberately assigns no FLOPs to ReLU, dropout and many shape/view/copy operations. "
                "Dropout is an identity in eval mode. See the installed policy list below; "
                "membership in that list does not mean the operator occurred in this trace."
            ),
            "fvcore_ignored_operator_types_policy": ignored_policy,
            "module_breakdown": "Inclusive nested-module totals; do not sum parent and child entries",
            "deployment_core_scope": (
                "Sum of Teacher + Student + Autoencoder feature-network forwards. "
                "Includes supported operators in each module's original forward. "
                "Excludes feature statistics normalization, feature distances, map upsampling/fusion, "
                "SAM3, external preprocessing, data loading and application overhead. "
                "This is not a full FabLoop inference cost or a latency measurement."
            ),
        },
        "modules": records,
        "student_only": {
            "parameters": records["student"]["parameters"],
            "supported_flops": records["student"]["supported_flops"],
            "convolution_flops": records["student"]["convolution_flops"],
            "future_reduction_targets": {"parameters_fraction_min": 0.40, "flops_fraction_min": 0.30},
        },
        "deployment_core_teacher_student_autoencoder": {
            "parameters": component_parameters,
            "supported_flops": sum(record["supported_flops"] for record in records.values()),
            "convolution_flops": sum(record["convolution_flops"] for record in records.values()),
            "supported_flops_by_operator": dict(sorted(aggregate_operators.items())),
            "unsupported_operator_occurrences": dict(sorted(aggregate_unsupported.items())),
        },
        "calibration_parameter_elements": calibration_parameters,
        "efficientad_model_parameter_elements_including_calibration": model_parameters,
        "quality_metrics": None,
        "compression_results": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientad.yaml")
    parser.add_argument("--output", default="docs/preflight/efficientad_baseline_profile.json")
    args = parser.parse_args()
    result = profile(resolve_path(args.config))
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, record in result["modules"].items():
        print(
            f"{name}: params={record['parameters']:,}; output={record['output_shape']}; "
            f"fvcore_supported_flops={record['supported_flops']:,}; "
            f"unsupported={record['unsupported_operator_occurrences']}",
        )
    print(f"Saved baseline profile: {output_path}")
    print("Synthetic architecture check only; no trained-model quality or latency result.")


if __name__ == "__main__":
    main()
