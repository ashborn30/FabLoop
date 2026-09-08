"""Compare the first Slim-0.5 candidate with the frozen EfficientAD-S baseline.

Both models use the same synthetic CPU input and fvcore counting policy. This
checks architecture cost only: it does not train, load a checkpoint, or measure
dataset quality, latency, or complete FabLoop/SAM3 deployment cost.
Exit code 0 means the Student compression targets pass; exit code 1 means
revise the architecture before training. Both outcomes save the measured report.

    .venv/Scripts/python.exe scripts/profile_efficientad_slim.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.profile_efficientad_baseline import profile, resolve_path, sha256


def reductions(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Report both absolute counts and relative reductions without rounding gates."""
    result = {}
    for metric in ("parameters", "supported_flops", "convolution_flops"):
        original = baseline[metric]
        slim = candidate[metric]
        fraction = (original - slim) / original
        result[metric] = {
            "baseline": original,
            "candidate": slim,
            "reduction_absolute": original - slim,
            "reduction_fraction": fraction,
            "reduction_percent": 100 * fraction,
        }
    return result


def compare(config_path: Path) -> dict[str, Any]:
    import torch
    import yaml

    from src.models.efficientad_slim import slim_model

    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    seed = int(config.get("seed", 42))
    image_size = config.get("data", {}).get("image_size", [256, 256])
    # A dedicated generator makes input values independent of how many random
    # parameters either architecture initializes. Both runs receive this tensor.
    sample = torch.rand(
        (1, 3, *image_size),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
        device="cpu",
    )
    baseline = profile(config_path, sample=sample)
    candidate = profile(config_path, model_factory=slim_model.SlimEfficientAdModel, sample=sample)
    if baseline["execution"] != candidate["execution"]:
        raise AssertionError("Baseline and candidate execution settings must match.")
    if baseline["counting_policy"] != candidate["counting_policy"]:
        raise AssertionError("Baseline and candidate fvcore policies must match.")
    for name in ("teacher", "student", "autoencoder"):
        if baseline["modules"][name]["output_shape"] != candidate["modules"][name]["output_shape"]:
            raise AssertionError(f"Candidate changed the {name} output shape.")
    teacher_unchanged = {
        metric: baseline["modules"]["teacher"][metric] == candidate["modules"]["teacher"][metric]
        for metric in ("parameters", "supported_flops", "convolution_flops", "output_shape")
    }
    if not all(teacher_unchanged.values()):
        raise AssertionError("Slim-0.5 must preserve Teacher parameters, FLOPs and output shape.")

    core_key = "deployment_core_teacher_student_autoencoder"
    changes = {
        "student_only": reductions(baseline["modules"]["student"], candidate["modules"]["student"]),
        "autoencoder_only": reductions(baseline["modules"]["autoencoder"], candidate["modules"]["autoencoder"]),
        core_key: reductions(baseline[core_key], candidate[core_key]),
    }
    student_change = changes["student_only"]
    parameter_gate = student_change["parameters"]["reduction_fraction"] >= 0.40
    flops_gate = student_change["supported_flops"]["reduction_fraction"] >= 0.30
    gate_passed = parameter_gate and flops_gate
    candidate_path = Path(slim_model.__file__).resolve()
    baseline["architecture_id"] = "efficientad-s"
    candidate["architecture_id"] = "efficientad-s-slim-0.5"
    for record in (baseline, candidate):
        record["source"]["source_manifest_sha256"] = sha256(ROOT / record["source"]["source_manifest"])
    candidate["source"]["candidate_file"] = candidate_path.relative_to(ROOT).as_posix()
    candidate["source"]["candidate_sha256"] = sha256(candidate_path)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "First Slim-0.5 candidate architecture comparison; not a final architecture selection",
        "shared_input": {
            "shape": list(sample.shape),
            "sha256_float32_bytes": hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
            "generation": "torch.rand with a dedicated CPU generator; uniform [0, 1)",
            "seed": seed,
            "same_tensor_used_for_both_profiles": True,
        },
        "baseline": baseline,
        "candidate": candidate,
        "reductions": changes,
        "student_compression_gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "applies_to": "Student only; the deployment core is reported separately",
            "parameters_fraction_min": 0.40,
            "supported_flops_fraction_min": 0.30,
            "parameters_passed": parameter_gate,
            "supported_flops_passed": flops_gate,
            "passed": gate_passed,
            "decision": (
                "PROCEED_TO_TRAINING_INTEGRATION" if gate_passed
                else "REVISE_ARCHITECTURE_BEFORE_TRAINING"
            ),
            "flops_scope": (
                "fvcore-supported FLOPs with unmodified default handlers; one multiply-add is one FLOP. "
                "Unsupported operator occurrences are reported for each module in both profiles."
            ),
        },
        "teacher_unchanged": teacher_unchanged,
        "limitations": [
            "Randomly initialized weights and synthetic input only; no pretrained checkpoint loaded.",
            "Teacher + Student + AE feature-forward core only; SAM3 and anomaly-map postprocessing excluded.",
            "FLOPs omit unsupported and deliberately ignored operators; see both counting-policy records.",
            "No training, dataset inference, accuracy, quality, latency, or Jetson benchmark was run.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientad.yaml")
    parser.add_argument("--output", default="docs/preflight/efficientad_slim_05_profile.json")
    args = parser.parse_args(argv)
    result = compare(resolve_path(args.config))
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for scope, record in result["reductions"].items():
        params = record["parameters"]
        flops = record["supported_flops"]
        print(
            f"{scope}: params {params['baseline']:,} -> {params['candidate']:,} "
            f"(-{params['reduction_percent']:.2f}%); "
            f"fvcore_supported_flops {flops['baseline']:,} -> {flops['candidate']:,} "
            f"(-{flops['reduction_percent']:.2f}%)",
        )
    print(f"Student-only compression gate passed: {result['student_compression_gate']['passed']}")
    print(f"Teacher cost/output unchanged: {all(result['teacher_unchanged'].values())}")
    print(f"Saved Slim-0.5 comparison: {output_path}")
    print("Synthetic architecture check only; no trained-model quality or latency result.")
    if not result["student_compression_gate"]["passed"]:
        print("FAIL: Student must reduce parameters by >=40% AND supported FLOPs by >=30%. Revise before training.")
        return 1
    print("PASS: Student compression targets met; candidate can proceed to training integration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
