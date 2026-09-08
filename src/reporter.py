"""Aggregate PCB category/seed results and evaluate the reproduction gate."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from data import report_categories
from trainer import category_output


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build_reports(config: dict[str, Any]) -> tuple[Path, bool]:
    root = Path(config["output_dir"])
    categories = report_categories(config)
    gate_config = config["reproduction_gate"]
    required_seeds = {int(seed) for seed in gate_config.get("required_seeds", [42, 43, 44])}
    category_reports: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for category in categories:
        category_dir = category_output(config, category)
        runs: list[dict[str, Any]] = []
        for metrics_path in sorted(category_dir.glob("predictions/seed_*/metrics.json")):
            metrics = _read_json(metrics_path)
            if not metrics:
                continue
            seed = int(metrics["seed"])
            benchmark = _read_json(category_dir / f"benchmark_seed_{seed}.json")
            validator = _read_json(category_dir / f"validator_seed_{seed}.json")
            run = {"seed": seed, "metrics": metrics, "benchmark": benchmark, "validator": validator}
            runs.append(run)
            summary_rows.append(
                {
                    "row_type": "seed_run",
                    "category": category,
                    "seed": seed,
                    "image_level_auroc": metrics["image_level_auroc"],
                    "image_level_auroc_mean": "",
                    "image_level_auroc_std": "",
                    "image_level_auroc_std_scope": "",
                    "pixel_level_auroc": metrics["pixel_level_auroc"],
                    "aupro": metrics["aupro"],
                    "recall": metrics.get("recall_default", metrics.get("recall", "")),
                    "false_positive_rate": metrics.get(
                        "false_positive_rate_default", metrics.get("false_positive_rate", "")
                    ),
                    "recall_at_fpr_10": metrics.get("recall_at_fpr_10", ""),
                    "actual_fpr_at_fpr_10": metrics.get("actual_fpr_at_fpr_10", ""),
                    "threshold_at_fpr_10": metrics.get("threshold_at_fpr_10", ""),
                    "tp_at_fpr_10": metrics.get("tp_at_fpr_10", ""),
                    "fp_at_fpr_10": metrics.get("fp_at_fpr_10", ""),
                    "tn_at_fpr_10": metrics.get("tn_at_fpr_10", ""),
                    "fn_at_fpr_10": metrics.get("fn_at_fpr_10", ""),
                    "recall_deployment": metrics.get("recall_deployment", ""),
                    "fpr_actual_deployment": metrics.get("fpr_actual_deployment", ""),
                    "threshold_deployment": metrics.get("threshold_deployment", ""),
                    "sam_mask_dice_detected": metrics.get("sam_mask_dice_detected", ""),
                    "sam_mask_iou_detected": metrics.get("sam_mask_iou_detected", ""),
                    "sam_mask_dice_end_to_end": metrics.get("sam_mask_dice_end_to_end", ""),
                    "sam_mask_iou_end_to_end": metrics.get("sam_mask_iou_end_to_end", ""),
                    "sam_mask_dice_detected_tiny": metrics.get("sam_mask_dice_detected_tiny", ""),
                    "sam_mask_iou_detected_tiny": metrics.get("sam_mask_iou_detected_tiny", ""),
                    "sam_mask_dice_detected_larger": metrics.get("sam_mask_dice_detected_larger", ""),
                    "sam_mask_iou_detected_larger": metrics.get("sam_mask_iou_detected_larger", ""),
                    "sam_candidate_images": metrics.get("sam_candidate_images", ""),
                    "sam_refine_calls": metrics.get("sam_refine_calls", ""),
                    "sam_masks": metrics.get("sam_masks_dir", ""),
                    "sam_validator": validator.get("sam_status", "MISSING") if validator else "MISSING",
                    "latency_ms": benchmark["latency_ms_mean"] if benchmark else "",
                    "peak_gpu_memory_mb": benchmark["peak_gpu_memory_mb"] if benchmark else "",
                    "anomaly_maps": metrics["maps_dir"],
                    "validator": validator["status"] if validator else "MISSING",
                }
            )
        aurocs = [float(run["metrics"]["image_level_auroc"]) for run in runs]
        category_report = {
            "category": category,
            "runs": runs,
            "seed_count": len(runs),
            "required_seeds": sorted(required_seeds),
            "missing_required_seeds": sorted(required_seeds - {run["seed"] for run in runs}),
            "image_level_auroc_mean": statistics.fmean(aurocs) if aurocs else None,
            "image_level_auroc_std": statistics.pstdev(aurocs) if aurocs else None,
            "anomaly_map_paths": [run["metrics"]["maps_dir"] for run in runs],
        }
        category_reports[category] = category_report
        benchmark_runs = [run["benchmark"] for run in runs if run["benchmark"]]
        summary_rows.append(
            {
                "row_type": "category_aggregate",
                "category": category,
                "seed": "ALL",
                "image_level_auroc": "",
                "image_level_auroc_mean": category_report["image_level_auroc_mean"] or "",
                "image_level_auroc_std": category_report["image_level_auroc_std"] or "",
                "image_level_auroc_std_scope": "across_seeds",
                "pixel_level_auroc": "",
                "aupro": "",
                "recall": "",
                "false_positive_rate": "",
                "latency_ms": (
                    statistics.fmean(item["latency_ms_mean"] for item in benchmark_runs)
                    if benchmark_runs
                    else ""
                ),
                "peak_gpu_memory_mb": (
                    statistics.fmean(item["peak_gpu_memory_mb"] for item in benchmark_runs)
                    if benchmark_runs
                    else ""
                ),
                "anomaly_maps": ";".join(category_report["anomaly_map_paths"]),
                "validator": "PASS" if runs and all(run["validator"] and run["validator"]["status"] == "PASS" for run in runs) else "FAIL",
            }
        )
        category_dir.mkdir(parents=True, exist_ok=True)
        (category_dir / "report.json").write_text(json.dumps(category_report, indent=2), encoding="utf-8")

    completed = [report for report in category_reports.values() if report["seed_count"] > 0]
    means = [float(report["image_level_auroc_mean"]) for report in completed]
    overall_category_mean_std = statistics.pstdev(means) if means else None
    max_seed_std = max((float(report["image_level_auroc_std"]) for report in completed), default=float("inf"))
    validators_pass = bool(completed) and all(
        run["validator"] and run["validator"].get("status") == "PASS"
        for report in completed
        for run in report["runs"]
    )
    teacher_checks_pass = bool(completed) and all(
        any(
            check["name"] == "teacher_checksum_unchanged" and check["status"] == "PASS"
            for check in run["validator"]["checks"]
        )
        for report in completed
        for run in report["runs"]
        if run["validator"]
    )
    calibration_checks_pass = bool(completed) and all(
        any(
            check["name"] == "calibration_uses_validation_normal_only" and check["status"] == "PASS"
            for check in run["validator"]["checks"]
        )
        for report in completed
        for run in report["runs"]
        if run["validator"]
    )
    required_seeds_present = len(completed) == len(categories) and all(
        required_seeds.issubset({run["seed"] for run in report["runs"]})
        for report in completed
    )
    checks = {
        "all_pcb_categories_present": len(completed) == len(categories),
        "required_seeds_present": required_seeds_present,
        "mean_image_level_auroc": bool(means) and statistics.fmean(means) >= float(gate_config["mean_image_auroc_min"]),
        "no_category_below_minimum": len(completed) == len(categories)
        and all(value >= float(gate_config["category_image_auroc_min"]) for value in means),
        "seed_standard_deviation": required_seeds_present
        and max_seed_std <= float(gate_config["seed_std_max"]),
        "teacher_checksum": teacher_checks_pass,
        "validation_normal_calibration": calibration_checks_pass,
        "anomaly_map_visual_review": (
            bool(gate_config.get("visual_review_pass")) if gate_config.get("require_visual_review", True) else True
        ),
        "validator": validators_pass,
    }
    gate_pass = all(checks.values())
    gate_report = {
        "status": "PASS" if gate_pass else "FAIL",
        "checks": checks,
        "mean_image_level_auroc": statistics.fmean(means) if means else None,
        "overall_category_mean_std": overall_category_mean_std,
        "max_category_seed_std": max_seed_std if completed else None,
        "required_seeds": sorted(required_seeds),
        "categories": category_reports,
        "note": "The 2 ms / 600 images/s paper figures are references, not an embedded-device requirement.",
    }
    root.mkdir(parents=True, exist_ok=True)
    gate_path = root / "reproduction_gate.json"
    gate_path.write_text(json.dumps(gate_report, indent=2), encoding="utf-8")

    summary_rows.append(
        {
            "row_type": "overall",
            "category": "ALL",
            "seed": "ALL",
            "image_level_auroc": "",
            "image_level_auroc_mean": statistics.fmean(means) if means else "",
            "image_level_auroc_std": overall_category_mean_std if completed else "",
            "image_level_auroc_std_scope": "across_category_means",
            "pixel_level_auroc": "",
            "aupro": "",
            "recall": "",
            "false_positive_rate": "",
            "latency_ms": "",
            "peak_gpu_memory_mb": "",
            "anomaly_maps": "",
            "validator": "PASS" if validators_pass else "FAIL",
        }
    )

    summary_path = root / "summary.csv"
    fieldnames = [
        "row_type",
        "category",
        "seed",
        "image_level_auroc",
        "image_level_auroc_mean",
        "image_level_auroc_std",
        "image_level_auroc_std_scope",
        "pixel_level_auroc",
        "aupro",
        "recall",
        "false_positive_rate",
        "recall_at_fpr_10",
        "actual_fpr_at_fpr_10",
        "threshold_at_fpr_10",
        "tp_at_fpr_10",
        "fp_at_fpr_10",
        "tn_at_fpr_10",
        "fn_at_fpr_10",
        "recall_deployment",
        "fpr_actual_deployment",
        "threshold_deployment",
        "sam_mask_dice_detected",
        "sam_mask_iou_detected",
        "sam_mask_dice_end_to_end",
        "sam_mask_iou_end_to_end",
        "sam_mask_dice_detected_tiny",
        "sam_mask_iou_detected_tiny",
        "sam_mask_dice_detected_larger",
        "sam_mask_iou_detected_larger",
        "sam_candidate_images",
        "sam_refine_calls",
        "sam_masks",
        "sam_validator",
        "latency_ms",
        "peak_gpu_memory_mb",
        "anomaly_maps",
        "validator",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return gate_path, gate_pass
