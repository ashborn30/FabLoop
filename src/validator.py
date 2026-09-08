"""Independent leakage, integrity and reload reproducibility checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from data import SplitBundle, dataset_root
from model import module_checksum
from trainer import build_from_checkpoint, category_output, checkpoint_path, load_checkpoint


def _run_check(name: str, callback: Callable[[], tuple[bool, str]]) -> dict[str, Any]:
    try:
        passed, detail = callback()
    except Exception as error:  # validation must report every failed invariant
        return {"name": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"}
    return {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def validate(
    config: dict[str, Any], category: str, seed: int, device: torch.device, splits: SplitBundle
) -> Path:
    checkpoint = load_checkpoint(checkpoint_path(config, category, seed), device)
    checks: list[dict[str, Any]] = []
    sam_checks: list[dict[str, Any]] = []

    checks.append(
        _run_check(
            "train_contains_only_normal",
            lambda: (
                all(row["label"] == "normal" for row in splits.manifest["train"]),
                f"train samples={len(splits.manifest['train'])}",
            ),
        )
    )

    def calibration_isolated() -> tuple[bool, str]:
        record = checkpoint.get("calibration") or {}
        calibration_ids = set(record.get("sample_ids", []))
        manifest_calibration = {row["image"] for row in splits.manifest["calibration"]}
        test_ids = {row["image"] for row in splits.manifest["test"]}
        passed = (
            record.get("source") == config["data"]["calibration_source"]
            and calibration_ids == manifest_calibration
            and calibration_ids.isdisjoint(test_ids)
            and not record.get("test_sample_ids")
        )
        return passed, f"calibration={len(calibration_ids)}, overlap_with_test={len(calibration_ids & test_ids)}"

    checks.append(_run_check("calibration_uses_validation_normal_only", calibration_isolated))

    def calibration_scores_are_isolated() -> tuple[bool, str]:
        record = checkpoint.get("calibration") or {}
        distribution = record.get("calibration_normal_score_distribution") or {}
        score_records = distribution.get("scores") or []
        expected_paths = {
            str((dataset_root(config) / row["image"]).resolve())
            for row in splits.manifest["calibration"]
        }
        actual_paths = {str(Path(item["image_path"]).resolve()) for item in score_records}
        scores_are_finite = all(math.isfinite(float(item["image_score"])) for item in score_records)
        passed = (
            int(distribution.get("sample_count", -1)) == len(score_records)
            and len(score_records) == len(expected_paths)
            and actual_paths == expected_paths
            and scores_are_finite
        )
        return passed, f"calibration_scores={len(score_records)}, expected={len(expected_paths)}"

    checks.append(
        _run_check("calibration_score_distribution_uses_validation_normal_only", calibration_scores_are_isolated)
    )

    def quantiles_present() -> tuple[bool, str]:
        calibration = checkpoint.get("calibration") or {}
        values = [
            calibration.get("local_map", {}).get("qa"),
            calibration.get("local_map", {}).get("qb"),
            calibration.get("global_map", {}).get("qa"),
            calibration.get("global_map", {}).get("qb"),
        ]
        passed = all(value is not None for value in values)
        if passed:
            passed = values[1] > values[0] and values[3] > values[2]
        return passed, f"quantiles={values}"

    checks.append(_run_check("checkpoint_has_valid_quantiles", quantiles_present))

    def teacher_unchanged() -> tuple[bool, str]:
        wrapper, payload, _ = build_from_checkpoint(config, category, seed, device)
        actual = module_checksum(wrapper.core.teacher)
        before = payload.get("teacher_checksum_before")
        after = payload.get("teacher_checksum_after")
        return before == after == actual, f"before={before}, after={after}, loaded={actual}"

    checks.append(_run_check("teacher_checksum_unchanged", teacher_unchanged))

    def masks_match_images() -> tuple[bool, str]:
        checked = 0
        for row in splits.manifest["test"]:
            if row["label"] != "anomaly":
                continue
            image_path = dataset_root(config) / row["image"]
            mask_values = row["mask"] if isinstance(row["mask"], list) else [row["mask"]]
            with Image.open(image_path) as image:
                for mask_value in mask_values:
                    mask_path = dataset_root(config) / mask_value
                    with Image.open(mask_path) as mask:
                        if image.size != mask.size:
                            return False, f"size mismatch: {image_path}={image.size}, {mask_path}={mask.size}"
            checked += 1
        return checked > 0, f"matched anomaly image/mask pairs={checked}"

    checks.append(_run_check("mask_dimensions_match", masks_match_images))

    def reproducible_reload() -> tuple[bool, str]:
        batch = next(iter(splits.calibration_loader(0)))
        image = batch.image.to(device)
        first, _, _ = build_from_checkpoint(config, category, seed, device)
        first.core.eval()
        with torch.inference_mode():
            first_map = first.core(image).anomaly_map.detach().cpu()
        del first
        if device.type == "cuda":
            torch.cuda.empty_cache()
        second, _, _ = build_from_checkpoint(config, category, seed, device)
        second.core.eval()
        with torch.inference_mode():
            second_map = second.core(image).anomaly_map.detach().cpu()
        passed = torch.equal(first_map, second_map) or torch.allclose(first_map, second_map, rtol=0, atol=1e-7)
        max_delta = float(torch.max(torch.abs(first_map - second_map)))
        return passed, f"max_abs_delta={max_delta:.3e}"

    checks.append(_run_check("checkpoint_reload_reproducible", reproducible_reload))

    if bool((config.get("sam") or {}).get("enabled", False)):
        prediction_dir = category_output(config, category) / "predictions" / f"seed_{seed}"

        def sam_call_accounting() -> tuple[bool, str]:
            metrics_path = prediction_dir / "metrics.json"
            samples_path = prediction_dir / "samples.jsonl"
            if not metrics_path.is_file() or not samples_path.is_file():
                return False, "SAM metrics or samples artifact is missing"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in samples_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            candidate_rows = [row for row in rows if len((row.get("sam") or {}).get("boxes", [])) > 0]
            refined_rows = [row for row in rows if bool((row.get("sam") or {}).get("refined"))]
            calls = int(metrics.get("sam_refine_calls", -1))
            prompts_valid = all(
                len(row["sam"].get("argmax_points", [])) == len(row["sam"]["boxes"])
                and all(
                    x1 <= point_x < x2 and y1 <= point_y < y2
                    for (x1, y1, x2, y2), (point_x, point_y) in zip(
                        row["sam"]["boxes"], row["sam"]["argmax_points"], strict=True
                    )
                )
                for row in candidate_rows
            )
            def crop_geometry_is_valid(row: dict[str, Any]) -> bool:
                geometry = row["sam"].get("geometry") or {}
                crop = geometry.get("crop_xyxy_original") or []
                original = geometry.get("original_resolution") or []
                crop_resolution = geometry.get("crop_resolution") or []
                if (
                    geometry.get("mode") != "original_resolution_union_crop"
                    or len(crop) != 4
                    or len(original) != 2
                    or len(crop_resolution) != 2
                ):
                    return False
                x1, y1, x2, y2 = (int(value) for value in crop)
                original_height, original_width = (int(value) for value in original)
                return (
                    0 <= x1 < x2 <= original_width
                    and 0 <= y1 < y2 <= original_height
                    and crop_resolution == [y2 - y1, x2 - x1]
                )

            crops_valid = all(crop_geometry_is_valid(row) for row in candidate_rows)
            passed = (
                metrics.get("sam_enabled") is True
                and metrics.get("sam_threshold_source") == "deployment"
                and float(metrics.get("sam_threshold")) == float(metrics.get("threshold_deployment"))
                and calls == len(candidate_rows) == len(refined_rows)
                and prompts_valid
                and crops_valid
                and all(bool(row["sam"]["flagged"]) for row in refined_rows)
                and all(row["sam"].get("mask_path") for row in refined_rows)
                and all(
                    not (row.get("sam") or {}).get("refined")
                    for row in rows
                    if not (row.get("sam") or {}).get("boxes")
                )
            )
            return passed, (
                f"calls={calls}, candidate_images={len(candidate_rows)}, "
                f"refined_images={len(refined_rows)}, prompts_valid={prompts_valid}, "
                f"crops_valid={crops_valid}"
            )

        sam_checks.append(_run_check("sam_only_called_for_candidate_boxes", sam_call_accounting))

        def sam_masks_match_evaluation_resolution() -> tuple[bool, str]:
            metrics_path = prediction_dir / "metrics.json"
            samples_path = prediction_dir / "samples.jsonl"
            if not metrics_path.is_file() or not samples_path.is_file():
                return False, "SAM metrics or samples artifact is missing"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            expected_height, expected_width = (
                int(value) for value in metrics["sam_output_resolution"]
            )
            checked = 0
            for line in samples_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                mask_path = (row.get("sam") or {}).get("mask_path")
                if not mask_path:
                    continue
                with Image.open(mask_path) as mask:
                    if mask.size != (expected_width, expected_height):
                        return False, f"size mismatch: {mask_path}={mask.size}"
                checked += 1
            return checked == int(metrics.get("sam_refine_calls", -1)), f"checked_masks={checked}"

        sam_checks.append(
            _run_check("sam_mask_dimensions_match_evaluation_resolution", sam_masks_match_evaluation_resolution)
        )

    passed = all(check["status"] == "PASS" for check in checks)
    sam_passed = all(check["status"] == "PASS" for check in sam_checks) if sam_checks else None
    result = {
        "category": category,
        "seed": seed,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "sam_status": "PASS" if sam_passed else "FAIL" if sam_passed is False else "DISABLED",
        "sam_checks": sam_checks,
    }
    output = category_output(config, category) / f"validator_seed_{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output
