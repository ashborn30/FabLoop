"""Single-pass calibrated EfficientAD evaluation and artifact export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from anomalib.metrics import AUPRO
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from calibrator import calibration_path
from data import dataset_name
from sam_refiner import SamRefiner
from trainer import build_from_checkpoint, category_output


LOCO_ANOMALY_GROUPS = ("logical_anomalies", "structural_anomalies")


def _safe_id(image_path: str, label: int, anomaly_group: str | None = None) -> str:
    if anomaly_group is not None:
        return f"{anomaly_group}_{Path(image_path).stem}"
    return f"{'anomaly' if label else 'normal'}_{Path(image_path).stem}"


def _to_uint8_map(value: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(finite.shape, dtype=np.uint8)
    return np.clip((finite - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _save_visualizations(
    directory: Path,
    sample_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    local_map: np.ndarray,
    global_map: np.ndarray,
    final_map: np.ndarray,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / f"{sample_id}_input.png"
    mask_path = directory / f"{sample_id}_mask.png"
    local_path = directory / f"{sample_id}_local.png"
    global_path = directory / f"{sample_id}_global.png"
    final_path = directory / f"{sample_id}_final.png"
    overlay_path = directory / f"{sample_id}_overlay.png"
    Image.fromarray(image).save(input_path)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(mask_path)
    local_vis = cv2.applyColorMap(_to_uint8_map(local_map), cv2.COLORMAP_JET)
    global_vis = cv2.applyColorMap(_to_uint8_map(global_map), cv2.COLORMAP_JET)
    final_vis = cv2.applyColorMap(_to_uint8_map(final_map), cv2.COLORMAP_JET)
    Image.fromarray(cv2.cvtColor(local_vis, cv2.COLOR_BGR2RGB)).save(local_path)
    Image.fromarray(cv2.cvtColor(global_vis, cv2.COLOR_BGR2RGB)).save(global_path)
    heat_rgb = cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.6 * image + 0.4 * heat_rgb, 0, 255).astype(np.uint8)
    Image.fromarray(cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB)).save(final_path)
    Image.fromarray(overlay).save(overlay_path)
    return {
        "input": str(input_path),
        "mask": str(mask_path),
        "local_map": str(local_path),
        "global_map": str(global_path),
        "final_map": str(final_path),
        "overlay": str(overlay_path),
    }


def _confusion_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int | float]:
    predictions = scores >= threshold
    anomaly = labels == 1
    normal = labels == 0
    tp = int(np.logical_and(predictions, anomaly).sum())
    fp = int(np.logical_and(predictions, normal).sum())
    tn = int(np.logical_and(~predictions, normal).sum())
    fn = int(np.logical_and(~predictions, anomaly).sum())
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": tp / (tp + fn),
        "fpr": fp / (fp + tn),
    }


def _benchmark_operating_point(
    labels: np.ndarray, scores: np.ndarray, target_fpr: float, policy: str
) -> dict[str, int | float]:
    if policy != "max_recall_under_target_fpr":
        raise ValueError("evaluation.threshold_policy must be max_recall_under_target_fpr")
    fpr, recall, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= target_fpr + np.finfo(np.float64).eps)
    if eligible.size == 0:
        raise RuntimeError(f"ROC curve has no operating point at or below target FPR={target_fpr}")
    best_recall = float(np.max(recall[eligible]))
    candidates = eligible[np.isclose(recall[eligible], best_recall, rtol=0.0, atol=1e-12)]
    best_fpr = float(np.min(fpr[candidates]))
    candidates = candidates[np.isclose(fpr[candidates], best_fpr, rtol=0.0, atol=1e-12)]
    finite_candidates = candidates[np.isfinite(thresholds[candidates])]
    if finite_candidates.size:
        selected = int(finite_candidates[np.argmax(thresholds[finite_candidates])])
        threshold = float(thresholds[selected])
    else:
        threshold = float(np.nextafter(np.max(scores), np.inf))
    return _confusion_at_threshold(labels, scores, threshold)


def _loco_anomaly_group(image_path: str) -> str | None:
    """Recover the official LOCO anomaly group from its image path."""

    path_parts = image_path.replace("\\", "/").split("/")
    return next((group for group in LOCO_ANOMALY_GROUPS if group in path_parts), None)


def _loco_subset_recall_at_threshold(
    rows: list[dict[str, Any]], threshold: float
) -> dict[str, dict[str, int | float]]:
    """Measure each LOCO anomaly group without selecting another threshold."""

    results: dict[str, dict[str, int | float]] = {}
    for group in LOCO_ANOMALY_GROUPS:
        subset = [row for row in rows if row.get("anomaly_group") == group]
        if not subset:
            raise RuntimeError(f"MVTec LOCO test split has no {group} samples")
        tp = sum(float(row["score"]) >= threshold for row in subset)
        fn = len(subset) - tp
        results[group] = {
            "samples": len(subset),
            "tp": tp,
            "fn": fn,
            "recall": tp / len(subset),
        }
    return results


def _deployment_threshold(calibration_record: dict[str, Any], target_fpr: float, policy: str) -> float:
    if policy != "percentile_of_calibration_normal":
        raise ValueError(
            "evaluation.deployment_threshold_policy must be percentile_of_calibration_normal"
        )
    distribution = calibration_record.get("calibration_normal_score_distribution") or {}
    records = distribution.get("scores") or []
    scores = np.asarray([record["image_score"] for record in records], dtype=np.float64)
    if scores.size == 0 or scores.size != int(distribution.get("sample_count", -1)):
        raise RuntimeError(
            "Calibration-normal score distribution is missing; rerun the calibrate stage first"
        )
    return float(np.quantile(scores, 1.0 - target_fpr, method="linear"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _binary_mask_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction_mask = prediction.astype(bool)
    target_mask = target.astype(bool)
    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            f"Prediction/target mask shape mismatch: {prediction_mask.shape} != {target_mask.shape}"
        )
    intersection = int(np.logical_and(prediction_mask, target_mask).sum())
    prediction_area = int(prediction_mask.sum())
    target_area = int(target_mask.sum())
    union = prediction_area + target_area - intersection
    dice_denominator = prediction_area + target_area
    dice = 1.0 if dice_denominator == 0 else 2.0 * intersection / dice_denominator
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _sam_metric_summary(
    rows: list[dict[str, Any]],
    refiner: SamRefiner,
    threshold: float,
    mask_dir: Path,
    tiny_ratio: float,
) -> dict[str, Any]:
    anomaly_rows = [row for row in rows if int(row["label"]) == 1]
    detected_rows = [row for row in anomaly_rows if row["sam"]["refined"]]
    summary: dict[str, Any] = {
        "sam_enabled": True,
        "sam_threshold_source": "deployment",
        "sam_threshold": float(threshold),
        "sam_checkpoint": str(refiner.checkpoint),
        "sam_checkpoint_sha256": refiner.checkpoint_sha256,
        "sam_multimask_output": False,
        "sam_minimum_component_area": refiner.minimum_component_area,
        "sam_tiny_defect_area_ratio": tiny_ratio,
        "sam_flagged_images": sum(row["sam"]["flagged"] for row in rows),
        "sam_candidate_images": sum(bool(row["sam"]["boxes"]) for row in rows),
        "sam_refine_calls": refiner.refine_calls,
        "sam_total_boxes": sum(len(row["sam"]["boxes"]) for row in rows),
        "sam_detected_anomaly_samples": len(detected_rows),
        "sam_anomaly_samples": len(anomaly_rows),
        "sam_mask_dice_detected": _mean_or_none(
            [float(row["sam"]["dice_detected"]) for row in detected_rows]
        ),
        "sam_mask_iou_detected": _mean_or_none(
            [float(row["sam"]["iou_detected"]) for row in detected_rows]
        ),
        "sam_mask_dice_end_to_end": _mean_or_none(
            [float(row["sam"]["dice_end_to_end"]) for row in anomaly_rows]
        ),
        "sam_mask_iou_end_to_end": _mean_or_none(
            [float(row["sam"]["iou_end_to_end"]) for row in anomaly_rows]
        ),
        "sam_masks_dir": str(mask_dir),
    }
    for group in ("tiny", "larger"):
        group_anomalies = [
            row for row in anomaly_rows if row["sam"]["defect_size_group"] == group
        ]
        group_detected = [row for row in group_anomalies if row["sam"]["refined"]]
        summary[f"sam_anomaly_{group}_samples"] = len(group_anomalies)
        summary[f"sam_detected_{group}_samples"] = len(group_detected)
        summary[f"sam_mask_dice_detected_{group}"] = _mean_or_none(
            [float(row["sam"]["dice_detected"]) for row in group_detected]
        )
        summary[f"sam_mask_iou_detected_{group}"] = _mean_or_none(
            [float(row["sam"]["iou_detected"]) for row in group_detected]
        )
        summary[f"sam_mask_dice_end_to_end_{group}"] = _mean_or_none(
            [float(row["sam"]["dice_end_to_end"]) for row in group_anomalies]
        )
        summary[f"sam_mask_iou_end_to_end_{group}"] = _mean_or_none(
            [float(row["sam"]["iou_end_to_end"]) for row in group_anomalies]
        )
    return summary


def evaluate(
    config: dict[str, Any], category: str, seed: int, device: torch.device, test_loader: DataLoader
) -> Path:
    wrapper, checkpoint, _ = build_from_checkpoint(config, category, seed, device)
    if not checkpoint.get("calibrated") or not checkpoint.get("calibration"):
        raise RuntimeError("Evaluation requires a calibrated checkpoint")
    wrapper.core.eval()
    threshold_default = float(checkpoint["calibration"]["image_threshold"])
    evaluation_config = config["evaluation"]
    is_mvtec_loco = dataset_name(config) == "MVTecLoco"
    target_fpr = float(evaluation_config["target_fpr"])
    if not np.isclose(target_fpr, 0.10):
        raise ValueError("The requested fpr_10 output schema requires evaluation.target_fpr: 0.10")

    calibration_artifact = calibration_path(config, category, seed)
    if not calibration_artifact.is_file():
        raise FileNotFoundError(f"Calibration artifact not found: {calibration_artifact}")
    calibration_record = json.loads(calibration_artifact.read_text(encoding="utf-8"))
    if calibration_record.get("category") != category or int(calibration_record.get("seed", -1)) != seed:
        raise ValueError("Calibration artifact provenance does not match the evaluation run")
    threshold_deployment = _deployment_threshold(
        calibration_record,
        target_fpr,
        str(evaluation_config["deployment_threshold_policy"]),
    )

    output = category_output(config, category)
    prediction_dir = output / "predictions" / f"seed_{seed}"
    map_dir = prediction_dir / "maps"
    sam_mask_dir = prediction_dir / "sam_masks"
    visual_dir = output / "visualizations" / f"seed_{seed}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    sam_config = config.get("sam") or {}
    sam_enabled = bool(sam_config.get("enabled", False))
    if sam_enabled and dataset_name(config) != "Visa":
        raise ValueError("SAM refinement is enabled only for the VisA PCB baseline")
    sam_refiner = SamRefiner(sam_config, device) if sam_enabled else None
    tiny_defect_ratio = float(sam_config.get("tiny_defect_area_ratio", 0.002))
    if sam_enabled:
        if not 0.0 < tiny_defect_ratio < 1.0:
            raise ValueError("sam.tiny_defect_area_ratio must be between 0 and 1")
        sam_mask_dir.mkdir(parents=True, exist_ok=True)

    image_scores: list[float] = []
    image_labels: list[int] = []
    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    aupro = AUPRO(
        fields=["anomaly_map", "gt_mask"],
        fpr_limit=float(config["evaluation"]["aupro_fpr_limit"]),
    )
    rows: list[dict[str, Any]] = []

    # This is the only traversal/inference pass over the official test split.
    with torch.inference_mode():
        progress = tqdm(
            test_loader,
            desc=f"Evaluate {category} seed={seed}",
            unit="image",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for batch in progress:
            image_tensor = batch.image.to(device, non_blocking=True)
            local_tensor, global_tensor = wrapper.core.get_maps(image_tensor, normalize=True)
            final_tensor = 0.5 * local_tensor + 0.5 * global_tensor
            score = float(torch.amax(final_tensor).detach().cpu())
            label = int(batch.gt_label.item())
            mask_tensor = batch.gt_mask.to(torch.uint8).cpu()
            final_cpu = final_tensor[:, 0].detach().cpu()
            aupro.update(batch.update(anomaly_map=final_cpu, gt_mask=mask_tensor))

            local_map = local_tensor[0, 0].detach().cpu().numpy()
            global_map = global_tensor[0, 0].detach().cpu().numpy()
            final_map = final_cpu[0].numpy()
            mask = mask_tensor[0].numpy()
            image = (
                batch.image[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy()
            )
            image_path = str(batch.image_path[0])
            anomaly_group = _loco_anomaly_group(image_path) if is_mvtec_loco and label == 1 else None
            sample_id = _safe_id(image_path, label, anomaly_group)
            raw_map_path = map_dir / f"{sample_id}.npz"
            np.savez_compressed(
                raw_map_path,
                local_map=local_map,
                global_map=global_map,
                final_anomaly_map=final_map,
                ground_truth_mask=mask,
            )
            visuals = {}
            if bool(config["evaluation"].get("save_visualizations", True)):
                visuals = _save_visualizations(
                    visual_dir, sample_id, image, mask, local_map, global_map, final_map
                )
            sam_record: dict[str, Any] | None = None
            if sam_refiner is not None:
                flagged = score >= threshold_deployment
                boxes, points = (
                    sam_refiner.prompts_from_map(final_map, threshold_deployment)
                    if flagged
                    else ([], [])
                )
                refined = bool(boxes)
                refined_mask = np.zeros(mask.shape, dtype=np.uint8)
                sam_mask_path: str | None = None
                sam_geometry: dict[str, Any] | None = None
                if refined:
                    with Image.open(image_path) as original:
                        original_image = np.asarray(original.convert("RGB"), dtype=np.uint8)
                    refined_mask, sam_geometry = sam_refiner.refine_original_crop(
                        original_image,
                        boxes,
                        points,
                        final_map.shape,
                    )
                    output_mask = sam_mask_dir / f"{sample_id}.png"
                    Image.fromarray(refined_mask * 255).save(output_mask)
                    sam_mask_path = str(output_mask)
                    progress.set_postfix(sam_calls=sam_refiner.refine_calls, refresh=False)
                defect_area_ratio = float((mask > 0).mean()) if label == 1 else None
                defect_size_group = (
                    "tiny" if defect_area_ratio is not None and defect_area_ratio < tiny_defect_ratio else "larger"
                ) if label == 1 else None
                dice_end_to_end: float | None = None
                iou_end_to_end: float | None = None
                dice_detected: float | None = None
                iou_detected: float | None = None
                if label == 1:
                    dice_end_to_end, iou_end_to_end = _binary_mask_metrics(refined_mask, mask)
                    if refined:
                        dice_detected, iou_detected = dice_end_to_end, iou_end_to_end
                sam_record = {
                    "threshold_source": "deployment",
                    "threshold": threshold_deployment,
                    "flagged": flagged,
                    "boxes": boxes,
                    "argmax_points": points,
                    "geometry": sam_geometry,
                    "refined": refined,
                    "mask_path": sam_mask_path,
                    "defect_area_ratio": defect_area_ratio,
                    "defect_size_group": defect_size_group,
                    "dice_detected": dice_detected,
                    "iou_detected": iou_detected,
                    "dice_end_to_end": dice_end_to_end,
                    "iou_end_to_end": iou_end_to_end,
                }
            row = {
                "sample_id": sample_id,
                "image_path": image_path,
                "label": label,
                "score": score,
                "image_score": score,
                "prediction": int(score >= threshold_default),
                "raw_maps": str(raw_map_path),
                "visualizations": visuals,
            }
            if sam_record is not None:
                row["sam"] = sam_record
            if is_mvtec_loco:
                row["anomaly_group"] = anomaly_group
            rows.append(row)
            image_scores.append(score)
            image_labels.append(label)
            pixel_scores.append(final_map.reshape(-1))
            pixel_labels.append(mask.reshape(-1))

    image_scores_array = np.asarray(image_scores, dtype=np.float64)
    image_labels_array = np.asarray(image_labels, dtype=np.uint8)
    pixel_scores_array = np.concatenate(pixel_scores)
    pixel_labels_array = np.concatenate(pixel_labels).astype(np.uint8)
    default = _confusion_at_threshold(image_labels_array, image_scores_array, threshold_default)
    benchmark = _benchmark_operating_point(
        image_labels_array,
        image_scores_array,
        target_fpr,
        str(evaluation_config["threshold_policy"]),
    )
    benchmark_loco_subsets = (
        _loco_subset_recall_at_threshold(rows, float(benchmark["threshold"]))
        if is_mvtec_loco
        else {}
    )
    deployment = _confusion_at_threshold(
        image_labels_array, image_scores_array, threshold_deployment
    )

    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    for row in rows:
        score = float(row["score"])
        label = int(row["label"])
        row["prediction_default"] = int(score >= threshold_default)
        row["prediction_at_fpr_10"] = int(score >= float(benchmark["threshold"]))
        row["prediction_deployment"] = int(score >= threshold_deployment)
        if label == 1 and row["prediction_at_fpr_10"] == 0:
            false_negatives.append({**row, "error_type": "false_negative_at_fpr_10"})
        elif label == 0 and row["prediction_at_fpr_10"] == 1:
            false_positives.append({**row, "error_type": "false_positive_at_fpr_10"})

    false_negative_path = prediction_dir / "false_negatives_at_fpr_10.jsonl"
    false_positive_path = prediction_dir / "false_positives_at_fpr_10.jsonl"
    _write_jsonl(false_negative_path, false_negatives)
    _write_jsonl(false_positive_path, false_positives)
    operating_point_path = prediction_dir / "operating_point_at_fpr_10.json"
    operating_point_record = {
        "category": category,
        "seed": seed,
        "policy": evaluation_config["threshold_policy"],
        "target_fpr": target_fpr,
        "recall": benchmark["recall"],
        "actual_fpr": benchmark["fpr"],
        "threshold": benchmark["threshold"],
        "tp": benchmark["tp"],
        "fp": benchmark["fp"],
        "tn": benchmark["tn"],
        "fn": benchmark["fn"],
        "false_negatives": str(false_negative_path),
        "false_positives": str(false_positive_path),
    }
    if benchmark_loco_subsets:
        operating_point_record["anomaly_subsets"] = benchmark_loco_subsets
    operating_point_path.write_text(
        json.dumps(
            operating_point_record,
            indent=2,
        ),
        encoding="utf-8",
    )
    deployment_path = prediction_dir / "deployment_threshold.json"
    deployment_path.write_text(
        json.dumps(
            {
                "category": category,
                "seed": seed,
                "policy": evaluation_config["deployment_threshold_policy"],
                "source": str(calibration_artifact),
                "target_fpr": target_fpr,
                "percentile": 1.0 - target_fpr,
                "threshold_deployment": threshold_deployment,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "seed": seed,
        "checkpoint_iteration": checkpoint["iteration"],
        "test_passes": 1,
        "test_samples": len(rows),
        "image_level_auroc": float(roc_auc_score(image_labels_array, image_scores_array)),
        "pixel_level_auroc": float(roc_auc_score(pixel_labels_array, pixel_scores_array)),
        "aupro": float(aupro.compute().cpu()),
        "recall_default": default["recall"],
        "false_positive_rate_default": default["fpr"],
        "threshold_default": default["threshold"],
        "threshold_source_default": "validation_normal_legacy_quantile",
        "recall": default["recall"],
        "false_positive_rate": default["fpr"],
        "image_threshold": default["threshold"],
        "threshold_source": "validation_normal",
        "benchmark_threshold_policy": evaluation_config["threshold_policy"],
        "target_fpr": target_fpr,
        "recall_at_fpr_10": benchmark["recall"],
        "actual_fpr_at_fpr_10": benchmark["fpr"],
        "threshold_at_fpr_10": benchmark["threshold"],
        "tp_at_fpr_10": benchmark["tp"],
        "fp_at_fpr_10": benchmark["fp"],
        "tn_at_fpr_10": benchmark["tn"],
        "fn_at_fpr_10": benchmark["fn"],
        "false_negatives_at_fpr_10": str(false_negative_path),
        "false_positives_at_fpr_10": str(false_positive_path),
        "deployment_threshold_policy": evaluation_config["deployment_threshold_policy"],
        "threshold_deployment": deployment["threshold"],
        "recall_deployment": deployment["recall"],
        "fpr_actual_deployment": deployment["fpr"],
        "tp_deployment": deployment["tp"],
        "fp_deployment": deployment["fp"],
        "tn_deployment": deployment["tn"],
        "fn_deployment": deployment["fn"],
        "deployment_threshold_artifact": str(deployment_path),
        "maps_dir": str(map_dir),
        "visualizations_dir": str(visual_dir),
    }
    for group, subset in benchmark_loco_subsets.items():
        metrics[f"recall_at_fpr_10_{group}"] = subset["recall"]
        metrics[f"tp_at_fpr_10_{group}"] = subset["tp"]
        metrics[f"fn_at_fpr_10_{group}"] = subset["fn"]
        metrics[f"samples_at_fpr_10_{group}"] = subset["samples"]
    if sam_refiner is not None:
        metrics.update(
            _sam_metric_summary(
                rows,
                sam_refiner,
                threshold_deployment,
                sam_mask_dir,
                tiny_defect_ratio,
            )
        )
        metrics["sam_input_mode"] = "original_resolution_union_crop"
        metrics["sam_input_resolution"] = [int(value) for value in config["data"]["image_size"]]
        metrics["sam_prompt_resolution"] = [int(value) for value in config["data"]["image_size"]]
        metrics["sam_output_resolution"] = [int(value) for value in config["data"]["image_size"]]
        metrics["sam_crop_margin_map_pixels"] = sam_refiner.crop_margin_map_pixels
        metrics["sam_processor_resolution"] = int(sam_config.get("processor_resolution", 1008))
    _write_jsonl(prediction_dir / "samples.jsonl", rows)
    metrics_path = prediction_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path
