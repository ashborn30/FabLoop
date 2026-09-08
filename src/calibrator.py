"""Normal-only calibration using EfficientAD's official map quantiles."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from model import module_checksum
from trainer import build_from_checkpoint, category_output, save_checkpoint


def calibration_path(config: dict[str, Any], category: str, seed: int) -> Path:
    return category_output(config, category) / "calibration" / f"seed_{seed}.json"


def calibrate(
    config: dict[str, Any],
    category: str,
    seed: int,
    device: torch.device,
    validation_normal_loader: DataLoader,
    validation_normal_ids: list[str],
) -> Path:
    if any(int(item) != 0 for item in validation_normal_loader.dataset.samples["label_index"]):
        raise ValueError("Calibration loader contains non-normal samples")
    wrapper, payload, checkpoint = build_from_checkpoint(config, category, seed, device)
    wrapper.core.eval()
    loader = validation_normal_loader

    calibration_config = config["calibration"]
    lower = float(calibration_config["lower_quantile"])
    upper = float(calibration_config["upper_quantile"])
    if (lower, upper) != (0.9, 0.995):
        raise ValueError("Anomalib EfficientAD 2.6 uses fixed map quantiles 0.9 and 0.995")
    # This official method creates unnormalized local/global maps and computes their quantiles.
    quantiles = wrapper.lightning_model.map_norm_quantiles(loader)
    wrapper.core.quantiles.update(quantiles)

    score_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            local_map, global_map = wrapper.core.get_maps(batch.image.to(device), normalize=True)
            score = float(torch.amax(0.5 * local_map + 0.5 * global_map).detach().cpu())
            score_records.append(
                {"image_path": batch.image_path[0], "image_score": score}
            )
    if len(score_records) != len(validation_normal_ids):
        raise RuntimeError(
            "Calibration score count does not match the validation-normal manifest"
        )
    scores = torch.tensor(
        [record["image_score"] for record in score_records], dtype=torch.float32
    )
    threshold_quantile = float(calibration_config["threshold_quantile"])
    image_threshold = torch.quantile(scores, threshold_quantile).item()

    calibration_ids = sorted(validation_normal_ids)
    values = {name: float(value.detach().cpu()) for name, value in quantiles.items()}
    record = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "seed": seed,
        "source": str(config["data"]["calibration_source"]),
        "sample_count": len(calibration_ids),
        "sample_ids": calibration_ids,
        "test_sample_ids": [],
        "lower_quantile": lower,
        "upper_quantile": upper,
        "local_map": {"qa": values["qa_st"], "qb": values["qb_st"]},
        "global_map": {"qa": values["qa_ae"], "qb": values["qb_ae"]},
        "image_threshold_quantile": threshold_quantile,
        "image_threshold": image_threshold,
        "calibration_normal_score_distribution": {
            "score_definition": "max_final_anomaly_map",
            "sample_count": len(score_records),
            "scores": score_records,
        },
    }
    output_path = calibration_path(config, category, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    payload["model_state_dict"] = {
        name: value.detach().cpu() for name, value in wrapper.core.state_dict().items()
    }
    payload["calibrated"] = True
    payload["calibration"] = record
    payload["teacher_checksum_after"] = module_checksum(wrapper.core.teacher)
    if payload["teacher_checksum_before"] != payload["teacher_checksum_after"]:
        raise RuntimeError("Teacher checksum changed during calibration")
    save_checkpoint(checkpoint, payload)
    return output_path
