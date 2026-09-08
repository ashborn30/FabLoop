"""Training lifecycle for one official Anomalib EfficientAD model."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import dataset_name
from model import EfficientAdWrapper, module_checksum


def checkpoint_path(config: dict[str, Any], category: str, seed: int) -> Path:
    return category_output(config, category) / "checkpoints" / f"seed_{seed}.ckpt"


def category_output(config: dict[str, Any], category: str) -> Path:
    root = Path(config["output_dir"])
    return root / category if dataset_name(config) == "Visa" else root


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def environment_info(device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in ("anomalib", "torch", "torchvision", "lightning", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def build_from_checkpoint(
    config: dict[str, Any], category: str, seed: int, device: torch.device
) -> tuple[EfficientAdWrapper, dict[str, Any], Path]:
    path = checkpoint_path(config, category, seed)
    payload = load_checkpoint(path, device)
    if payload.get("category") != category or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Checkpoint provenance does not match category={category}, seed={seed}")
    wrapper = EfficientAdWrapper(config, device)
    wrapper.load_core_state(payload["model_state_dict"])
    expected_checksum = payload.get("teacher_checksum_before")
    if expected_checksum and wrapper.teacher_checksum != expected_checksum:
        raise RuntimeError("Teacher checksum in checkpoint does not match loaded Teacher")
    return wrapper, payload, path


def _checkpoint_payload(
    *,
    wrapper: EfficientAdWrapper,
    config: dict[str, Any],
    category: str,
    seed: int,
    iteration: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    manifest: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    checksum_after = module_checksum(wrapper.core.teacher)
    return {
        "format_version": 1,
        "category": category,
        "seed": seed,
        "iteration": iteration,
        "config": config,
        "model_state_dict": {name: value.detach().cpu() for name, value in wrapper.core.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "teacher_checksum_before": wrapper.teacher_checksum,
        "teacher_checksum_after": checksum_after,
        "student_output_groups": [wrapper.core.teacher_out_channels, wrapper.core.teacher_out_channels],
        "split_provenance": {
            "official_csv": manifest["official_csv"],
            "official_csv_sha256": manifest["official_csv_sha256"],
            "split_seed": manifest["split_seed"],
            "calibration_source": manifest["calibration_source"],
        },
        "calibrated": False,
        "calibration": None,
        "environment": environment,
    }


def train(
    config: dict[str, Any],
    category: str,
    seed: int,
    device: torch.device,
    train_loader: DataLoader,
    manifest: dict[str, Any],
) -> Path:
    """Train only Student and AE through Anomalib's official loss implementation."""

    train_config = config["training"]
    if int(train_config["batch_size"]) != 1:
        raise ValueError("EfficientAD training.batch_size must be 1")
    if str(train_config["optimizer"]).lower() != "adam":
        raise ValueError("EfficientAD pipeline supports the required Adam optimizer only")
    seed_everything(seed, bool(config.get("deterministic", True)))

    output = category_output(config, category)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = checkpoint_dir / f"config_seed_{seed}.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    environment = environment_info(device)
    (checkpoint_dir / f"environment_seed_{seed}.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    wrapper = EfficientAdWrapper(config, device)
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    shape_report_path = checkpoint_dir / f"shape_check_seed_{seed}.json"
    try:
        shape_report = wrapper.verify_feature_shapes(image_size)
    except (RuntimeError, ValueError) as error:
        shape_report_path.write_text(
            json.dumps({"status": "FAIL", "error": str(error)}, indent=2), encoding="utf-8"
        )
        raise
    shape_report_path.write_text(json.dumps(shape_report, indent=2), encoding="utf-8")
    wrapper.load_pretrained_teacher()
    wrapper.lightning_model.prepare_imagenette_data(image_size)
    mean_std = wrapper.lightning_model.teacher_channel_mean_std(train_loader)
    wrapper.core.mean_std.update(mean_std)

    optimizer = torch.optim.Adam(
        wrapper.trainable_parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler_config = train_config["scheduler"]
    if scheduler_config["type"] != "StepLR":
        raise ValueError("training.scheduler.type must be StepLR")
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(scheduler_config["step_size"]),
        gamma=float(scheduler_config["gamma"]),
    )

    log_path = checkpoint_dir / f"losses_seed_{seed}.csv"
    iterations = int(train_config["iterations"])
    checkpoint_interval = int(train_config["checkpoint_interval"])
    progress_refresh_interval = int(train_config.get("progress_refresh_interval", 10))
    if progress_refresh_interval < 1:
        raise ValueError("training.progress_refresh_interval must be at least 1")
    train_iterator = iter(train_loader)
    imagenette_iterator = iter(wrapper.lightning_model.imagenet_loader)
    wrapper.core.train()
    wrapper.enforce_teacher_frozen()

    with log_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["iteration", "local_loss", "autoencoder_loss", "student_autoencoder_loss", "total_loss", "lr"],
        )
        writer.writeheader()
        progress = tqdm(
            range(1, iterations + 1),
            desc=f"Train {category} seed={seed}",
            unit="iter",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for iteration in progress:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            try:
                imagenette = next(imagenette_iterator)[0]
            except StopIteration:
                imagenette_iterator = iter(wrapper.lightning_model.imagenet_loader)
                imagenette = next(imagenette_iterator)[0]

            image = batch.image.to(device, non_blocking=True)
            imagenette = imagenette.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Architecture, hard-feature loss, AE loss, ST-AE loss and penalty are all Anomalib-owned.
            local_loss, ae_loss, stae_loss = wrapper.core(image, batch_imagenet=imagenette)
            total_loss = local_loss + ae_loss + stae_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            wrapper.enforce_teacher_frozen()

            local_value = float(local_loss.detach())
            ae_value = float(ae_loss.detach())
            stae_value = float(stae_loss.detach())
            total_value = float(total_loss.detach())
            learning_rate = float(optimizer.param_groups[0]["lr"])
            writer.writerow(
                {
                    "iteration": iteration,
                    "local_loss": local_value,
                    "autoencoder_loss": ae_value,
                    "student_autoencoder_loss": stae_value,
                    "total_loss": total_value,
                    "lr": learning_rate,
                }
            )
            if iteration == 1 or iteration % progress_refresh_interval == 0:
                progress.set_postfix(
                    local=f"{local_value:.5f}",
                    ae=f"{ae_value:.5f}",
                    stae=f"{stae_value:.5f}",
                    total=f"{total_value:.5f}",
                    lr=f"{learning_rate:.2e}",
                    refresh=False,
                )
            if iteration % 100 == 0:
                stream.flush()
            if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
                periodic_checkpoint = checkpoint_dir / f"seed_{seed}_iter_{iteration}.ckpt"
                save_checkpoint(
                    periodic_checkpoint,
                    _checkpoint_payload(
                        wrapper=wrapper,
                        config=config,
                        category=category,
                        seed=seed,
                        iteration=iteration,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        manifest=manifest,
                        environment=environment,
                    ),
                )
                progress.write(f"Saved checkpoint: {periodic_checkpoint}")

    final_path = checkpoint_path(config, category, seed)
    payload = _checkpoint_payload(
        wrapper=wrapper,
        config=config,
        category=category,
        seed=seed,
        iteration=iterations,
        optimizer=optimizer,
        scheduler=scheduler,
        manifest=manifest,
        environment=environment,
    )
    if payload["teacher_checksum_before"] != payload["teacher_checksum_after"]:
        raise RuntimeError("Teacher changed during training; checkpoint was not accepted")
    save_checkpoint(final_path, payload)
    return final_path
