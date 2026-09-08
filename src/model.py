"""Thin wrapper around Anomalib's official EfficientAD implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from anomalib.models import EfficientAd

if __package__:
    from .models.efficientad_slim.shape_check import check_feature_shapes
else:
    from models.efficientad_slim.shape_check import check_feature_shapes


def _anomalib_model_size(value: str) -> str:
    aliases = {"small": "small", "s": "small", "medium": "medium", "m": "medium"}
    try:
        return aliases[value.lower()]
    except KeyError as error:
        raise ValueError("model.size must be one of: small, s, medium, m") from error


def module_checksum(module: torch.nn.Module) -> str:
    """Create a deterministic SHA-256 over a module state dict."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class EfficientAdWrapper:
    """Own model lifecycle checks while leaving architecture, maps and losses to Anomalib."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        model_config = config["model"]
        train_config = config["training"]
        self.lightning_model = EfficientAd(
            imagenet_dir=Path(config["imagenette"]["root"]),
            teacher_out_channels=int(model_config["teacher_out_channels"]),
            model_size=_anomalib_model_size(str(model_config["size"])),
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            padding=bool(model_config["padding"]),
            pad_maps=bool(model_config["pad_maps"]),
            pre_processor=False,
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
        self.lightning_model.to(device)
        self.core = self.lightning_model.model
        self.device = device
        self.teacher_checksum: str | None = None

    def load_pretrained_teacher(self) -> str:
        """Use Anomalib's downloader/loader, then freeze the teacher completely."""

        self.lightning_model.to(self.device)
        self.lightning_model.prepare_pretrained_model()
        self.freeze_teacher()
        self.teacher_checksum = module_checksum(self.core.teacher)
        return self.teacher_checksum

    def freeze_teacher(self) -> None:
        self.core.teacher.eval()
        self.core.teacher.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.core.teacher.parameters()):
            raise RuntimeError("EfficientAD teacher is not fully frozen")

    def verify_feature_shapes(self, image_size: tuple[int, int]) -> dict[str, Any]:
        """Require matching Teacher, both Student heads and AE before training."""
        return check_feature_shapes(self.core, image_size=image_size)

    def verify_student_output_channels(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Compatibility entrypoint; now checks the complete feature contract."""
        self.verify_feature_shapes(image_size)
        channels = int(self.core.teacher_out_channels)
        return channels, channels

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.core.student.parameters()) + list(self.core.ae.parameters())

    def enforce_teacher_frozen(self) -> None:
        """Keep teacher frozen/eval after recursive ``train()`` calls."""

        self.core.teacher.eval()
        if any(parameter.requires_grad for parameter in self.core.teacher.parameters()):
            raise RuntimeError("Teacher unexpectedly became trainable")

    def load_core_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.core.load_state_dict(state_dict, strict=True)
        self.freeze_teacher()
        self.teacher_checksum = module_checksum(self.core.teacher)
