"""Mandatory raw-feature shape gate for the Slim EfficientAD candidate."""

import torch
from torch import nn


def check_feature_shapes(
    core: nn.Module, image_size: tuple[int, int] = (256, 256)
) -> dict:
    """Check the 384 / 768 / 384 feature contract without changing model state.

    This uses raw features, so it needs neither pretrained weights nor Teacher
    calibration. It validates subtraction compatibility; it does not change or
    replace the original training losses or anomaly-map calculations.
    """
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(type(size) is not int or size <= 0 for size in image_size)
    ):
        raise ValueError("image_size must be a tuple of two positive integers.")
    if getattr(core, "teacher_out_channels", None) != 384:
        raise RuntimeError("Shape gate requires teacher_out_channels=384.")
    try:
        reference = next(core.teacher.parameters())
    except (AttributeError, StopIteration) as error:
        raise RuntimeError("Shape gate requires a Teacher with parameters.") from error

    modes = [(module, module.training) for module in core.modules()]
    parameters = [(parameter, parameter.requires_grad) for parameter in core.parameters()]
    state = [(tensor, tensor.detach().clone()) for tensor in (*core.parameters(), *core.buffers())]
    cuda_devices = [reference.device.index] if reference.device.type == "cuda" else []
    input_shape = [1, 3, *image_size]
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            core.eval()
            dummy = torch.zeros(input_shape, device=reference.device, dtype=reference.dtype)
            features = {
                "teacher": core.teacher(dummy),
                "student": core.student(dummy),
                "autoencoder": core.ae(dummy, image_size),
            }
            for name, feature in features.items():
                if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                    raise RuntimeError(f"{name} must return a four-dimensional Tensor.")
            teacher_shape = tuple(features["teacher"].shape)
            if teacher_shape[:2] != (1, 384) or min(teacher_shape[2:]) <= 0:
                raise RuntimeError(f"Teacher shape must be [1, 384, H, W]; got {teacher_shape}.")
            student_shape = (1, 768, *teacher_shape[2:])
            if tuple(features["student"].shape) != student_shape:
                raise RuntimeError(
                    f"Student shape must be {student_shape}; got {tuple(features['student'].shape)}."
                )
            features["student_teacher"] = features["student"][:, :384]
            features["student_autoencoder"] = features["student"][:, 384:]
            for name, feature in features.items():
                expected = student_shape if name == "student" else teacher_shape
                if tuple(feature.shape) != expected:
                    raise RuntimeError(f"{name} shape must be {expected}; got {tuple(feature.shape)}.")
                if not torch.isfinite(feature).all().item():
                    raise RuntimeError(f"{name} contains non-finite features.")
            # All exact shapes are checked first: broadcasting cannot pass this gate.
            differences = {
                "T-S_T": features["teacher"] - features["student_teacher"],
                "T-A": features["teacher"] - features["autoencoder"],
                "A-S_A": features["autoencoder"] - features["student_autoencoder"],
            }
            for name, difference in differences.items():
                if not torch.isfinite(difference).all().item():
                    raise RuntimeError(f"{name} contains non-finite differences.")
            return {
                "status": "PASS",
                "input_shape": input_shape,
                "features": {name: list(value.shape) for name, value in features.items()},
                "differences": {name: list(value.shape) for name, value in differences.items()},
                "finite": True,
                "device": str(reference.device),
                "dtype": str(reference.dtype),
                "fixed_groups": {
                    "teacher": 384, "student_teacher": 384, "student_autoencoder": 384, "autoencoder": 384,
                },
                "description": "Raw-feature shape/subtraction gate; no calibration or loss rewrite.",
            }
    except Exception as error:
        raise RuntimeError(f"EfficientAD shape gate failed: {error}") from error
    finally:
        with torch.no_grad():
            for tensor, value in state:
                if not torch.equal(tensor, value):
                    tensor.copy_(value)
            for parameter, requires_grad in parameters:
                parameter.requires_grad_(requires_grad)
        # Assign directly to preserve mixed Teacher-eval / Student-train modes.
        for module, training in modes:
            module.training = training
