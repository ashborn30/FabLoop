# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Slim-0.5 candidate built on the frozen Anomalib EfficientAD-S source.

Only the internal convolution widths of the Student and Autoencoder change.
Their forward methods, normalization, spatial geometry, activation, dropout,
losses, and anomaly-map calculations are inherited without modification from
``torch_model.py``. The feature contract remains Teacher / Student / AE =
384 / 768 / 384, including the Student's two 384-channel groups.

These constructors initialize weights; they do not load pretrained Teacher
weights or a trained checkpoint. Teacher loading and checksum validation belong
to the pipeline that will use this candidate.
"""

from typing import Self

from torch import nn

from .torch_model import (
    AutoEncoder,
    EfficientAdModel,
    EfficientAdModelSize,
    SmallPatchDescriptionNetwork,
)


def _with_channels(layer: nn.Conv2d, in_channels: int, out_channels: int) -> nn.Conv2d:
    """Initialize a convolution with new widths and the original geometry."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=layer.bias is not None,
        padding_mode=layer.padding_mode,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )


def _set_convolution_widths(module: nn.Module, prefix: str, channels: tuple[int, ...]) -> None:
    """Replace only the named convolutions, retaining all other source layers."""
    for index, (in_channels, out_channels) in enumerate(zip(channels, channels[1:]), start=1):
        name = f"{prefix}{index}"
        layer = getattr(module, name)
        if not isinstance(layer, nn.Conv2d):
            raise TypeError(f"Expected {name} to be Conv2d, got {type(layer).__name__}")
        setattr(module, name, _with_channels(layer, in_channels, out_channels))


class SlimStudent(SmallPatchDescriptionNetwork):
    """Candidate Student with widths 3 -> 64 -> 128 -> 128 -> 768.

    The inherited forward retains the original PDN-S normalization, pooling,
    activations, and output geometry. The final width is fixed to 768 so the
    existing Teacher and AE comparisons each receive 384 channels.
    """

    channels = (3, 64, 128, 128, 768)

    def __init__(self, out_channels: int = 768, padding: bool = False) -> None:
        if out_channels != 768:
            raise ValueError("Slim-0.5 Student output must be 768 channels (384 ST + 384 SA).")
        super().__init__(out_channels=out_channels, padding=padding)
        _set_convolution_widths(self, "conv", self.channels)


class SlimAutoEncoder(AutoEncoder):
    """Candidate AE with half-width hidden layers and 384 output channels.

    Encoder, Decoder, and AutoEncoder forward methods remain those of the frozen
    source. All six encoder and eight decoder convolutions retain their original
    geometry, and the original decoder interpolation and dropout are preserved.
    """

    encoder_channels = (3, 16, 16, 32, 32, 32, 32)
    decoder_channels = (32, 32, 32, 32, 32, 32, 32, 32, 384)

    def __init__(self, out_channels: int = 384, padding: bool = False) -> None:
        if out_channels != 384:
            raise ValueError("Slim-0.5 Autoencoder output must be 384 channels.")
        super().__init__(out_channels=out_channels, padding=padding)
        _set_convolution_widths(self.encoder, "enconv", self.encoder_channels)
        _set_convolution_widths(self.decoder, "deconv", self.decoder_channels)


class SlimEfficientAdModel(EfficientAdModel):
    """EfficientAD-S with the experimental Slim-0.5 Student and Autoencoder.

    The Teacher is the original full-width PDN-S, has gradients disabled, and
    remains in evaluation mode when the candidate enters training mode. Its
    weights still need to be loaded from the verified pretrained Teacher before
    training or deployment. Losses and anomaly-map methods are inherited intact.
    """

    architecture_id = "efficientad-s-slim-0.5"
    width_multiplier = 0.5
    candidate = True
    student_channels = SlimStudent.channels
    encoder_channels = SlimAutoEncoder.encoder_channels
    decoder_channels = SlimAutoEncoder.decoder_channels

    def __init__(
        self,
        teacher_out_channels: int = 384,
        padding: bool = False,
        pad_maps: bool = True,
    ) -> None:
        if teacher_out_channels != 384:
            raise ValueError("Slim-0.5 Teacher output must remain 384 channels.")
        super().__init__(
            teacher_out_channels=teacher_out_channels,
            model_size=EfficientAdModelSize.S,
            padding=padding,
            pad_maps=pad_maps,
        )
        self.student = SlimStudent(padding=padding)
        self.ae = SlimAutoEncoder(padding=padding)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True) -> Self:
        """Set Student/AE mode while keeping the frozen Teacher in evaluation."""
        super().train(mode)
        self.teacher.eval()
        return self


__all__ = ["SlimStudent", "SlimAutoEncoder", "SlimEfficientAdModel"]
