"""Thin frozen SAM3 wrapper for box-and-point anomaly mask refinement."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def isolate_board_classical(
    original_image: np.ndarray,
    border_width: int = 10,
    background_std_factor: float = 4.0,
    background_value_std_factor: float = 8.0,
    closing_kernel_size: int = 15,
) -> np.ndarray:
    """Isolate the largest non-background region using adaptive HSV border statistics."""

    if original_image.ndim != 3 or original_image.shape[2] != 3:
        raise ValueError(f"Expected an original RGB HWC image, got {original_image.shape}")
    height, width = original_image.shape[:2]
    if border_width < 1 or 2 * border_width >= min(height, width):
        raise ValueError("border_width must fit inside the image")
    if background_std_factor <= 0:
        raise ValueError("background_std_factor must be positive")
    if background_value_std_factor <= 0:
        raise ValueError("background_value_std_factor must be positive")
    if closing_kernel_size < 1:
        raise ValueError("closing_kernel_size must be positive")
    if closing_kernel_size % 2 == 0:
        closing_kernel_size += 1

    rgb = np.ascontiguousarray(original_image, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    border_samples = np.concatenate(
        [
            hsv[:border_width].reshape(-1, 3),
            hsv[-border_width:].reshape(-1, 3),
            hsv[border_width:-border_width, :border_width].reshape(-1, 3),
            hsv[border_width:-border_width, -border_width:].reshape(-1, 3),
        ],
        axis=0,
    )
    background_mean = border_samples.mean(axis=0)
    background_std = np.maximum(border_samples.std(axis=0), 1.0)
    channel_factors = np.asarray(
        [background_std_factor, background_std_factor, background_value_std_factor],
        dtype=np.float32,
    )
    lower = background_mean - channel_factors * background_std
    upper = background_mean + channel_factors * background_std
    background = np.all((hsv >= lower) & (hsv <= upper), axis=2)
    board_candidate = np.ascontiguousarray(~background, dtype=np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_kernel_size, closing_kernel_size),
    )
    closed = cv2.morphologyEx(board_candidate, cv2.MORPH_CLOSE, kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    if component_count <= 1:
        return np.zeros((height, width), dtype=np.uint8)
    largest_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_component).astype(np.uint8)


class SamRefiner:
    """Load official SAM3 once and refine all boxes for one image in one call."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        if not bool(config.get("enabled", False)):
            raise ValueError("SamRefiner requires sam.enabled: true")
        if str(config.get("threshold_source")) != "deployment":
            raise ValueError("sam.threshold_source must be deployment")
        if bool(config.get("multimask_output", True)):
            raise ValueError("sam.multimask_output must be false for deterministic box prompts")

        checkpoint = Path(config["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.minimum_component_area = int(config.get("minimum_component_area", 5))
        if self.minimum_component_area < 1:
            raise ValueError("sam.minimum_component_area must be at least 1")
        self.crop_margin_map_pixels = int(config.get("crop_margin_map_pixels", 16))
        if self.crop_margin_map_pixels < 0:
            raise ValueError("sam.crop_margin_map_pixels cannot be negative")
        self.refine_calls = 0
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = _file_sha256(checkpoint)

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.model = build_sam3_image_model(
            device=str(device),
            eval_mode=True,
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            compile=False,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("SAM3 must be frozen completely")
        self.processor = Sam3Processor(
            self.model,
            resolution=int(config.get("processor_resolution", 1008)),
            device=str(device),
        )

    def prompts_from_map(
        self, anomaly_map: np.ndarray, threshold: float
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Create one XYXY box and one in-component argmax point per component."""

        if anomaly_map.ndim != 2:
            raise ValueError(f"Expected a 2D anomaly map, got {anomaly_map.shape}")
        binary = np.ascontiguousarray(anomaly_map >= float(threshold), dtype=np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        boxes: list[list[int]] = []
        points: list[list[int]] = []
        for component in range(1, component_count):
            x, y, width, height, area = (int(value) for value in stats[component])
            if area < self.minimum_component_area:
                continue
            component_mask = labels == component
            component_coordinates = np.argwhere(component_mask)
            point_y, point_x = component_coordinates[
                int(np.argmax(anomaly_map[component_mask]))
            ]
            boxes.append([x, y, x + width, y + height])
            points.append([int(point_x), int(point_y)])
        return boxes, points

    def boxes_from_map(self, anomaly_map: np.ndarray, threshold: float) -> list[list[int]]:
        """Return component boxes for callers that do not need point prompts."""

        boxes, _ = self.prompts_from_map(anomaly_map, threshold)
        return boxes

    def refine(
        self,
        image: np.ndarray,
        boxes: list[list[int]],
        points: list[list[int]],
    ) -> np.ndarray:
        """Run one batched box-and-point SAM3 call and union its instance masks."""

        if not boxes:
            raise ValueError("SAM3 refine requires at least one candidate box")
        if len(points) != len(boxes):
            raise ValueError("SAM3 refine requires exactly one point per candidate box")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB HWC image, got {image.shape}")
        height, width = image.shape[:2]
        for box, point in zip(boxes, points, strict=True):
            x1, y1, x2, y2 = box
            point_x, point_y = point
            if not (0 <= x1 <= point_x < x2 <= width and 0 <= y1 <= point_y < y2 <= height):
                raise ValueError(f"SAM3 argmax point {point} must be inside its box {box}")
        rgb = np.ascontiguousarray(image, dtype=np.uint8)
        box_array = np.asarray(boxes, dtype=np.float32)
        point_array = np.asarray(points, dtype=np.float32)[:, None, :]
        point_labels = np.ones((len(points), 1), dtype=np.int32)
        with torch.inference_mode():
            state = self.processor.set_image(Image.fromarray(rgb, mode="RGB"))
            masks, _, _ = self.model.predict_inst(
                state,
                point_coords=point_array,
                point_labels=point_labels,
                box=box_array,
                multimask_output=False,
            )
        masks = np.asarray(masks)
        if masks.shape[0] != len(boxes) or masks.shape[-2:] != image.shape[:2]:
            raise RuntimeError(
                f"SAM3 returned masks {masks.shape} for {len(boxes)} boxes and image {image.shape}"
            )
        if masks.ndim == 4 and masks.shape[1] == 1:
            instance_masks = masks[:, 0]
        elif masks.ndim == 3:
            instance_masks = masks
        else:
            raise RuntimeError(f"Unexpected SAM3 mask shape: {masks.shape}")
        self.refine_calls += 1
        return np.any(instance_masks > 0.5, axis=0).astype(np.uint8)

    def refine_original_crop(
        self,
        original_image: np.ndarray,
        boxes: list[list[int]],
        points: list[list[int]],
        prompt_shape: tuple[int, int],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Refine prompts on an original-resolution union crop and return to prompt space."""

        if original_image.ndim != 3 or original_image.shape[2] != 3:
            raise ValueError(f"Expected an original RGB HWC image, got {original_image.shape}")
        if not boxes or len(boxes) != len(points):
            raise ValueError("Original-resolution refinement requires one point per box")
        prompt_height, prompt_width = (int(value) for value in prompt_shape)
        if prompt_height < 1 or prompt_width < 1:
            raise ValueError(f"Invalid prompt shape: {prompt_shape}")

        original_height, original_width = original_image.shape[:2]
        scale_x = original_width / prompt_width
        scale_y = original_height / prompt_height
        boxes_original = [
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            for x1, y1, x2, y2 in boxes
        ]
        points_original = [
            [(point_x + 0.5) * scale_x, (point_y + 0.5) * scale_y]
            for point_x, point_y in points
        ]

        margin_x = self.crop_margin_map_pixels * scale_x
        margin_y = self.crop_margin_map_pixels * scale_y
        crop_x1 = max(0, int(np.floor(min(box[0] for box in boxes_original) - margin_x)))
        crop_y1 = max(0, int(np.floor(min(box[1] for box in boxes_original) - margin_y)))
        crop_x2 = min(
            original_width,
            int(np.ceil(max(box[2] for box in boxes_original) + margin_x)),
        )
        crop_y2 = min(
            original_height,
            int(np.ceil(max(box[3] for box in boxes_original) + margin_y)),
        )
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise RuntimeError(f"Invalid original-resolution crop: {[crop_x1, crop_y1, crop_x2, crop_y2]}")

        crop = np.ascontiguousarray(
            original_image[crop_y1:crop_y2, crop_x1:crop_x2], dtype=np.uint8
        )
        boxes_crop = [
            [x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1]
            for x1, y1, x2, y2 in boxes_original
        ]
        points_crop = [
            [point_x - crop_x1, point_y - crop_y1]
            for point_x, point_y in points_original
        ]
        crop_mask = self.refine(crop, boxes_crop, points_crop)

        original_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        original_mask[crop_y1:crop_y2, crop_x1:crop_x2] = crop_mask
        output_mask = cv2.resize(
            original_mask,
            (prompt_width, prompt_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
        geometry = {
            "mode": "original_resolution_union_crop",
            "prompt_resolution": [prompt_height, prompt_width],
            "original_resolution": [original_height, original_width],
            "crop_xyxy_original": [crop_x1, crop_y1, crop_x2, crop_y2],
            "crop_resolution": [crop_y2 - crop_y1, crop_x2 - crop_x1],
            "boxes_original": boxes_original,
            "argmax_points_original": points_original,
        }
        return output_mask, geometry

    def isolate_board(self, original_image: np.ndarray) -> np.ndarray:
        """Segment the board with a positive center and four negative corner points."""

        if original_image.ndim != 3 or original_image.shape[2] != 3:
            raise ValueError(f"Expected an original RGB HWC image, got {original_image.shape}")
        height, width = original_image.shape[:2]
        rgb = np.ascontiguousarray(original_image, dtype=np.uint8)
        point_coords = np.asarray(
            [
                [width / 2.0, height / 2.0],
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [0.0, height - 1.0],
                [width - 1.0, height - 1.0],
            ],
            dtype=np.float32,
        )
        point_labels = np.asarray([1, 0, 0, 0, 0], dtype=np.int32)
        with torch.inference_mode():
            state = self.processor.set_image(Image.fromarray(rgb, mode="RGB"))
            masks, _, _ = self.model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=False,
            )
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[:2] == (1, 1):
            board_mask = masks[0, 0]
        elif masks.ndim == 3 and masks.shape[0] == 1:
            board_mask = masks[0]
        else:
            raise RuntimeError(f"Unexpected SAM3 board mask shape: {masks.shape}")
        if board_mask.shape != (height, width):
            raise RuntimeError(
                f"SAM3 returned board mask {board_mask.shape} for image {original_image.shape}"
            )
        return board_mask.astype(np.uint8)
