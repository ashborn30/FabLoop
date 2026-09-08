"""Normal-map relighting utilities for lightweight pseudo-3D previews."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


NORMAL_CANDIDATES = (
    "normal_est.npy",
    "normal_l2_nominal.npy",
    "normal.npy",
    "normal_est_rgb.png",
    "normal_rgb.png",
    "normal_l2_nominal.png",
)
ALBEDO_CANDIDATES = (
    "albedo.npy",
    "albedo_l2_nominal.npy",
    "albedo_l2_nominal.png",
    "albedo.png",
    "reference_rgb.png",
)


@dataclass(frozen=True)
class RelightCase:
    """Files needed to relight one board result folder."""

    name: str
    folder: Path
    normal_path: Path
    albedo_path: Path


@dataclass(frozen=True)
class RelightInputs:
    """Loaded arrays for one pseudo-3D relighting case."""

    case: RelightCase
    normals: np.ndarray
    albedo: np.ndarray


def natural_key(value: str) -> list[int | str]:
    parts = re.split(r"(\d+)", value.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def normalize_normals(normals: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Return per-pixel unit normals and keep zero vectors at zero."""

    values = np.asarray(normals, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"normal map must have shape HxWx3, got {values.shape}")
    lengths = np.linalg.norm(values, axis=2, keepdims=True)
    result = np.zeros_like(values, dtype=np.float32)
    np.divide(values, lengths, out=result, where=lengths > eps)
    return result


def light_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Build a unit light vector in x-right/y-down/z-camera image coordinates."""

    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    xy = np.cos(elevation)
    vector = np.array(
        [
            xy * np.cos(azimuth),
            xy * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float32,
    )
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-8:
        raise ValueError("light direction must be nonzero")
    return vector / length


def render_relight(
    normals: np.ndarray,
    albedo: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Render max(0, normal dot L) * albedo as an RGB uint8 image."""

    normal_values = normalize_normals(normals)
    albedo_values = np.asarray(albedo, dtype=np.float32)
    if albedo_values.ndim == 2:
        albedo_rgb = np.repeat(albedo_values[:, :, None], 3, axis=2)
    elif albedo_values.ndim == 3 and albedo_values.shape[2] in (1, 3, 4):
        albedo_rgb = albedo_values[:, :, :3]
        if albedo_rgb.shape[2] == 1:
            albedo_rgb = np.repeat(albedo_rgb, 3, axis=2)
    else:
        raise ValueError(f"albedo must have shape HxW or HxWxC, got {albedo_values.shape}")
    if albedo_rgb.shape[:2] != normal_values.shape[:2]:
        raise ValueError(
            f"normal/albedo shapes must match, got {normal_values.shape[:2]} and {albedo_rgb.shape[:2]}"
        )
    albedo_rgb = np.clip(albedo_rgb, 0.0, 1.0)
    light = light_direction(azimuth_deg, elevation_deg)
    shading = np.maximum(0.0, np.einsum("hwc,c->hw", normal_values, light))
    rendered = albedo_rgb * shading[:, :, None]
    return np.clip(rendered * 255.0, 0.0, 255.0).astype(np.uint8)


def load_normal_map(path: Path) -> np.ndarray:
    """Load a float normal map from .npy or an RGB-encoded normal image."""

    path = Path(path)
    if path.suffix.lower() == ".npy":
        return normalize_normals(np.load(path))
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return normalize_normals(image * 2.0 - 1.0)


def load_albedo(path: Path) -> np.ndarray:
    """Load albedo as normalized float grayscale or RGB."""

    path = Path(path)
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim == 3 and values.shape[2] == 4:
            values = values[:, :, :3]
        if np.issubdtype(raw.dtype, np.integer):
            values = values / float(np.iinfo(raw.dtype).max)
        elif values.max(initial=0.0) > 1.0:
            finite_positive = values[np.isfinite(values) & (values > 0)]
            high = float(np.percentile(finite_positive, 99.5)) if finite_positive.size else 1.0
            if high > 1.5:
                values = values / high
        return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    image = Image.open(path)
    if image.mode in ("L", "I;16", "I", "F"):
        values = np.asarray(image, dtype=np.float32)
        maximum = float(values.max(initial=0.0))
        if maximum > 1.0:
            values = values / (65535.0 if maximum > 255.0 else 255.0)
        return np.clip(values, 0.0, 1.0)
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def resize_for_display(values: np.ndarray, max_edge: int | None) -> np.ndarray:
    """Downsample large arrays for responsive slider updates."""

    if max_edge is None or max_edge <= 0:
        return values
    height, width = values.shape[:2]
    edge = max(height, width)
    if edge <= max_edge:
        return values
    scale = float(max_edge) / float(edge)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    try:
        import cv2

        return cv2.resize(values, size, interpolation=cv2.INTER_AREA)
    except ImportError:
        if values.ndim == 2:
            image = Image.fromarray(values.astype(np.float32), mode="F")
            return np.asarray(image.resize(size, resample=Image.Resampling.BOX), dtype=values.dtype)
        channels = [
            np.asarray(
                Image.fromarray(values[:, :, channel].astype(np.float32), mode="F").resize(
                    size, resample=Image.Resampling.BOX
                ),
                dtype=values.dtype,
            )
            for channel in range(values.shape[2])
        ]
        return np.stack(channels, axis=2)


def load_relight_inputs(case: RelightCase, max_edge: int | None = None) -> RelightInputs:
    normals = resize_for_display(load_normal_map(case.normal_path), max_edge)
    albedo = resize_for_display(load_albedo(case.albedo_path), max_edge)
    if albedo.shape[:2] != normals.shape[:2]:
        albedo = resize_to_shape(albedo, normals.shape[:2])
    return RelightInputs(case=case, normals=normalize_normals(normals), albedo=np.clip(albedo, 0.0, 1.0))


def resize_to_shape(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize albedo/reference images to a normal-map shape."""

    target_height, target_width = shape
    try:
        import cv2

        return cv2.resize(values, (target_width, target_height), interpolation=cv2.INTER_AREA)
    except ImportError:
        max_value = float(values.max(initial=0.0))
        scaled = values if max_value <= 1.0 else values / max_value
        if scaled.ndim == 2:
            image = Image.fromarray(np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
            resized = image.resize((target_width, target_height), resample=Image.Resampling.BOX)
            return np.asarray(resized, dtype=np.float32) / 255.0
        image = Image.fromarray(np.clip(scaled[:, :, :3] * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
        resized = image.resize((target_width, target_height), resample=Image.Resampling.BOX)
        return np.asarray(resized, dtype=np.float32) / 255.0


def find_first(folder: Path, candidates: Iterable[str]) -> Path | None:
    names = {path.name.lower(): path for path in folder.iterdir() if path.is_file()}
    for candidate in candidates:
        found = names.get(candidate.lower())
        if found is not None:
            return found
    return None


def discover_relight_cases(results_root: Path) -> list[RelightCase]:
    """Find board result folders containing normal and albedo-like files."""

    root = Path(results_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"results root does not exist: {root}")
    cases: list[RelightCase] = []
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        normal_path = find_first(folder, NORMAL_CANDIDATES)
        albedo_path = find_first(folder, ALBEDO_CANDIDATES)
        if normal_path is None or albedo_path is None:
            continue
        cases.append(
            RelightCase(
                name=folder.name,
                folder=folder,
                normal_path=normal_path,
                albedo_path=albedo_path,
            )
        )
    return sorted(cases, key=lambda case: natural_key(case.name))


def write_relight_summary(path: Path, cases: list[RelightCase]) -> None:
    records = [
        {
            "name": case.name,
            "folder": str(case.folder),
            "normal_path": str(case.normal_path),
            "albedo_path": str(case.albedo_path),
        }
        for case in cases
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
