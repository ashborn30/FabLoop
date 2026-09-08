"""Export calibrated, registered four-light PCBA normals and relative height."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .diligent_validate import (
    SOLVER_ATTRS, import_cv2, import_rps_class, normal_to_rgb,
    normals_to_height_frankot_chellappa, solve_with_rps,
)

LIGHT_ORDER = ("F", "B", "L", "R")
COORDINATE_FRAME = "x_right_y_down_z_towards_camera"


def _provenance(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _input_path(value: Any, parent: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires an explicit file path.")
    path = Path(value)
    path = (path if path.is_absolute() else parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _matrix(values: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain measured numeric values with shape 4x3.") from error
    if result.shape != (4, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite measured values with shape 4x3.")
    return result


def _load_config(path: Path) -> tuple[dict, list[Path], Path, np.ndarray, np.ndarray]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Capture config requires schema_version=1.")
    for flag in ("registration_verified", "calibrated"):
        if config.get(flag) is not True:
            raise ValueError(f"{flag} must be true after explicit verification.")
    if config.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError(f"coordinate_frame must be {COORDINATE_FRAME!r}.")
    if config.get("image_linearity") not in ("linear", "srgb"):
        raise ValueError("image_linearity must explicitly be 'linear' or 'srgb'.")
    if config.get("light_order") != list(LIGHT_ORDER):
        raise ValueError("light_order must be exactly ['F', 'B', 'L', 'R'].")
    if config.get("ambient_path") is not None:
        raise ValueError("ambient_path is unsupported: supply ambient-corrected or ambient-negligible observations.")
    image_values = config.get("images")
    if not isinstance(image_values, dict) or set(image_values) != set(LIGHT_ORDER):
        raise ValueError("images must contain exactly the four explicit F, B, L, R paths.")
    directions = _matrix(config.get("light_directions"), "light_directions")
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(lengths <= 1.0e-8):
        raise ValueError("light_directions must be nonzero vectors.")
    directions = directions / lengths
    if np.linalg.matrix_rank(directions) != 3:
        raise ValueError("light_directions must have rank 3.")
    intensities = _matrix(config.get("light_intensities"), "light_intensities")
    if np.any(intensities <= 0):
        raise ValueError("light_intensities must be strictly positive, in RGB order.")
    images = [_input_path(image_values[key], path.parent, f"images.{key}") for key in LIGHT_ORDER]
    mask = _input_path(config.get("mask"), path.parent, "mask")
    if len(set([*images, mask])) != 5:
        raise ValueError("Four distinct observations and a separate mask are required.")
    return config, images, mask, directions, intensities


def _read_linear_rgb(path: Path, linearity: str) -> np.ndarray:
    cv2 = import_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Observation must be a readable three-channel RGB image: {path}")
    if image.dtype in (np.uint8, np.uint16):
        rgb = image[:, :, ::-1].astype(np.float64) / np.iinfo(image.dtype).max
    elif np.issubdtype(image.dtype, np.floating):
        rgb = image[:, :, ::-1].astype(np.float64)
    else:
        raise ValueError(f"Unsupported observation dtype {image.dtype}: {path}")
    if not np.all(np.isfinite(rgb)) or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError(f"Observation values must be finite and in normalized range [0, 1]: {path}")
    if linearity == "srgb":
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return rgb


def process_capture(
    capture_config: Path, output_dir: Path, rps_root: Path | None = None, solver: str = "l2",
) -> dict[str, Any]:
    """Validate measured metadata and export PS products; no labels or ground truth needed.

    Images must already be registered and either ambient-corrected or captured
    with negligible ambient illumination. No registration, resizing, exposure
    estimation, or ambient subtraction is performed by this bridge.
    """
    capture_config, output_dir = Path(capture_config).resolve(), Path(output_dir).resolve()
    if solver not in SOLVER_ATTRS:
        raise ValueError(f"Unsupported solver: {solver}")
    config, paths, mask_path, directions, intensities = _load_config(capture_config)
    cv2 = import_cv2()
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None or mask_image.ndim != 2 or not np.all(np.isfinite(mask_image)):
        raise ValueError("mask must be a readable, finite, single-channel image.")
    mask = mask_image != 0
    if not np.any(mask):
        raise ValueError("mask must be nonempty.")
    observations = []
    for path, intensity in zip(paths, intensities):
        image = _read_linear_rgb(path, config["image_linearity"])
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image shape {image.shape[:2]} must match mask shape {mask.shape}: {path}")
        observations.append((image / intensity.reshape(1, 1, 3)).mean(axis=2).reshape(-1))
    measurements = np.stack(observations, axis=1)
    rps_root = (Path(rps_root) if rps_root is not None else
                Path(__file__).resolve().parents[3] / "third-party/RobustPhotometricStereo").resolve()
    RPS = import_rps_class(rps_root)
    normal, elapsed = solve_with_rps(RPS, measurements, directions, mask, solver)
    if not np.all(np.isfinite(normal[mask])) or np.any(np.linalg.norm(normal[mask], axis=1) <= 1.0e-8):
        raise ValueError("Solver produced invalid or zero normals inside mask.")
    if np.any(np.abs(normal[mask, 2]) <= 1.0e-8):
        raise ValueError("Normals with near-zero z cannot produce a finite height gradient.")
    height = normals_to_height_frankot_chellappa(normal, mask)
    if not np.all(np.isfinite(height[mask])):
        raise ValueError("Relative height is not finite inside mask.")
    height[~mask] = np.nan
    names = ("normal_est.npy", "normal_rgb.png", "relativeheight.npy", "report.json")
    if {output_dir / name for name in names} & {capture_config, mask_path, *paths}:
        raise ValueError("Output paths must not overwrite capture inputs.")
    report: dict[str, Any] = {
        "schema_version": 1, "status": "EXPORTED", "light_order": list(LIGHT_ORDER),
        "image_shape": list(mask.shape), "mask_pixels": int(mask.sum()), "solver": solver,
        "solve_seconds": elapsed, "coordinate_frame": COORDINATE_FRAME,
        "registration_verified": True, "calibrated": True,
        "image_linearity": config["image_linearity"], "ambient_correction": "not_performed",
        "ambient_requirement": "Observations must already be ambient-corrected or ambient-negligible.",
        "measurement_preprocessing": "Integer full-scale normalization; sRGB decoding if declared; divide RGB by measured RGB intensities; channel mean.",
        "light_directions_unit": directions.tolist(), "light_intensities_rgb": intensities.tolist(),
        "light_matrix_rank": 3, "light_matrix_condition_number": float(np.linalg.cond(directions)),
        "relative_height": {
            "units": "relative_pixel_units", "metric_measurement": False,
            "method": "Frankot-Chellappa, median inside mask removed; NaN outside mask",
            "boundary_assumption": "Periodic FFT, zero gradients outside mask; boundary artifacts and loss of global tilt are possible.",
        },
        "provenance": {
            "capture_config": _provenance(capture_config), "mask": _provenance(mask_path),
            "images": {key: _provenance(path) for key, path in zip(LIGHT_ORDER, paths)},
            "implementation": _provenance(Path(__file__)),
            "validation_bridge": _provenance(Path(__file__).with_name("diligent_validate.py")),
            "rps_sources": {name: _provenance(rps_root / name) for name in ("rps.py", "rpsnumerics.py", "psutil.py")},
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "normal_est.npy", normal)
    np.save(output_dir / "relativeheight.npy", height)
    if not cv2.imwrite(str(output_dir / "normal_rgb.png"), normal_to_rgb(normal, mask)[:, :, ::-1]):
        raise OSError("Could not write normal_rgb.png.")
    report["outputs"] = {name: _provenance(output_dir / name) for name in names[:-1]}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
