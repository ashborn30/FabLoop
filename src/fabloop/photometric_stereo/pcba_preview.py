"""Qualitative PCBA visualization using explicitly assumed, uncalibrated lights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .diligent_validate import (
    import_cv2, import_rps_class, normal_to_rgb,
    normals_to_height_frankot_chellappa, solve_with_rps,
)
from .preview_alignment import align_lights

LIGHT_ORDER = ("F", "B", "L", "R")
NOMINAL_LIGHT_DIRECTIONS = np.array([[0., -1., 1.], [0., 1., 1.], [-1., 0., 1.], [1., 0., 1.]]) / np.sqrt(2.)
SIGNAL_EPSILON = 1.0e-6
NORMAL_EPSILON = 1.0e-8
MIN_NORMAL_Z = 0.1


def _file_record(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _check_output(board_dir: Path, output_dir: Path) -> None:
    source_roots = [board_dir]
    source_roots.extend(parent for parent in board_dir.parents
                        if parent.name.lower() in {"pcba_4light", "pcba_4light_edited"})
    if any(output_dir.is_relative_to(root) for root in source_roots):
        raise ValueError("Preview output must be outside the source capture tree.")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Preview output must be a new or empty directory; existing files are never overwritten.")


def _render_height(height: np.ndarray, mask: np.ndarray, output_dir: Path) -> dict[str, Any]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    low, high = np.percentile(height[mask], [2, 98])
    if high - low < 1.0e-12:
        low, high = float(low) - .5, float(high) + .5
    figure = Figure(figsize=(8, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot()
    view = axis.imshow(np.ma.masked_invalid(height), cmap="viridis", vmin=low, vmax=high)
    axis.set_title("QUALITATIVE PREVIEW | assumed lighting\nRelative height; display limits P2-P98")
    axis.set_xlabel("x (preview pixels)")
    axis.set_ylabel("y (preview pixels, down)")
    figure.colorbar(view, ax=axis, label="Relative pixel units; not metric depth")
    figure.savefig(output_dir / "height_preview.png", dpi=140)

    rows = np.linspace(0, height.shape[0] - 1, min(150, height.shape[0]), dtype=int)
    columns = np.linspace(0, height.shape[1] - 1, min(150, height.shape[1]), dtype=int)
    x, y = np.meshgrid(columns, rows)
    z = height[np.ix_(rows, columns)]
    span = float(np.ptp(height[mask]))
    exaggeration = min(50., max(1., .15 * max(height.shape) / span)) if span > 1.0e-12 else 1.
    figure = Figure(figsize=(9, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(projection="3d")
    surface = axis.plot_surface(x, y, z, cmap="viridis", vmin=low, vmax=high,
                                rcount=len(rows), ccount=len(columns), linewidth=0, antialiased=True)
    axis.set_title(f"QUALITATIVE PSEUDO-3D | assumed lighting\nZ display exaggeration x{exaggeration:.2f}; not metric geometry")
    axis.set_xlabel("x (preview pixels)")
    axis.set_ylabel("y (preview pixels, down)")
    axis.set_zlabel("Relative pixel units")
    display_z_span = span * exaggeration if span > 1.0e-12 else 1.
    axis.set_box_aspect((max(height.shape[1] - 1, 1), max(height.shape[0] - 1, 1), display_z_span))
    axis.invert_yaxis()
    axis.view_init(elev=35, azim=-65)
    figure.colorbar(surface, ax=axis, shrink=.6, pad=.08, label="Relative height")
    figure.savefig(output_dir / "pseudo3d.png", dpi=140)
    return {"height_color_percentiles": [2, 98], "height_color_limits": [float(low), float(high)],
            "surface_grid_shape": [len(rows), len(columns)], "z_display_exaggeration": exaggeration,
            "raw_height_rescaled": False}


def _render_alignment(images: dict[str, np.ndarray], mask: np.ndarray, output_dir: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 8), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle("UNVERIFIED ALIGNMENT PREVIEW | F reference\nCyan outline: common image overlap, not a board mask")
    for index, key in enumerate(LIGHT_ORDER, start=1):
        axis = figure.add_subplot(2, 2, index)
        axis.imshow(images[key])
        if mask.any() and not mask.all():
            axis.contour(mask.astype(np.uint8), levels=[.5], colors=["cyan"], linewidths=.7)
        axis.set_title(f"Light {key}" + (" (reference)" if key == "F" else " (estimated alignment)"))
        axis.set_axis_off()
    figure.savefig(output_dir / "alignment_preview.png", dpi=130)


def export_preview(board_dir: Path, output_dir: Path, max_edge: int = 1024) -> dict[str, Any]:
    """Export a view-only hypothesis, keeping calibrated processing and training gates separate."""
    board_dir, output_dir = Path(board_dir).resolve(), Path(output_dir).resolve()
    if type(max_edge) is not int or max_edge < 2:
        raise ValueError("max_edge must be an integer of at least 2 pixels.")
    _check_output(board_dir, output_dir)
    images, common_mask, alignment = align_lights(board_dir, max_edge=max_edge)
    if not isinstance(common_mask, np.ndarray) or common_mask.ndim != 2 or common_mask.dtype != np.bool_:
        raise ValueError("Alignment must provide a 2D boolean common-valid mask.")
    if min(common_mask.shape) < 2 or not np.any(common_mask) or set(images) != set(LIGHT_ORDER):
        raise ValueError("Alignment needs all four lights and a nonempty common-valid image region.")
    observations = []
    for key in LIGHT_ORDER:
        image = images[key]
        if image.dtype != np.uint8 or image.shape != (*common_mask.shape, 3):
            raise ValueError(f"Aligned {key} must be an RGB uint8 image matching the common mask.")
        rgb = image.astype(np.float64) / 255.
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        observations.append(linear.mean(axis=2).reshape(-1))
    measurements = np.stack(observations, axis=1)
    signal_ok = measurements.mean(axis=1).reshape(common_mask.shape) > SIGNAL_EPSILON
    solve_mask = common_mask & signal_ok
    if not np.any(solve_mask):
        raise ValueError("No common-valid pixels have enough signal for a qualitative preview.")
    rps_root = Path(__file__).resolve().parents[3] / "third-party/RobustPhotometricStereo"
    normals, elapsed = solve_with_rps(import_rps_class(rps_root), measurements,
                                      NOMINAL_LIGHT_DIRECTIONS, solve_mask, "l2")
    nonzero = np.isfinite(normals).all(axis=2) & (np.linalg.norm(normals, axis=2) > NORMAL_EPSILON)
    stable_z = normals[:, :, 2] > MIN_NORMAL_Z
    valid_mask = solve_mask & nonzero & stable_z
    if not np.any(valid_mask):
        raise ValueError("No pixels pass the preview-only normal and gradient heuristics.")
    normals[~valid_mask] = 0.
    height = normals_to_height_frankot_chellappa(normals, valid_mask, normal_y_axis="down")
    if not np.isfinite(height[valid_mask]).all():
        raise ValueError("Relative-height integration produced non-finite values inside the preview mask.")
    height[~valid_mask] = np.nan
    report: dict[str, Any] = {
        "schema_version": 1, "status": "QUALITATIVE_PREVIEW", "calibrated": False,
        "registration_verified": False, "training_ready": False, "normal_gt_available": False,
        "board_mask_available": False,
        "board_dir": str(board_dir), "output_dir": str(output_dir), "image_shape": list(valid_mask.shape),
        "light_order": list(LIGHT_ORDER), "coordinate_frame": "x_right_y_down_z_towards_camera",
        "assumptions": {
            "directions_source": "Nominal filename-based hypothesis; no measured lighting calibration.",
            "registration_transform_effect": "Image homographies do not calibrate physical light directions; the named nominal vectors remain an unverified hypothesis even after rotations.",
            "light_directions_unit": {key: row.tolist() for key, row in zip(LIGHT_ORDER, NOMINAL_LIGHT_DIRECTIONS)},
            "nominal_light_matrix_rank": int(np.linalg.matrix_rank(NOMINAL_LIGHT_DIRECTIONS)),
            "nominal_light_matrix_condition_number": float(np.linalg.cond(NOMINAL_LIGHT_DIRECTIONS)),
            "light_intensities": "Equal intensities of 1 assumed for every light and RGB channel.",
            "image_linearity": "sRGB assumed; decoded to linear RGB and channel-averaged.",
            "ambient_correction": "Not performed; ambient.jpg is unused.",
            "reflectance": "Diffuse Lambertian approximation; PCBA shadows and reflections can produce artifacts.",
        },
        "alignment": alignment,
        "numerical_valid_mask": {
            "purpose": "Numerical visualization heuristic, not a board mask or ground-truth mask.",
            "signal_mean_min": SIGNAL_EPSILON, "normal_length_min": NORMAL_EPSILON,
            "normal_z_min": MIN_NORMAL_Z, "max_gradient_magnitude_implied": float(np.sqrt(1. - MIN_NORMAL_Z ** 2) / MIN_NORMAL_Z),
            "total_pixels": int(valid_mask.size), "common_overlap_pixels": int(common_mask.sum()),
            "low_signal_excluded": int((common_mask & ~signal_ok).sum()),
            "invalid_normal_excluded": int((solve_mask & ~nonzero).sum()),
            "unstable_z_excluded": int((solve_mask & nonzero & ~stable_z).sum()),
            "valid_pixels": int(valid_mask.sum()),
        },
        "solver": "RobustPhotometricStereo L2 least squares with assumed lighting", "solve_seconds": elapsed,
        "relative_height": {
            "units": "relative_preview_pixel_units", "metric_measurement": False,
            "method": "Frankot-Chellappa; zero median inside valid mask; NaN outside.",
            "limitations": "Periodic FFT and zero outside-mask gradients cause boundary artifacts and remove global tilt; uncalibrated normals do not establish metric shape.",
        },
        "provenance": {
            "source_images": alignment.get("sources", {}),
            "preview_implementation": _file_record(Path(__file__)),
            "alignment_implementation": _file_record(Path(__file__).with_name("preview_alignment.py")),
            "integration_implementation": _file_record(Path(__file__).with_name("diligent_validate.py")),
            "rps_sources": {name: _file_record(rps_root / name) for name in ("rps.py", "rpsnumerics.py", "psutil.py")},
        },
    }
    _check_output(board_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "normal_est.npy", normals)
    np.save(output_dir / "relativeheight.npy", height)
    cv2 = import_cv2()
    for name, rgb in (("normal_rgb.png", normal_to_rgb(normals, valid_mask)), ("reference_rgb.png", images["F"])):
        if not cv2.imwrite(str(output_dir / name), rgb[:, :, ::-1]):
            raise OSError(f"Could not write {name}.")
    if not cv2.imwrite(str(output_dir / "valid_mask.png"), valid_mask.astype(np.uint8) * 255):
        raise OSError("Could not write valid_mask.png.")
    _render_alignment(images, common_mask, output_dir)
    report["display"] = _render_height(height, valid_mask, output_dir)
    report["outputs"] = {path.name: _file_record(path) for path in sorted(output_dir.iterdir()) if path.is_file()}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
