from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.fft import fft2, fftfreq, ifft2
from scipy.io import loadmat


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
KNOWN_DILIGENT_OBJECTS = (
    "ball",
    "cat",
    "pot1",
    "bear",
    "pot2",
    "buddha",
    "goblet",
    "reading",
    "cow",
    "harvest",
)
SOLVER_ATTRS = {
    "l2": "L2_SOLVER",
    "l1": "L1_SOLVER",
    "l1-multicore": "L1_SOLVER_MULTICORE",
}


@dataclass(frozen=True)
class DiligentSample:
    object_name: str
    object_dir: Path
    image_paths: list[Path]
    light_directions: np.ndarray
    light_intensities: np.ndarray | None
    mask: np.ndarray
    normal_gt: np.ndarray


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_baseline_path() -> Path:
    return repo_root() / "references" / "diligent_main_l2_baseline.csv"


def normalize_object_name(name: str) -> str:
    compact = "".join(ch for ch in name.lower() if ch.isalnum())
    for known in sorted(KNOWN_DILIGENT_OBJECTS, key=len, reverse=True):
        if known in compact:
            return known
    return compact


def natural_key(path: Path) -> list[Any]:
    import re

    chunks = re.split(r"(\d+)", path.name.lower())
    return [int(chunk) if chunk.isdigit() else chunk for chunk in chunks]


def find_file_case_insensitive(folder: Path, candidates: Sequence[str]) -> Path | None:
    by_name = {path.name.lower(): path for path in folder.iterdir() if path.is_file()}
    for candidate in candidates:
        found = by_name.get(candidate.lower())
        if found is not None:
            return found
    return None


def read_filenames(object_dir: Path) -> list[Path]:
    filenames_path = find_file_case_insensitive(object_dir, ["filenames.txt"])
    if filenames_path is not None:
        image_paths = []
        for line in filenames_path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name:
                continue
            path = Path(name)
            image_paths.append(path if path.is_absolute() else object_dir / path)
        return image_paths

    excluded = {"mask.png", "normal_gt.png"}
    image_paths = [
        path
        for path in object_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name.lower() not in excluded
    ]
    return sorted(image_paths, key=natural_key)


def coerce_n_by_3_vectors(
    values: np.ndarray, *, name: str, expected_count: int | None = None
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")

    candidates: list[np.ndarray] = []
    if array.shape[1] == 3:
        candidates.append(array)
    if array.shape[0] == 3:
        candidates.append(array.T)

    if not candidates:
        raise ValueError(f"{name} must have shape Nx3 or 3xN, got {array.shape}")
    if expected_count is not None:
        matched = [candidate for candidate in candidates if candidate.shape[0] == expected_count]
        if matched:
            return matched[0]
    if len(candidates) == 1:
        return candidates[0]
    return candidates[0]


def normalize_normals(normals: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    result = np.zeros_like(normals, dtype=np.float64)
    np.divide(normals, lengths, out=result, where=lengths > eps)
    return result


def read_mask(path: Path) -> np.ndarray:
    cv2 = import_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read mask image: {path}")
    return image != 0


def load_ground_truth_normals(path: Path) -> np.ndarray:
    data = loadmat(str(path))
    normal = data.get("Normal_gt")
    if normal is None:
        arrays = [
            value
            for key, value in data.items()
            if not key.startswith("__")
            and isinstance(value, np.ndarray)
            and value.ndim >= 3
            and 3 in value.shape
        ]
        if len(arrays) != 1:
            keys = ", ".join(key for key in data.keys() if not key.startswith("__"))
            raise ValueError(f"Could not identify Normal_gt in {path}; keys: {keys}")
        normal = arrays[0]

    normal = np.squeeze(np.asarray(normal, dtype=np.float64))
    if normal.ndim != 3:
        raise ValueError(f"Normal_gt must be 3D, got shape {normal.shape}")
    if normal.shape[-1] == 3:
        pass
    elif normal.shape[0] == 3:
        normal = np.moveaxis(normal, 0, -1)
    else:
        raise ValueError(f"Normal_gt must have a vector dimension of size 3, got {normal.shape}")
    return normalize_normals(normal)


def load_diligent_sample(object_dir: Path, object_name: str | None = None) -> DiligentSample:
    object_dir = object_dir.resolve()
    if not object_dir.is_dir():
        raise FileNotFoundError(f"Object directory does not exist: {object_dir}")

    mask_path = find_file_case_insensitive(object_dir, ["mask.png"])
    light_path = find_file_case_insensitive(object_dir, ["light_directions.txt"])
    intensity_path = find_file_case_insensitive(object_dir, ["light_intensities.txt"])
    normal_path = find_file_case_insensitive(object_dir, ["Normal_gt.mat", "normal_gt.mat"])

    missing = [
        label
        for label, path in (
            ("mask.png", mask_path),
            ("light_directions.txt", light_path),
            ("Normal_gt.mat", normal_path),
        )
        if path is None
    ]
    if missing:
        raise FileNotFoundError(f"{object_dir} is missing required files: {', '.join(missing)}")

    image_paths = read_filenames(object_dir)
    if not image_paths:
        raise FileNotFoundError(f"No observation images found in {object_dir}")
    missing_images = [path for path in image_paths if not path.exists()]
    if missing_images:
        preview = ", ".join(str(path) for path in missing_images[:5])
        raise FileNotFoundError(f"filenames.txt references missing images: {preview}")

    light_directions = coerce_n_by_3_vectors(
        np.loadtxt(light_path), name="light_directions", expected_count=len(image_paths)
    )
    light_directions = normalize_normals(light_directions)
    if light_directions.shape[0] != len(image_paths):
        raise ValueError(
            f"Found {len(image_paths)} images but {light_directions.shape[0]} light directions"
        )

    light_intensities = None
    if intensity_path is not None:
        light_intensities = coerce_n_by_3_vectors(
            np.loadtxt(intensity_path),
            name="light_intensities",
            expected_count=len(image_paths),
        )
        if light_intensities.shape[0] != len(image_paths):
            raise ValueError(
                f"Found {len(image_paths)} images but {light_intensities.shape[0]} light intensities"
            )

    mask = read_mask(mask_path)
    normal_gt = load_ground_truth_normals(normal_path)
    if normal_gt.shape[:2] != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match Normal_gt shape {normal_gt.shape}")

    name = normalize_object_name(object_name or object_dir.name)
    return DiligentSample(
        object_name=name,
        object_dir=object_dir,
        image_paths=image_paths,
        light_directions=light_directions,
        light_intensities=light_intensities,
        mask=mask,
        normal_gt=normal_gt,
    )


def import_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise ImportError("opencv-python is required for DiLiGenT image I/O") from exc


def load_observation(path: Path, light_intensity: np.ndarray | None) -> np.ndarray:
    cv2 = import_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read observation image: {path}")

    array = image.astype(np.float64)
    eps = 1.0e-8
    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[:, :, :3][:, :, ::-1]
        if light_intensity is not None:
            rgb = rgb / np.maximum(light_intensity.reshape(1, 1, 3), eps)
        return rgb.mean(axis=2)

    if array.ndim != 2:
        raise ValueError(f"Unsupported image shape {array.shape} for {path}")
    if light_intensity is not None:
        array = array / max(float(np.mean(light_intensity)), eps)
    return array


def build_measurement_matrix(
    sample: DiligentSample, indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    for index in indices:
        intensity = None
        if sample.light_intensities is not None:
            intensity = sample.light_intensities[index]
        image = load_observation(sample.image_paths[index], intensity)
        if image.shape != sample.mask.shape:
            raise ValueError(
                f"Image shape {image.shape} does not match mask shape {sample.mask.shape}: "
                f"{sample.image_paths[index]}"
            )
        images.append(image.reshape(-1))

    measurements = np.stack(images, axis=1)
    light_directions = sample.light_directions[np.asarray(indices), :]
    return measurements, light_directions


def import_rps_class(rps_root: Path) -> Any:
    rps_root = rps_root.resolve()
    if not (rps_root / "rps.py").exists():
        raise FileNotFoundError(
            f"Could not find rps.py under {rps_root}. "
            "Clone https://github.com/yasumat/RobustPhotometricStereo there first."
        )

    root_str = str(rps_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    loaded_psutil = sys.modules.get("psutil")
    loaded_path = getattr(loaded_psutil, "__file__", "") if loaded_psutil is not None else ""
    if loaded_psutil is not None and not str(loaded_path).startswith(root_str):
        del sys.modules["psutil"]

    return importlib.import_module("rps").RPS


def solve_with_rps(
    RPS: Any,
    measurements: np.ndarray,
    light_directions: np.ndarray,
    mask: np.ndarray,
    solver_name: str,
) -> tuple[np.ndarray, float]:
    if solver_name not in SOLVER_ATTRS:
        raise ValueError(f"Unsupported solver {solver_name}")

    rps = RPS()
    rps.M = measurements
    rps.L = light_directions.T
    rps.height, rps.width = mask.shape
    mask_flat = mask.reshape(-1)
    rps.foreground_ind = np.where(mask_flat)[0]
    rps.background_ind = np.where(~mask_flat)[0]

    start = time.perf_counter()
    rps.solve(getattr(RPS, SOLVER_ATTRS[solver_name]))
    elapsed = time.perf_counter() - start

    normal = np.reshape(rps.N, (rps.height, rps.width, 3))
    normal[~mask] = 0.0
    return normalize_normals(normal), elapsed


def angular_error_map(estimated: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    estimated = normalize_normals(estimated)
    ground_truth = normalize_normals(ground_truth)
    dots = np.sum(estimated * ground_truth, axis=-1)
    dots = np.clip(dots, -1.0, 1.0)
    errors = np.degrees(np.arccos(dots))
    valid = mask & np.isfinite(errors)
    result = np.full(mask.shape, np.nan, dtype=np.float64)
    result[valid] = errors[valid]
    return result


def mean_angular_error(estimated: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> float:
    errors = angular_error_map(estimated, ground_truth, mask)
    return float(np.nanmean(errors[mask]))


def normals_to_height_frankot_chellappa(
    normals: np.ndarray, mask: np.ndarray, eps: float = 1.0e-8
) -> np.ndarray:
    normals = normalize_normals(normals)
    nz = np.where(np.abs(normals[:, :, 2]) < eps, np.nan, normals[:, :, 2])
    dz_dx = -normals[:, :, 0] / nz
    dz_dy = -normals[:, :, 1] / nz
    dz_dx = np.where(mask & np.isfinite(dz_dx), dz_dx, 0.0)
    dz_dy = np.where(mask & np.isfinite(dz_dy), dz_dy, 0.0)
    return frankot_chellappa(dz_dx, dz_dy, mask)


def frankot_chellappa(dz_dx: np.ndarray, dz_dy: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    height, width = dz_dx.shape
    wx = 2.0 * np.pi * fftfreq(width)
    wy = 2.0 * np.pi * fftfreq(height)
    grid_wx, grid_wy = np.meshgrid(wx, wy)
    denom = grid_wx * grid_wx + grid_wy * grid_wy
    denom[0, 0] = 1.0

    fft_x = fft2(dz_dx)
    fft_y = fft2(dz_dy)
    surface_fft = (-1j * grid_wx * fft_x - 1j * grid_wy * fft_y) / denom
    surface_fft[0, 0] = 0.0
    surface = np.real(ifft2(surface_fft))
    if mask is not None and np.any(mask):
        surface = surface - float(np.nanmedian(surface[mask]))
        surface = np.where(mask, surface, np.nan)
    return surface


def normal_to_rgb(normals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = ((np.clip(normals, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    cv2 = import_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb[:, :, ::-1])


def save_grayscale_png(path: Path, values: np.ndarray, mask: np.ndarray) -> None:
    cv2 = import_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros(mask.shape, dtype=np.uint8)
    valid = mask & np.isfinite(values)
    if np.any(valid):
        valid_values = values[valid]
        span = float(valid_values.max() - valid_values.min())
        if span > 1.0e-12:
            scaled = (valid_values - valid_values.min()) / span
        else:
            scaled = np.zeros_like(valid_values)
        image[valid] = np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), image)


def save_error_heatmap(path: Path, errors: np.ndarray, mask: np.ndarray, max_degrees: float) -> None:
    cv2 = import_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = np.zeros(mask.shape, dtype=np.uint8)
    valid = mask & np.isfinite(errors)
    if np.any(valid):
        scaled_values = np.clip(errors[valid] / max_degrees, 0.0, 1.0)
        scaled[valid] = (scaled_values * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    heatmap[~mask] = 0
    cv2.imwrite(str(path), heatmap)


def save_mesh_outputs(output_dir: Path, height: np.ndarray, mask: np.ndarray) -> dict[str, str]:
    try:
        import pyvista as pv
    except ImportError:
        return {"mesh_warning": "pyvista is not installed; mesh rendering skipped"}

    output_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = height.shape
    x, y = np.meshgrid(np.arange(cols, dtype=np.float64), np.arange(rows, dtype=np.float64))
    z = np.where(mask & np.isfinite(height), height, np.nan)

    grid = pv.StructuredGrid(x, y, z)
    mesh_path = output_dir / "height_mesh.vtk"
    screenshot_path = output_dir / "height_mesh.png"
    outputs = {"height_mesh": str(mesh_path)}
    grid.save(str(mesh_path))

    try:
        plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
        plotter.add_mesh(grid, cmap="viridis", show_edges=False)
        plotter.view_isometric()
        plotter.show(screenshot=str(screenshot_path))
        outputs["height_mesh_png"] = str(screenshot_path)
    except Exception as exc:  # pragma: no cover - depends on local rendering backend
        outputs["mesh_warning"] = f"pyvista screenshot failed: {exc}"
    return outputs


def parse_indices(indices: str, total: int) -> list[int]:
    parsed = [int(part.strip()) for part in indices.split(",") if part.strip()]
    if not parsed:
        raise ValueError("--indices did not contain any indices")
    for index in parsed:
        if index < 0 or index >= total:
            raise ValueError(f"Image index {index} is outside valid range [0, {total - 1}]")
    return parsed


def build_image_sets(args: argparse.Namespace, total: int) -> list[tuple[str, list[int]]]:
    if args.indices:
        indices = parse_indices(args.indices, total)
        return [("custom_indices", indices)]

    sets: list[tuple[str, list[int]]] = []
    for token in args.image_counts:
        lowered = token.lower()
        if lowered in {"all", "full"}:
            indices = list(range(total))
            sets.append(("lights_all", indices))
            continue

        count = int(token)
        if count <= 0:
            raise ValueError("Image counts must be positive")
        end = args.start_index + count
        if args.start_index < 0 or end > total:
            raise ValueError(
                f"Cannot take {count} images from start index {args.start_index}; "
                f"dataset has {total} images"
            )
        sets.append((f"lights_{count:03d}", list(range(args.start_index, end))))
    return sets


def load_baselines(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def attach_baseline(
    record: dict[str, Any], baselines: list[dict[str, str]], object_name: str, solver: str, image_count: int
) -> None:
    for row in baselines:
        if (
            normalize_object_name(row["object"]) == object_name
            and row["solver"].lower() == solver
            and int(row["image_count"]) == image_count
        ):
            baseline_mae = float(row["mae_deg"])
            record["baseline_mae_deg"] = baseline_mae
            record["baseline_delta_deg"] = record["mae_deg"] - baseline_mae
            record["baseline_ratio"] = record["mae_deg"] / baseline_mae if baseline_mae else None
            record["baseline_source_url"] = row.get("source_url", "")
            record["baseline_notes"] = row.get("notes", "")
            record["baseline_status"] = "reference_available"
            return
    record["baseline_status"] = "not_applicable_or_missing_reference"


def write_summary(output_dir: Path, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if not records:
        return

    fieldnames: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_validation(args: argparse.Namespace) -> list[dict[str, Any]]:
    sample = load_diligent_sample(args.object_dir, object_name=args.object_name)
    output_dir = (args.output_dir or Path("outputs") / "photometric_stereo" / sample.object_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    RPS = import_rps_class(args.rps_root)
    baselines = load_baselines(args.baseline_csv)
    records: list[dict[str, Any]] = []

    save_rgb_png(output_dir / "normal_gt_rgb.png", normal_to_rgb(sample.normal_gt, sample.mask))

    for image_set_name, indices in build_image_sets(args, len(sample.image_paths)):
        measurements, light_directions = build_measurement_matrix(sample, indices)

        for solver_name in args.solvers:
            case_dir = output_dir / image_set_name / solver_name
            case_dir.mkdir(parents=True, exist_ok=True)

            estimated, elapsed = solve_with_rps(
                RPS, measurements, light_directions, sample.mask, solver_name
            )
            errors = angular_error_map(estimated, sample.normal_gt, sample.mask)
            mae = mean_angular_error(estimated, sample.normal_gt, sample.mask)
            height = normals_to_height_frankot_chellappa(estimated, sample.mask)

            np.save(case_dir / "normal_est.npy", estimated)
            np.save(case_dir / "angular_error_deg.npy", errors)
            np.save(case_dir / "height_frankot_chellappa.npy", height)
            save_rgb_png(case_dir / "normal_est_rgb.png", normal_to_rgb(estimated, sample.mask))
            save_rgb_png(case_dir / "normal_gt_rgb.png", normal_to_rgb(sample.normal_gt, sample.mask))
            save_error_heatmap(case_dir / "angular_error_heatmap.png", errors, sample.mask, args.error_max_deg)
            save_grayscale_png(case_dir / "height_frankot_chellappa.png", height, sample.mask)

            outputs = {
                "normal_est_npy": str(case_dir / "normal_est.npy"),
                "normal_est_rgb": str(case_dir / "normal_est_rgb.png"),
                "normal_gt_rgb": str(case_dir / "normal_gt_rgb.png"),
                "angular_error_npy": str(case_dir / "angular_error_deg.npy"),
                "angular_error_heatmap": str(case_dir / "angular_error_heatmap.png"),
                "height_npy": str(case_dir / "height_frankot_chellappa.npy"),
                "height_png": str(case_dir / "height_frankot_chellappa.png"),
            }
            if not args.skip_mesh:
                outputs.update(save_mesh_outputs(case_dir, height, sample.mask))

            record: dict[str, Any] = {
                "object": sample.object_name,
                "solver": solver_name,
                "image_set": image_set_name,
                "image_count": len(indices),
                "image_indices_zero_based": ",".join(str(index) for index in indices),
                "mae_deg": mae,
                "elapsed_sec": elapsed,
                "output_dir": str(case_dir),
            }
            attach_baseline(record, baselines, sample.object_name, solver_name, len(indices))
            records.append(record)

            baseline = ""
            if record["baseline_status"] == "reference_available":
                baseline = (
                    f", baseline={record['baseline_mae_deg']:.3f}, "
                    f"delta={record['baseline_delta_deg']:+.3f}"
                )
            print(
                f"{sample.object_name} {image_set_name} {solver_name}: "
                f"MAE={mae:.3f} deg, elapsed={elapsed:.2f}s{baseline}"
            )

    write_summary(output_dir, records)
    return records


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate RobustPhotometricStereo solvers on one DiLiGenT single-view object."
        )
    )
    parser.add_argument("--object-dir", type=Path, required=True, help="Folder for one DiLiGenT object.")
    parser.add_argument(
        "--rps-root",
        type=Path,
        default=Path("third_party") / "RobustPhotometricStereo",
        help="Local clone of https://github.com/yasumat/RobustPhotometricStereo.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder.")
    parser.add_argument("--object-name", default=None, help="Override object name used for baseline lookup.")
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=sorted(SOLVER_ATTRS),
        default=["l2", "l1"],
        help="RPS solver names to run.",
    )
    parser.add_argument(
        "--image-counts",
        nargs="+",
        default=["4"],
        help="Counts to run from --start-index, or 'all' for the full object.",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated zero-based image indices. Overrides --image-counts.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based first image used when --image-counts contains numeric counts.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=default_baseline_path(),
        help="CSV of published baselines to attach to matching runs.",
    )
    parser.add_argument(
        "--error-max-deg",
        type=float,
        default=90.0,
        help="Maximum degree value mapped to the top heatmap color.",
    )
    parser.add_argument("--skip-mesh", action="store_true", help="Skip pyvista mesh VTK/PNG outputs.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    run_validation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
