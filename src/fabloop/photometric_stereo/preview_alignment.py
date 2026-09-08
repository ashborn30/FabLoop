"""Estimate image alignment for qualitative PCBA previews, not calibrated PS.

Planar feature matches cannot validate lighting, camera pose or alignment of raised
components. The thresholds below reject gross fitting failures; they are transparent
diagnostic heuristics, not scientifically validated acceptance criteria.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

LIGHT_ORDER = ("F", "B", "L", "R")
HEURISTICS = {
    "sift_nfeatures": 8000,
    "sift_contrast_threshold": 0.02,
    "descriptor_ratio_max": 0.75,
    "minimum_matches": 12,
    "minimum_inliers": 12,
    "minimum_inlier_ratio": 0.20,
    "ransac_reprojection_threshold_pixels": 3.0,
    "minimum_inlier_hull_fraction": 0.02,
    "minimum_reference_overlap_fraction": 0.35,
    "minimum_common_valid_fraction": 0.25,
    "projected_area_ratio_range": [0.20, 4.0],
    "maximum_normalized_homography_condition": 10000.0,
    "orientation_warning_degrees": 15.0,
    "intersection_erosion_pixels": 1,
}


def _load_image(path: Path, max_edge: int) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing directional capture: {path}")
    data = path.read_bytes()
    with Image.open(BytesIO(data)) as image:
        original_mode = image.mode
        exif_orientation = image.getexif().get(274, 1)
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    if (resized_height, resized_width) != (height, width):
        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    sx, sy = resized_width / width, resized_height / height
    # OpenCV resize maps pixel centres: dst = (src + 0.5) * scale - 0.5.
    source_to_resized = [[sx, 0.0, 0.5 * (sx - 1.0)],
                         [0.0, sy, 0.5 * (sy - 1.0)], [0.0, 0.0, 1.0]]
    return rgb, {
        "path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
        "original_shape_hw": [height, width], "resized_shape_hw": [resized_height, resized_width],
        "original_mode": original_mode, "exif_orientation_applied": False,
        "exif_orientation_tag": exif_orientation,
        "source_to_resized": source_to_resized,
        "resize": "aspect_preserving_downsample_only_with_integer_dimension_rounding",
    }


def _normalizer(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    return np.array([[2.0 / width, 0.0, -1.0], [0.0, 2.0 / height, -1.0], [0.0, 0.0, 1.0]])


def _validate_homography(
    homography: np.ndarray, source_hw: tuple[int, int], reference_hw: tuple[int, int],
) -> dict[str, float]:
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("Non-finite or malformed alignment homography")
    normalized = _normalizer(reference_hw) @ homography @ np.linalg.inv(_normalizer(source_hw))
    condition = float(np.linalg.cond(normalized))
    if not np.isfinite(condition) or condition > HEURISTICS["maximum_normalized_homography_condition"]:
        raise ValueError("Alignment homography is singular or excessively ill-conditioned")
    height, width = source_hw
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    homogeneous = np.c_[corners, np.ones(4)] @ homography.T
    denominators = homogeneous[:, 2]
    if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
        raise ValueError("Alignment homography has a projective pole across the source image")
    projected = (homogeneous[:, :2] / denominators[:, None]).astype(np.float32)
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        raise ValueError("Alignment produces an invalid source-image quadrilateral")
    area_ratio = float(cv2.contourArea(projected, oriented=True) / np.prod(reference_hw))
    minimum, maximum = HEURISTICS["projected_area_ratio_range"]
    if not minimum <= area_ratio <= maximum:
        raise ValueError(f"Implausible or mirrored projected image area ratio: {area_ratio:.4f}")
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0, 1.0])
    numerator = homography @ center
    jacobian = (homography[:2, :2] * numerator[2]
                - numerator[:2, None] * homography[2, :2]) / numerator[2] ** 2
    u, _, vh = np.linalg.svd(jacobian)
    rotation = u @ vh
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return {"normalized_homography_condition": condition, "projected_area_ratio": area_ratio,
            "estimated_image_rotation_degrees_at_source_center": angle}


def _estimate_alignment(
    source: np.ndarray, source_keypoints: tuple, source_descriptors: np.ndarray,
    reference: np.ndarray, reference_keypoints: tuple, reference_descriptors: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_descriptors, reference_descriptors, k=2)
    candidates = sorted((pair[0] for pair in pairs if len(pair) == 2
                         and pair[0].distance < HEURISTICS["descriptor_ratio_max"] * pair[1].distance),
                        key=lambda match: match.distance)
    # Repeated patterns must not inflate support by reusing one reference feature.
    used_reference = set()
    matches = []
    for match in candidates:
        if match.trainIdx not in used_reference:
            matches.append(match)
            used_reference.add(match.trainIdx)
    if len(matches) < HEURISTICS["minimum_matches"]:
        raise ValueError(f"Insufficient distinctive correspondences: {len(matches)}")
    source_points = np.float32([source_keypoints[match.queryIdx].pt for match in matches])
    reference_points = np.float32([reference_keypoints[match.trainIdx].pt for match in matches])
    homography, inlier_mask = cv2.findHomography(
        source_points, reference_points, cv2.RANSAC,
        HEURISTICS["ransac_reprojection_threshold_pixels"], maxIters=5000, confidence=0.999,
    )
    if homography is None or inlier_mask is None:
        raise ValueError("RANSAC could not estimate an alignment homography")
    selected = inlier_mask.ravel().astype(bool)
    inliers = int(selected.sum())
    inlier_ratio = inliers / len(matches)
    if inliers < HEURISTICS["minimum_inliers"] or inlier_ratio < HEURISTICS["minimum_inlier_ratio"]:
        raise ValueError(f"Insufficient RANSAC support: {inliers}/{len(matches)} inliers")
    diagnostics = _validate_homography(homography, source.shape[:2], reference.shape[:2])
    source_hull = float(cv2.contourArea(cv2.convexHull(source_points[selected])) / np.prod(source.shape[:2]))
    reference_hull = float(cv2.contourArea(cv2.convexHull(reference_points[selected])) / np.prod(reference.shape[:2]))
    if min(source_hull, reference_hull) < HEURISTICS["minimum_inlier_hull_fraction"]:
        raise ValueError("Alignment inliers cover too little of the images")
    predicted = cv2.perspectiveTransform(source_points[selected, None, :], homography)[:, 0]
    errors = np.linalg.norm(predicted - reference_points[selected], axis=1)
    return homography, {
        **diagnostics, "method": "SIFT_ratio_unique_matches_RANSAC_homography",
        "matches": len(matches), "inliers": inliers, "inlier_ratio": inlier_ratio,
        "median_inlier_reprojection_error_pixels": float(np.median(errors)),
        "p95_inlier_reprojection_error_pixels": float(np.percentile(errors, 95)),
        "source_inlier_hull_fraction": source_hull, "reference_inlier_hull_fraction": reference_hull,
    }


def align_lights(
    board_dir: Path, max_edge: int = 1024,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Return RGB captures on the resized F grid and conservative common support.

    This estimates a planar alignment only. It never resizes unequal captures to
    the reference aspect ratio, invents a failed fit, or handles ambient images.
    Missing captures or unreliable feature fits raise an explicit exception.
    """
    if isinstance(max_edge, bool) or not isinstance(max_edge, int) or max_edge < 64:
        raise ValueError("max_edge must be an integer of at least 64 pixels")
    board_dir = Path(board_dir)
    images, sources = {}, {}
    for direction in LIGHT_ORDER:
        images[direction], sources[direction] = _load_image(board_dir / f"light_{direction}.jpg", max_edge)
    sift = cv2.SIFT_create(nfeatures=HEURISTICS["sift_nfeatures"],
                           contrastThreshold=HEURISTICS["sift_contrast_threshold"])
    features = {}
    for direction, rgb in images.items():
        keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
        if descriptors is None or len(keypoints) < HEURISTICS["minimum_matches"]:
            raise ValueError(f"Light {direction}: insufficient image texture for estimated alignment")
        features[direction] = (keypoints, descriptors)
    reference = images["F"]
    height, width = reference.shape[:2]
    aligned = {"F": reference.copy()}
    common_mask = np.ones((height, width), dtype=np.uint8)
    alignment = {
        "F": {"method": "reference_identity", "source_to_reference_homography": np.eye(3).tolist(),
              "original_source_to_reference_homography": sources["F"]["source_to_resized"],
              "reference_overlap_fraction": 1.0, "keypoints": len(features["F"][0])},
    }
    for direction in LIGHT_ORDER[1:]:
        source = images[direction]
        try:
            homography, diagnostics = _estimate_alignment(source, *features[direction], reference, *features["F"])
        except ValueError as error:
            raise ValueError(f"Light {direction}: {error}") from error
        support = cv2.warpPerspective(np.ones(source.shape[:2], dtype=np.uint8), homography,
                                      (width, height), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Also reject partial zero-padding with the image's exact interpolation
        # rule. Nearest support plus one output-pixel erosion is insufficient when
        # a homography magnifies the source by more than two at a boundary.
        interpolation_support = cv2.warpPerspective(np.ones(source.shape[:2], dtype=np.float32), homography,
                                                     (width, height), flags=cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        support &= (interpolation_support >= 1.0 - 1e-6).astype(np.uint8)
        overlap = float(np.mean(support))
        if overlap < HEURISTICS["minimum_reference_overlap_fraction"]:
            raise ValueError(f"Light {direction}: insufficient reference overlap ({overlap:.3f})")
        aligned[direction] = cv2.warpPerspective(source, homography, (width, height), flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        common_mask &= support
        alignment[direction] = {
            **diagnostics, "source_to_reference_homography": homography.tolist(),
            "original_source_to_reference_homography": (homography @ np.asarray(sources[direction]["source_to_resized"])).tolist(),
            "reference_overlap_fraction": overlap, "keypoints": len(features[direction][0]),
        }
    # Keep a conservative extra pixel away from the verified sampling boundary.
    common_mask = cv2.erode(common_mask, np.ones((3, 3), dtype=np.uint8), iterations=1,
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    common_fraction = float(np.mean(common_mask))
    if common_fraction < HEURISTICS["minimum_common_valid_fraction"]:
        raise ValueError(f"Insufficient common support across four captures: {common_fraction:.3f}")
    metadata = {
        "stage": "qualitative_preview_alignment", "reference_light": "F", "light_order": list(LIGHT_ORDER),
        "max_edge": max_edge, "output_shape_hw": [height, width], "opencv_version": cv2.__version__,
        "alignment_status": "estimated_unverified", "registration_verified": False,
        "calibrated": False, "ready_for_training": False,
        "heuristics": dict(HEURISTICS), "sources": sources, "alignment": alignment,
        "common_valid_fraction": common_fraction,
        "mask_scope": "intersection_of_warped_image_support_eroded_one_pixel; not_a_board_mask",
        "orientation_warnings": [
            f"Light {direction}: large estimated image rotation; camera/board/edit orientation and light-frame mapping remain unknown."
            for direction in LIGHT_ORDER[1:]
            if abs(alignment[direction]["estimated_image_rotation_degrees_at_source_center"]) > HEURISTICS["orientation_warning_degrees"]
        ],
        "limitations": [
            "Diagnostic thresholds reject gross fitting failures and are not validated acceptance criteria.",
            "Planar homography matches do not verify alignment of raised components, parallax, or camera pose.",
            "No measured light directions, intensities, response linearity, or ambient correction are supplied here.",
            "Source images remain unchanged; this result is for qualitative previews only.",
        ],
    }
    return aligned, common_mask, metadata
