"""Eye-muscle width/depth geometry derived from an EMA segmentation mask."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import measure

from ..contracts import Calibration
from .protocols import EyeMuscleProtocol


@dataclass(frozen=True)
class EyeMuscleGeometry:
    width_segment: tuple[tuple[float, float], tuple[float, float]]
    depth_segment: tuple[tuple[float, float], tuple[float, float]]
    width_mm: float
    depth_mm: float
    diagnostics: dict[str, object]


def _primary_contour(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("Eye-muscle measurement expects a 2D mask.")
    labels, count = ndimage.label(binary)
    if count == 0:
        raise ValueError("EMA mask is empty.")
    component_sizes = ndimage.sum(binary, labels, range(1, count + 1))
    primary = labels == int(np.argmax(component_sizes)) + 1
    contours = measure.find_contours(primary.astype(np.uint8), level=0.5)
    if not contours:
        raise ValueError("No contour found in EMA mask.")
    # skimage returns (row, column); measurement geometry uses (x, y).
    contours.sort(key=lambda contour: int(contour.shape[0]))
    points = np.asarray(contours[-1])[:, ::-1].astype(np.float64)
    if len(points) < 3:
        raise ValueError("Insufficient EMA contour points.")
    return points


def _resample_closed_contour(points: np.ndarray, count: int) -> np.ndarray:
    closed = np.vstack([points, points[0]])
    deltas = np.diff(closed, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    total = float(lengths.sum())
    if total <= 0.0:
        raise ValueError("Degenerate EMA contour.")
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(0.0, total, count, endpoint=False)
    output = np.zeros((count, 2), dtype=np.float64)
    for index, distance in enumerate(samples):
        segment_index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment_index = min(max(segment_index, 0), len(lengths) - 1)
        segment_length = float(lengths[segment_index])
        if segment_length <= 0.0:
            output[index] = closed[segment_index]
            continue
        fraction = (distance - cumulative[segment_index]) / segment_length
        output[index] = closed[segment_index] + fraction * deltas[segment_index]
    return output


def _top_reference(
    sampled: np.ndarray, protocol: EyeMuscleProtocol
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    x_min, y_min = sampled.min(axis=0)
    x_max, y_max = sampled.max(axis=0)
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    y_threshold = float(y_min + protocol.top_region_ratio * height)
    x_margin = float(protocol.top_margin_ratio * width)
    fit = sampled[
        (sampled[:, 1] <= y_threshold)
        & (sampled[:, 0] >= x_min + x_margin)
        & (sampled[:, 0] <= x_max - x_margin)
    ]
    if len(fit) < 8:
        fit = sampled[sampled[:, 1] <= np.percentile(sampled[:, 1], 25.0)]
    if len(fit) < 2:
        raise ValueError("Insufficient points to fit EMA top reference.")
    slope, intercept = np.polyfit(fit[:, 0], fit[:, 1], deg=1)
    start_x = float(fit[:, 0].min())
    end_x = float(fit[:, 0].max())
    return (
        np.array([start_x, slope * start_x + intercept]),
        np.array([end_x, slope * end_x + intercept]),
        {
            "topReferenceSlope": float(slope),
            "topReferencePointCount": int(len(fit)),
        },
    )


def _width_points(
    sampled: np.ndarray,
    top_start: np.ndarray,
    top_end: np.ndarray,
    protocol: EyeMuscleProtocol,
) -> tuple[np.ndarray, np.ndarray]:
    y_min = float(sampled[:, 1].min())
    y_max = float(sampled[:, 1].max())
    upper = sampled[
        sampled[:, 1] <= y_min + protocol.top_region_ratio * (y_max - y_min)
    ]
    if len(upper) < 8:
        upper = sampled[sampled[:, 1] <= np.percentile(sampled[:, 1], 25.0)]
    direction = top_end - top_start
    norm = float(np.linalg.norm(direction))
    if len(upper) < 2 or norm <= 1e-12:
        raise ValueError("Unable to select EMA width points.")
    unit = direction / norm
    projections = (upper - top_start) @ unit
    return upper[int(np.argmin(projections))], upper[int(np.argmax(projections))]


def _bottom_anchor(
    sampled: np.ndarray,
    top_start: np.ndarray,
    top_end: np.ndarray,
    protocol: EyeMuscleProtocol,
) -> np.ndarray:
    x_min, y_min = sampled.min(axis=0)
    x_max, y_max = sampled.max(axis=0)
    x_mid = float((top_start[0] + top_end[0]) / 2.0)
    y_mid = float((y_min + y_max) / 2.0)
    band_half = float(protocol.center_band_ratio * (x_max - x_min))
    candidates = sampled[
        (np.abs(sampled[:, 0] - x_mid) <= band_half) & (sampled[:, 1] >= y_mid)
    ]
    if len(candidates) < 3:
        nearest = sampled[
            np.argsort(np.abs(sampled[:, 0] - x_mid))[
                : max(6, int(0.08 * len(sampled)))
            ]
        ]
        lower = nearest[nearest[:, 1] >= y_mid]
        candidates = lower if len(lower) else nearest
    return candidates[int(np.argmax(candidates[:, 1]))]


def _depth_top(
    mask: np.ndarray,
    bottom: np.ndarray,
    width_left: np.ndarray,
    width_right: np.ndarray,
) -> np.ndarray:
    width_direction = width_right - width_left
    norm = float(np.linalg.norm(width_direction))
    if norm <= 1e-12:
        raise ValueError("Degenerate EMA width segment.")
    width_unit = width_direction / norm
    normal = np.array([-width_unit[1], width_unit[0]], dtype=np.float64)
    if normal[1] >= 0.0:
        normal = -normal
    height, width = mask.shape
    distances = np.arange(0.0, float(np.hypot(height, width)) * 1.5, 0.25)
    points = bottom + distances[:, None] * normal
    xs = np.round(points[:, 0]).astype(int)
    ys = np.round(points[:, 1]).astype(int)
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    seen_inside = False
    last_inside: int | None = None
    for index in range(len(points)):
        if not in_bounds[index]:
            if seen_inside:
                break
            continue
        if mask[ys[index], xs[index]]:
            seen_inside = True
            last_inside = index
        elif seen_inside:
            break
    if last_inside is None:
        raise ValueError("Unable to find EMA depth top intersection.")
    return points[last_inside]


def _physical_length(
    first: np.ndarray, second: np.ndarray, calibration: Calibration
) -> float:
    delta = second - first
    return float(
        np.hypot(
            delta[0] * calibration.pixel_size_x_mm,
            delta[1] * calibration.pixel_size_y_mm,
        )
    )


def measure_eye_muscle(
    mask: np.ndarray,
    calibration: Calibration,
    protocol: EyeMuscleProtocol,
) -> EyeMuscleGeometry:
    contour = _primary_contour(mask)
    sampled = _resample_closed_contour(contour, protocol.contour_num_points)
    top_start, top_end, diagnostics = _top_reference(sampled, protocol)
    width_left, width_right = _width_points(sampled, top_start, top_end, protocol)
    bottom = _bottom_anchor(sampled, top_start, top_end, protocol)
    depth_top = _depth_top(
        np.asarray(mask, dtype=bool), bottom, width_left, width_right
    )
    return EyeMuscleGeometry(
        width_segment=(tuple(width_left), tuple(width_right)),
        depth_segment=(tuple(bottom), tuple(depth_top)),
        width_mm=_physical_length(width_left, width_right, calibration),
        depth_mm=_physical_length(bottom, depth_top, calibration),
        diagnostics=diagnostics,
    )
