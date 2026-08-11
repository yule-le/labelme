"""Fat and skin depth measured vertically from the EMA depth reference axis."""

from __future__ import annotations

import numpy as np

from ..contracts import Calibration
from ..contracts import PolygonPoint


def measure_layer_depth(
    mask: np.ndarray,
    eye_depth_segment: tuple[PolygonPoint, PolygonPoint],
    calibration: Calibration,
    *,
    axis_offset_mm: float = 0.0,
) -> tuple[float, tuple[PolygonPoint, PolygonPoint], dict[str, object]]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("Layer depth measurement expects a 2D mask.")
    bottom = np.asarray(eye_depth_segment[0], dtype=np.float64)
    top = np.asarray(eye_depth_segment[1], dtype=np.float64)
    if bottom[1] < top[1]:
        bottom, top = top, bottom
    direction = top - bottom
    direction_mm = np.array(
        [
            direction[0] * calibration.pixel_size_x_mm,
            direction[1] * calibration.pixel_size_y_mm,
        ]
    )
    physical_norm = float(np.linalg.norm(direction_mm))
    if physical_norm <= 1e-12:
        raise ValueError("Degenerate EMA depth axis.")
    if abs(axis_offset_mm) > 1e-12:
        # Match the inference contract: offset is perpendicular to the final
        # image-vertical measurement line, so a positive offset moves right.
        offset_mm = np.array([axis_offset_mm, 0.0], dtype=np.float64)
        offset_px = np.array(
            [
                offset_mm[0] / calibration.pixel_size_x_mm,
                offset_mm[1] / calibration.pixel_size_y_mm,
            ]
        )
        bottom += offset_px
        top += offset_px

    # The EMD top endpoint selects the anatomical measurement location. The
    # layer itself is sampled vertically in image space, not along a possibly
    # slanted continuation of the EMD axis.
    direction = np.array([0.0, -1.0], dtype=np.float64)
    unit = direction
    height, width = binary.shape
    distances = np.arange(0.0, float(np.hypot(height, width)) * 1.5, 0.25)
    points = top + distances[:, None] * unit
    xs = np.round(points[:, 0]).astype(int)
    ys = np.round(points[:, 1]).astype(int)
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

    entry_index: int | None = None
    exit_index: int | None = None
    in_run = False
    for index in range(len(points)):
        inside = bool(in_bounds[index] and binary[ys[index], xs[index]])
        if inside and not in_run:
            entry_index = index
            in_run = True
        elif not inside and in_run:
            exit_index = index - 1
            break
    if in_run and exit_index is None:
        exit_index = len(points) - 1
    if entry_index is None or exit_index is None or exit_index < entry_index:
        raise ValueError("EMA depth axis does not intersect the layer.")

    entry = points[entry_index]
    exit_ = points[exit_index]
    delta = exit_ - entry
    value_mm = float(
        np.hypot(
            delta[0] * calibration.pixel_size_x_mm,
            delta[1] * calibration.pixel_size_y_mm,
        )
    )
    return (
        value_mm,
        (tuple(entry), tuple(exit_)),
        {
            "axisOffsetMm": float(axis_offset_mm),
            "axisMode": "image_vertical_from_eye_top",
        },
    )
