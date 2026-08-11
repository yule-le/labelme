"""Convert framework-independent predictions into native Labelme shapes."""

from __future__ import annotations

from typing import cast

import numpy as np

from labelme._shape import Shape

from .contracts import AnnotationPrediction
from .contracts import Calibration
from .contracts import MeasurementResult


def prediction_to_shape(prediction: AnnotationPrediction) -> Shape:
    """Create an ordinary editable Labelme polygon."""

    if prediction.shape_type != "polygon" or len(prediction.points) < 3:
        raise ValueError(
            "Ultrasound annotations must be polygons with at least 3 points."
        )
    return Shape(
        label=prediction.label,
        shape_type="polygon",
        points=np.asarray(prediction.points, dtype=np.float64),
        flags={},
        description="",
        closed=True,
    )


def measurement_to_shape(measurement: MeasurementResult) -> Shape | None:
    """Create an editable native line for one valid measurement result."""

    if not measurement.valid or measurement.segment is None:
        return None
    return Shape(
        label=measurement.code,
        shape_type="line",
        points=np.asarray(measurement.segment, dtype=np.float64),
        flags={},
        description=(
            f"{measurement.code}: {measurement.value_mm:.2f} mm"
            if measurement.value_mm is not None
            else measurement.code
        ),
        other_data={
            "ultrasoundMeasurement": {
                "code": measurement.code,
                "valueMm": measurement.value_mm,
                "valid": measurement.valid,
                "methodVersion": measurement.method_version,
            }
        },
        closed=False,
    )


def update_ultrasound_metadata(
    other_data: dict[str, object],
    *,
    calibration: Calibration,
    original_image_path: str | None,
    measurements: tuple[MeasurementResult, ...],
) -> None:
    """Merge calibration and selected results without deleting other metadata."""

    metadata = update_depth_metadata(
        other_data,
        depth_setting_mm=calibration.depth_setting_mm,
        original_image_path=original_image_path,
    )
    metadata.update(
        {
            "calibration": {
                "measurementImageSpace": "cropped_image",
                "roiHeightPx": calibration.image_height_px,
                "pixelSizeXMm": calibration.pixel_size_x_mm,
                "pixelSizeYMm": calibration.pixel_size_y_mm,
            },
        }
    )
    stored_value = metadata.get("measurements")
    stored = (
        cast(dict[str, object], stored_value)
        if isinstance(stored_value, dict)
        else {}
    )
    if not isinstance(stored_value, dict):
        metadata["measurements"] = stored
    for result in measurements:
        payload: dict[str, object] = {
            "valueMm": result.value_mm,
            "valid": result.valid,
            "methodVersion": result.method_version,
        }
        if result.error is not None:
            payload["error"] = result.error
        if result.diagnostics:
            payload["diagnostics"] = result.diagnostics
        stored[result.code] = payload


def update_depth_metadata(
    other_data: dict[str, object],
    *,
    depth_setting_mm: float,
    original_image_path: str | None,
) -> dict[str, object]:
    """Persist the imported CSV row even before measurements are generated."""

    metadata_value = other_data.get("ultrasoundMetadata")
    metadata = (
        cast(dict[str, object], metadata_value)
        if isinstance(metadata_value, dict)
        else {}
    )
    if not isinstance(metadata_value, dict):
        other_data["ultrasoundMetadata"] = metadata
    metadata.update(
        {
            "depthSettingMm": float(depth_setting_mm),
            "depthSource": "csv",
        }
    )
    if original_image_path is not None:
        if "/" in original_image_path or "\\" in original_image_path:
            metadata["originalImagePath"] = original_image_path
            metadata.pop("sourceFilename", None)
        else:
            metadata["sourceFilename"] = original_image_path
            metadata.pop("originalImagePath", None)
    return metadata
