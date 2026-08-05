"""Convert a single-target probability map into a Labelme-ready polygon."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage import measure

from .contracts import AnnotationPrediction
from .contracts import GeometryTransform
from .contracts import ModelSpec
from .contracts import ProcessedSegmentation
from .preprocessing import restore_mask_to_original


def probability_to_annotation(
    probability: npt.NDArray[np.floating],
    *,
    spec: ModelSpec,
    transform: GeometryTransform,
) -> AnnotationPrediction | None:
    """Threshold, clean, contour, simplify, and restore one model output."""

    segmentation = probability_to_segmentation(
        probability, spec=spec, transform=transform
    )
    return None if segmentation is None else segmentation.annotation


def probability_to_segmentation(
    probability: npt.NDArray[np.floating],
    *,
    spec: ModelSpec,
    transform: GeometryTransform,
) -> ProcessedSegmentation | None:
    """Return both the cleaned original-space mask and its polygon."""

    probability_array = np.asarray(probability, dtype=np.float32)
    if probability_array.shape != transform.model_hw:
        raise ValueError(
            f"Probability shape {probability_array.shape} does not match "
            f"model shape {transform.model_hw}."
        )
    if not np.isfinite(probability_array).all():
        raise ValueError("Probability map contains non-finite values.")

    mask = probability_array >= spec.threshold
    mask = _largest_component(mask, min_area=spec.min_component_area)
    if mask is None:
        return None
    mask = ndimage.binary_fill_holes(mask)

    restored_mask = restore_mask_to_original(mask, transform=transform)
    polygon = mask_to_polygon(
        restored_mask, tolerance=spec.polygon_tolerance
    )
    if polygon is None:
        return None

    confidence = float(probability_array[mask].mean())
    return ProcessedSegmentation(
        annotation=AnnotationPrediction(
            label=spec.export_label,
            points=polygon,
            confidence=confidence,
            source_target=spec.target,
            strategy="single_model",
            model_id=spec.model_id,
            artifact_id=spec.artifact_id,
        ),
        mask=restored_mask,
    )


def _largest_component(
    mask: npt.NDArray[np.bool_],
    *,
    min_area: int,
) -> npt.NDArray[np.bool_] | None:
    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    component_id = int(np.argmax(areas))
    if int(areas[component_id]) < min_area:
        return None
    return labels == component_id


def mask_to_polygon(
    mask: npt.NDArray[np.bool_],
    *,
    tolerance: float,
) -> tuple[tuple[float, float], ...] | None:
    padded = np.pad(mask.astype(np.uint8), 1)
    contours = measure.find_contours(padded, level=0.5)
    if not contours:
        return None
    contour = max(contours, key=lambda item: item.shape[0]) - 1.0
    simplified = measure.approximate_polygon(contour, tolerance=tolerance)
    if len(simplified) > 1 and np.allclose(simplified[0], simplified[-1]):
        simplified = simplified[:-1]
    if len(simplified) < 3:
        return None

    height, width = mask.shape
    points = tuple(
        (
            float(np.clip(column, 0.0, width - 1.0)),
            float(np.clip(row, 0.0, height - 1.0)),
        )
        for row, column in simplified
    )
    if abs(_signed_area(points)) < 1.0:
        return None
    return points


def _signed_area(points: tuple[tuple[float, float], ...]) -> float:
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
