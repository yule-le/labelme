"""Derive fat from the anatomical boundaries of skin and EMA masks."""

from __future__ import annotations

import numpy as np

from ..contracts import AnnotationPrediction
from ..contracts import ProcessedSegmentation
from ..postprocessing import mask_to_polygon


def derive_fat_segmentation(
    *,
    skin: ProcessedSegmentation,
    eye_muscle: ProcessedSegmentation,
    polygon_tolerance: float = 1.5,
    minimum_width: int = 3,
) -> ProcessedSegmentation | None:
    """Fill the band from skin's lower boundary to EMA's upper boundary."""

    if skin.annotation.source_target != "skin":
        raise ValueError("Expected a skin segmentation.")
    if eye_muscle.annotation.source_target != "eye_muscle":
        raise ValueError("Expected an eye-muscle segmentation.")
    if skin.mask.shape != eye_muscle.mask.shape:
        raise ValueError("Skin and EMA masks must share original-image coordinates.")

    skin_columns = skin.mask.any(axis=0)
    eye_columns = eye_muscle.mask.any(axis=0)
    shared_columns = np.flatnonzero(skin_columns & eye_columns)
    if len(shared_columns) < minimum_width:
        return None

    skin_lower = np.full(skin.mask.shape[1], np.nan, dtype=np.float64)
    eye_upper = np.full(eye_muscle.mask.shape[1], np.nan, dtype=np.float64)
    for column in shared_columns:
        skin_lower[column] = float(np.flatnonzero(skin.mask[:, column]).max())
        eye_upper[column] = float(
            np.flatnonzero(eye_muscle.mask[:, column]).min()
        )

    valid = np.flatnonzero(
        np.isfinite(skin_lower)
        & np.isfinite(eye_upper)
        & (eye_upper > skin_lower)
    )
    if len(valid) < minimum_width:
        return None

    left = int(valid.min())
    right = int(valid.max())
    columns = np.arange(left, right + 1)
    top = np.interp(columns, valid, skin_lower[valid])
    bottom = np.interp(columns, valid, eye_upper[valid])
    valid_gap = bottom > top
    if int(valid_gap.sum()) < minimum_width:
        return None

    fat_mask = np.zeros_like(skin.mask, dtype=np.bool_)
    height = fat_mask.shape[0]
    for column, upper, lower, is_valid in zip(
        columns, top, bottom, valid_gap, strict=True
    ):
        if not is_valid:
            continue
        start = int(np.clip(np.ceil(upper), 0, height - 1))
        stop = int(np.clip(np.floor(lower), 0, height - 1))
        if stop > start:
            fat_mask[start : stop + 1, column] = True

    points = mask_to_polygon(fat_mask, tolerance=polygon_tolerance)
    if points is None:
        return None
    confidences = (
        skin.annotation.confidence,
        eye_muscle.annotation.confidence,
    )
    confidence = (
        min(value for value in confidences if value is not None)
        if any(value is not None for value in confidences)
        else None
    )
    return ProcessedSegmentation(
        annotation=AnnotationPrediction(
            label="fat",
            points=points,
            confidence=confidence,
            source_target="derived_fat",
            strategy="boundary_derived",
        ),
        mask=fat_mask,
    )
