from __future__ import annotations

import numpy as np

from labelme._ultrasound import AnnotationPrediction
from labelme._ultrasound import ProcessedSegmentation
from labelme._ultrasound.strategies import derive_fat_segmentation


def _segmentation(
    *,
    target: str,
    label: str,
    mask: np.ndarray,
    confidence: float,
) -> ProcessedSegmentation:
    return ProcessedSegmentation(
        annotation=AnnotationPrediction(
            label=label,  # type: ignore[arg-type]
            points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            confidence=confidence,
            source_target=target,  # type: ignore[arg-type]
            strategy="single_model",
        ),
        mask=mask.astype(np.bool_),
    )


def test_fat_is_between_skin_lower_and_ema_upper_boundaries() -> None:
    skin_mask = np.zeros((100, 100), dtype=np.bool_)
    skin_mask[10:20, 10:90] = True
    eye_mask = np.zeros((100, 100), dtype=np.bool_)
    eye_mask[50:80, 20:80] = True

    fat = derive_fat_segmentation(
        skin=_segmentation(
            target="skin", label="skin", mask=skin_mask, confidence=0.95
        ),
        eye_muscle=_segmentation(
            target="eye_muscle", label="EMA", mask=eye_mask, confidence=0.9
        ),
        polygon_tolerance=0.5,
    )

    assert fat is not None
    assert fat.annotation.label == "fat"
    assert fat.annotation.source_target == "derived_fat"
    assert fat.annotation.confidence == 0.9
    assert not fat.mask[:, :20].any()
    assert not fat.mask[:, 80:].any()
    assert fat.mask[19:51, 20:80].all()
    assert not fat.mask[:19].any()
    assert not fat.mask[51:].any()


def test_fat_is_not_created_when_boundaries_are_reversed() -> None:
    skin_mask = np.zeros((50, 50), dtype=np.bool_)
    skin_mask[30:40, 5:45] = True
    eye_mask = np.zeros((50, 50), dtype=np.bool_)
    eye_mask[10:20, 5:45] = True

    fat = derive_fat_segmentation(
        skin=_segmentation(
            target="skin", label="skin", mask=skin_mask, confidence=0.9
        ),
        eye_muscle=_segmentation(
            target="eye_muscle", label="EMA", mask=eye_mask, confidence=0.9
        ),
    )

    assert fat is None
