from __future__ import annotations

from labelme._ultrasound import AnnotationPrediction


def test_annotation_prediction_is_always_a_polygon() -> None:
    prediction = AnnotationPrediction(
        label="fat",
        points=((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
        confidence=0.9,
        strategy="boundary_derived",
    )

    assert prediction.shape_type == "polygon"
    assert prediction.label == "fat"
    assert prediction.strategy == "boundary_derived"
