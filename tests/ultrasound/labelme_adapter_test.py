from __future__ import annotations

from labelme._ultrasound import AnnotationPrediction
from labelme._ultrasound.labelme_adapter import prediction_to_shape


def test_prediction_becomes_ordinary_editable_native_shape() -> None:
    prediction = AnnotationPrediction(
        label="EMA",
        points=((1.0, 2.0), (5.0, 2.0), (3.0, 6.0)),
        confidence=0.91,
        source_target="eye_muscle",
        strategy="single_model",
        model_id="seg_eye_muscle",
        artifact_id="seg_eye_muscle__transverse__v1",
    )

    shape = prediction_to_shape(prediction)

    assert shape.label == "EMA"
    assert shape.shape_type == "polygon"
    assert shape.closed is True
    assert shape.flags == {}
    assert shape.description == ""
    assert shape.can_add_point()
