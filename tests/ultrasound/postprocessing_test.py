from __future__ import annotations

import numpy as np

from labelme._ultrasound import GeometryTransform
from labelme._ultrasound import ModelSpec
from labelme._ultrasound import RoiPolicy
from labelme._ultrasound.postprocessing import probability_to_annotation


def _spec() -> ModelSpec:
    return ModelSpec(
        model_id="seg_eye_muscle",
        artifact_id="seg_eye_muscle__transverse__v1",
        target="eye_muscle",
        export_label="EMA",
        weights_path="weights.pth",
        config_path="model_config.yaml",
        threshold=0.5,
        roi=RoiPolicy(
            raw_size_wh=(800, 600),
            rectangle_xywh=(92, 31, 536, 536),
        ),
        min_component_area=20,
        polygon_tolerance=1.0,
    )


def test_largest_component_becomes_ema_polygon_in_original_coordinates() -> None:
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[10:40, 12:50] = 0.9
    probability[2:4, 2:4] = 0.99
    transform = GeometryTransform(
        original_hw=(600, 800),
        roi_xywh=(92, 31, 536, 536),
        model_hw=(64, 64),
    )

    prediction = probability_to_annotation(
        probability, spec=_spec(), transform=transform
    )

    assert prediction is not None
    assert prediction.label == "EMA"
    assert prediction.source_target == "eye_muscle"
    assert prediction.model_id == "seg_eye_muscle"
    assert len(prediction.points) >= 4
    assert min(point[0] for point in prediction.points) >= 92
    assert min(point[1] for point in prediction.points) >= 31
    assert prediction.confidence is not None
    assert prediction.confidence > 0.89


def test_no_component_returns_none() -> None:
    transform = GeometryTransform(
        original_hw=(536, 536),
        roi_xywh=(0, 0, 536, 536),
        model_hw=(64, 64),
    )
    assert (
        probability_to_annotation(
            np.zeros((64, 64), dtype=np.float32),
            spec=_spec(),
            transform=transform,
        )
        is None
    )
