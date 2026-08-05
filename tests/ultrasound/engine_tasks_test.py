from __future__ import annotations

import numpy as np
import pytest

from labelme._ultrasound import AnnotationPrediction
from labelme._ultrasound import InferenceRequest
from labelme._ultrasound import ProcessedSegmentation
from labelme._ultrasound.engine import UltrasoundInferenceEngine


def _segmentation(target: str) -> ProcessedSegmentation:
    mask = np.zeros((40, 40), dtype=np.bool_)
    if target == "skin":
        mask[2:5, 5:35] = True
        label = "skin"
    else:
        mask[20:30, 10:30] = True
        label = "EMA"
    return ProcessedSegmentation(
        annotation=AnnotationPrediction(
            label=label,  # type: ignore[arg-type]
            points=((5.0, 2.0), (35.0, 2.0), (35.0, 5.0)),
            confidence=0.9,
            source_target=target,  # type: ignore[arg-type]
            strategy="single_model",
        ),
        mask=mask,
    )


@pytest.fixture
def fake_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[UltrasoundInferenceEngine, list[str]]:
    called: list[str] = []

    def _fake_infer_target(
        self: UltrasoundInferenceEngine,
        request: InferenceRequest,
        *,
        target: str,
    ) -> ProcessedSegmentation:
        del self, request
        called.append(target)
        return _segmentation(target)

    monkeypatch.setattr(
        UltrasoundInferenceEngine,
        "_infer_target",
        _fake_infer_target,
    )
    return UltrasoundInferenceEngine(), called


def test_ema_and_skin_tasks_run_only_their_models(
    fake_engine: tuple[UltrasoundInferenceEngine, list[str]],
) -> None:
    engine, called = fake_engine
    result = engine.infer(
        InferenceRequest(
            request_id="selected-models",
            image_path="image.png",
            image=np.zeros((40, 40), dtype=np.uint8),
            tasks=("EMA", "skin"),
        )
    )

    assert called == ["eye_muscle", "skin"]
    assert [prediction.label for prediction in result.predictions] == [
        "EMA",
        "skin",
    ]


def test_fat_only_runs_boundary_models_but_exports_only_fat(
    fake_engine: tuple[UltrasoundInferenceEngine, list[str]],
) -> None:
    engine, called = fake_engine
    result = engine.infer(
        InferenceRequest(
            request_id="fat-only",
            image_path="image.png",
            image=np.zeros((40, 40), dtype=np.uint8),
            tasks=("fat",),
        )
    )

    assert called == ["eye_muscle", "skin"]
    assert [prediction.label for prediction in result.predictions] == ["fat"]
