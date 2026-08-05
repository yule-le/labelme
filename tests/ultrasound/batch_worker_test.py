from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import cast

from PIL import Image

from labelme._label_file import read_label_file
from labelme._ultrasound import AnnotationPrediction
from labelme._ultrasound import BatchInferenceItem
from labelme._ultrasound import BatchInferenceRequest
from labelme._ultrasound import BatchInferenceResult
from labelme._ultrasound import InferenceRequest
from labelme._ultrasound import InferenceResult
from labelme._ultrasound.worker import UltrasoundInferenceWorker


class _FakeEngine:
    def infer(self, request: InferenceRequest) -> InferenceResult:
        predictions = tuple(
            AnnotationPrediction(
                label=task,
                points=((1.0, 1.0), (8.0, 1.0), (8.0, 8.0)),
                confidence=0.9,
                source_target=(
                    "eye_muscle"
                    if task == "EMA"
                    else "skin"
                    if task == "skin"
                    else "derived_fat"
                ),
                strategy=(
                    "boundary_derived" if task == "fat" else "single_model"
                ),
            )
            for task in request.tasks
        )
        return InferenceResult(
            request_id=request.request_id,
            image_path=request.image_path,
            predictions=predictions,
        )


def test_batch_creates_standard_json_and_skips_existing(tmp_path: Path) -> None:
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    Image.new("L", (20, 15), color=100).save(first_image)
    Image.new("L", (20, 15), color=100).save(second_image)
    first_json = tmp_path / "first.json"
    second_json = tmp_path / "second.json"
    second_json.write_text("already exists", encoding="utf-8")

    worker = UltrasoundInferenceWorker()
    worker._engine = cast(Any, _FakeEngine())
    completed: list[BatchInferenceResult] = []
    worker.batch_completed.connect(completed.append)
    worker.infer_batch(
        BatchInferenceRequest(
            request_id="batch",
            items=(
                BatchInferenceItem(str(first_image), str(first_json)),
                BatchInferenceItem(str(second_image), str(second_json)),
            ),
            tasks=("EMA", "skin"),
        )
    )

    assert len(completed) == 1
    result = completed[0]
    assert result.succeeded == (str(first_image),)
    assert result.skipped == (str(second_image),)
    assert result.failures == ()
    annotation = read_label_file(filename=str(first_json))
    assert [shape["label"] for shape in annotation.shapes] == ["EMA", "skin"]
    assert all(shape["flags"] == {} for shape in annotation.shapes)
    assert all(shape["description"] == "" for shape in annotation.shapes)
    assert second_json.read_text(encoding="utf-8") == "already exists"
