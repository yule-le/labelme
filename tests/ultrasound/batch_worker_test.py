from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import cast

from PIL import Image

from labelme._label_file import Annotation
from labelme._label_file import read_image_file
from labelme._label_file import read_label_file
from labelme._label_file import write_label_file
from labelme._ultrasound import AnnotationPrediction
from labelme._ultrasound import BatchInferenceItem
from labelme._ultrasound import BatchInferenceRequest
from labelme._ultrasound import BatchInferenceResult
from labelme._ultrasound import Calibration
from labelme._ultrasound import InferenceRequest
from labelme._ultrasound import InferenceResult
from labelme._ultrasound import MeasurementResult
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
                strategy=("boundary_derived" if task == "fat" else "single_model"),
            )
            for task in request.tasks
        )
        return InferenceResult(
            request_id=request.request_id,
            image_path=request.image_path,
            predictions=predictions,
        )


class _FakeMeasurementEngine:
    def infer(self, request: InferenceRequest) -> InferenceResult:
        assert request.depth_setting_mm == 100.0
        return InferenceResult(
            request_id=request.request_id,
            image_path=request.image_path,
            predictions=(),
            measurements=(
                MeasurementResult(
                    code="EMW",
                    value_mm=20.0,
                    segment=((2.0, 5.0), (12.0, 5.0)),
                    valid=True,
                    method_version="geom-depth-width-v1.1",
                ),
            ),
            calibration=Calibration(
                depth_setting_mm=100.0,
                image_height_px=15,
                pixel_size_x_mm=100.0 / 15.0,
                pixel_size_y_mm=100.0 / 15.0,
            ),
            depth_setting_mm=100.0,
            original_image_path=request.original_image_path,
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


def test_batch_measurement_updates_existing_json_and_preserves_polygon(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "cropped.jpg"
    Image.new("L", (20, 15), color=100).save(image_path)
    label_path = image_path.with_suffix(".json")
    image_data = read_image_file(filename=str(image_path))
    write_label_file(
        filename=str(label_path),
        annotation=Annotation(
            image_path=image_path.name,
            image_data=image_data,
            shapes=[
                {
                    "label": "EMA",
                    "points": [[2.0, 6.0], [17.0, 6.0], [10.0, 14.0]],
                    "shape_type": "polygon",
                    "flags": {},
                    "description": "",
                    "group_id": None,
                    "mask": None,
                    "other_data": {},
                },
                {
                    "label": "EMW",
                    "points": [[1.0, 1.0], [2.0, 1.0]],
                    "shape_type": "line",
                    "flags": {},
                    "description": "old",
                    "group_id": None,
                    "mask": None,
                    "other_data": {},
                },
            ],
            flags={"reviewed": True},
            other_data={"custom": 42},
        ),
        image_height=15,
        image_width=20,
        save_image_data=False,
    )
    worker = UltrasoundInferenceWorker()
    worker._engine = cast(Any, _FakeMeasurementEngine())
    completed: list[BatchInferenceResult] = []
    worker.batch_completed.connect(completed.append)

    worker.infer_batch(
        BatchInferenceRequest(
            request_id="measure-batch",
            items=(
                BatchInferenceItem(
                    str(image_path),
                    str(label_path),
                    depth_setting_mm=100.0,
                    original_image_path="backend/data/raw/batch/cropped.jpg",
                ),
            ),
            tasks=("EMW",),
        )
    )

    assert completed[0].succeeded == (str(image_path),)
    annotation = read_label_file(filename=str(label_path))
    assert [shape["label"] for shape in annotation.shapes] == ["EMA", "EMW"]
    assert annotation.shapes[1]["points"] == [[2.0, 5.0], [12.0, 5.0]]
    assert annotation.flags == {"reviewed": True}
    assert annotation.other_data["custom"] == 42
    metadata = annotation.other_data["ultrasoundMetadata"]
    assert metadata["depthSettingMm"] == 100.0
    assert metadata["originalImagePath"].endswith("cropped.jpg")
    assert metadata["measurements"]["EMW"]["valueMm"] == 20.0
