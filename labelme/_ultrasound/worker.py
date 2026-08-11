"""Persistent Qt worker for non-blocking ultrasound inference."""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from PySide6 import QtCore

from labelme import _utils
from labelme._label_file import Annotation
from labelme._label_file import read_image_file
from labelme._label_file import read_label_file
from labelme._label_file import write_label_file
from labelme._utils.shape import ShapeDict

from .contracts import BatchInferenceRequest
from .contracts import BatchInferenceResult
from .contracts import ExistingSegmentation
from .contracts import InferenceRequest
from .engine import UltrasoundInferenceEngine
from .labelme_adapter import measurement_to_shape
from .labelme_adapter import prediction_to_shape
from .labelme_adapter import update_depth_metadata
from .labelme_adapter import update_ultrasound_metadata
from .measurement import MEASUREMENT_CODES


class UltrasoundInferenceWorker(QtCore.QObject):
    """Own a cached inference engine and never touch UI objects."""

    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str, str, str)
    batch_progress = QtCore.Signal(str, int, int, str)
    batch_completed = QtCore.Signal(object)

    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__()
        self._engine = UltrasoundInferenceEngine(config_path=config_path)
        self._cancel_batch = False

    @QtCore.Slot(object)
    def infer(self, request: object) -> None:
        if not isinstance(request, InferenceRequest):
            raise TypeError("Expected InferenceRequest.")
        try:
            result = self._engine.infer(request)
        except Exception as exc:
            logger.opt(exception=exc).error("Ultrasound inference failed")
            self.failed.emit(
                request.request_id,
                request.image_path,
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.completed.emit(result)

    def cancel_batch(self) -> None:
        """Request cancellation between images in the active batch."""

        self._cancel_batch = True

    @QtCore.Slot(object)
    def infer_batch(self, request: object) -> None:
        if not isinstance(request, BatchInferenceRequest):
            raise TypeError("Expected BatchInferenceRequest.")
        self._cancel_batch = False
        succeeded: list[str] = []
        skipped: list[str] = []
        failures: list[tuple[str, str]] = []
        total = len(request.items)
        has_measurements = bool(set(request.tasks) & set(MEASUREMENT_CODES))
        for index, item in enumerate(request.items, start=1):
            if self._cancel_batch:
                break
            if Path(item.label_path).exists() and not has_measurements:
                skipped.append(item.image_path)
                self.batch_progress.emit(
                    request.request_id, index, total, item.image_path
                )
                continue
            try:
                image_data = read_image_file(filename=item.image_path)
                image = _utils.img_data_to_arr(img_data=image_data)
                label_path = Path(item.label_path)
                existing_annotation = (
                    read_label_file(filename=str(label_path))
                    if label_path.exists()
                    else None
                )
                result = self._engine.infer(
                    InferenceRequest(
                        request_id=f"{request.request_id}:{index}",
                        image_path=item.image_path,
                        image=image,
                        tasks=request.tasks,
                        depth_setting_mm=item.depth_setting_mm,
                        original_image_path=item.original_image_path,
                        existing_segmentations=(
                            _existing_segmentations(
                                existing_annotation.shapes,
                                image_shape=image.shape[:2],
                            )
                            if existing_annotation is not None and has_measurements
                            else ()
                        ),
                    )
                )
                generated_shapes = [
                    _shape_to_dict(prediction_to_shape(prediction))
                    for prediction in result.predictions
                ]
                generated_shapes.extend(
                    _shape_to_dict(shape)
                    for measurement in result.measurements
                    if (shape := measurement_to_shape(measurement)) is not None
                )
                replaced_labels = {shape["label"] for shape in generated_shapes} | {
                    code for code in MEASUREMENT_CODES if code in request.tasks
                }
                retained_shapes = (
                    [
                        shape
                        for shape in existing_annotation.shapes
                        if shape["label"] not in replaced_labels
                    ]
                    if existing_annotation is not None
                    else []
                )
                shapes = [*retained_shapes, *generated_shapes]
                other_data = (
                    dict(existing_annotation.other_data)
                    if existing_annotation is not None
                    else {}
                )
                if item.depth_setting_mm is not None:
                    update_depth_metadata(
                        other_data,
                        depth_setting_mm=item.depth_setting_mm,
                        original_image_path=item.original_image_path,
                    )
                if result.calibration is not None:
                    update_ultrasound_metadata(
                        other_data,
                        calibration=result.calibration,
                        original_image_path=result.original_image_path,
                        measurements=result.measurements,
                    )
                label_path.parent.mkdir(parents=True, exist_ok=True)
                write_label_file(
                    filename=str(label_path),
                    annotation=Annotation(
                        image_path=os.path.relpath(item.image_path, label_path.parent),
                        image_data=image_data,
                        shapes=shapes,
                        flags=(
                            existing_annotation.flags
                            if existing_annotation is not None
                            else {}
                        ),
                        other_data=other_data,
                    ),
                    image_height=int(image.shape[0]),
                    image_width=int(image.shape[1]),
                    save_image_data=request.save_image_data,
                )
                succeeded.append(item.image_path)
            except Exception as exc:
                logger.opt(exception=exc).error(
                    "Ultrasound batch inference failed for {!r}",
                    item.image_path,
                )
                failures.append((item.image_path, f"{type(exc).__name__}: {exc}"))
            self.batch_progress.emit(request.request_id, index, total, item.image_path)
        self.batch_completed.emit(
            BatchInferenceResult(
                request_id=request.request_id,
                succeeded=tuple(succeeded),
                skipped=tuple(skipped),
                failures=tuple(failures),
                cancelled=self._cancel_batch,
            )
        )


def _shape_to_dict(shape: object) -> ShapeDict:
    from labelme._shape import Shape

    if not isinstance(shape, Shape):
        raise TypeError("Expected Shape.")
    assert shape.label is not None
    return ShapeDict(
        label=shape.label,
        points=shape.points.tolist(),
        shape_type=shape.shape_type,
        flags={},
        description="",
        group_id=shape.group_id,
        mask=shape.mask,
        other_data=shape.other_data,
    )


def _existing_segmentations(
    shapes: list[ShapeDict], *, image_shape: tuple[int, int]
) -> tuple[ExistingSegmentation, ...]:
    output: list[ExistingSegmentation] = []
    for label in ("EMA", "fat", "skin"):
        matching = [
            shape
            for shape in shapes
            if shape["label"] == label and shape["shape_type"] == "polygon"
        ]
        if len(matching) != 1:
            continue
        shape = matching[0]
        mask = _utils.shape_to_mask(
            img_shape=image_shape,
            points=shape["points"],
            shape_type="polygon",
        )
        output.append(ExistingSegmentation(label=label, mask=mask))
    return tuple(output)
