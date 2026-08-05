"""Persistent Qt worker for non-blocking ultrasound inference."""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from PySide6 import QtCore

from labelme import _utils
from labelme._label_file import Annotation
from labelme._label_file import read_image_file
from labelme._label_file import write_label_file
from labelme._utils.shape import ShapeDict

from .contracts import BatchInferenceRequest
from .contracts import BatchInferenceResult
from .contracts import InferenceRequest
from .engine import UltrasoundInferenceEngine
from .labelme_adapter import prediction_to_shape


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
        for index, item in enumerate(request.items, start=1):
            if self._cancel_batch:
                break
            if Path(item.label_path).exists():
                skipped.append(item.image_path)
                self.batch_progress.emit(
                    request.request_id, index, total, item.image_path
                )
                continue
            try:
                image_data = read_image_file(filename=item.image_path)
                image = _utils.img_data_to_arr(img_data=image_data)
                result = self._engine.infer(
                    InferenceRequest(
                        request_id=f"{request.request_id}:{index}",
                        image_path=item.image_path,
                        image=image,
                        tasks=request.tasks,
                    )
                )
                shapes = [
                    _shape_to_dict(prediction_to_shape(prediction))
                    for prediction in result.predictions
                ]
                label_path = Path(item.label_path)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                write_label_file(
                    filename=str(label_path),
                    annotation=Annotation(
                        image_path=os.path.relpath(
                            item.image_path, label_path.parent
                        ),
                        image_data=image_data,
                        shapes=shapes,
                        flags={},
                        other_data={},
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
                failures.append(
                    (item.image_path, f"{type(exc).__name__}: {exc}")
                )
            self.batch_progress.emit(
                request.request_id, index, total, item.image_path
            )
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
