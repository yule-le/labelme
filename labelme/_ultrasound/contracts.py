"""Framework-independent contracts for ultrasound auto-annotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

type PolygonPoint = tuple[float, float]
type UltrasoundTarget = Literal["eye_muscle", "skin"]
type LabelmeUltrasoundLabel = Literal["EMA", "skin", "fat"]
type MeasurementCode = Literal["EMW", "EMD", "FD", "SD"]
type UltrasoundTask = LabelmeUltrasoundLabel | MeasurementCode


@dataclass(frozen=True)
class RoiPolicy:
    """A fixed raw-image ROI with explicit prepared-ROI acceptance rules."""

    raw_size_wh: tuple[int, int]
    rectangle_xywh: tuple[int, int, int, int]
    accept_cropped_size: bool = True
    accept_variable_cropped_size: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """Validated runtime contract for one single-target segmentation model."""

    model_id: str
    artifact_id: str
    target: UltrasoundTarget
    export_label: LabelmeUltrasoundLabel
    weights_path: str
    config_path: str
    threshold: float
    roi: RoiPolicy
    min_component_area: int = 64
    polygon_tolerance: float = 1.5


@dataclass(frozen=True)
class GeometryTransform:
    """Reversible mapping from original image through ROI to model space."""

    original_hw: tuple[int, int]
    roi_xywh: tuple[int, int, int, int]
    model_hw: tuple[int, int]


@dataclass(frozen=True)
class PreparedInput:
    """Model input plus the geometry needed to restore predictions."""

    tensor: npt.NDArray[np.float32]
    transform: GeometryTransform


@dataclass(frozen=True)
class InferenceRequest:
    """Immutable image snapshot submitted to a background worker."""

    request_id: str
    image_path: str
    image: npt.NDArray[np.uint8]
    tasks: tuple[UltrasoundTask, ...] = ("EMA", "fat", "skin")
    depth_setting_mm: float | None = None
    original_image_path: str | None = None
    existing_segmentations: tuple[ExistingSegmentation, ...] = ()


@dataclass(frozen=True)
class InferenceResult:
    """Background result carrying the originating image identity."""

    request_id: str
    image_path: str
    predictions: tuple[AnnotationPrediction, ...]
    measurements: tuple[MeasurementResult, ...] = ()
    calibration: Calibration | None = None
    depth_setting_mm: float | None = None
    original_image_path: str | None = None


@dataclass(frozen=True)
class BatchInferenceItem:
    """One image and its destination Labelme JSON path."""

    image_path: str
    label_path: str
    depth_setting_mm: float | None = None
    original_image_path: str | None = None


@dataclass(frozen=True)
class BatchInferenceRequest:
    """A folder batch that creates annotations only when JSON is absent."""

    request_id: str
    items: tuple[BatchInferenceItem, ...]
    tasks: tuple[UltrasoundTask, ...]
    save_image_data: bool = False


@dataclass(frozen=True)
class BatchInferenceResult:
    """Batch completion summary without stopping on individual failures."""

    request_id: str
    succeeded: tuple[str, ...]
    skipped: tuple[str, ...]
    failures: tuple[tuple[str, str], ...]
    cancelled: bool = False


@dataclass(frozen=True)
class AnnotationPrediction:
    """A model-agnostic polygon ready to be adapted into a Labelme shape."""

    label: LabelmeUltrasoundLabel
    points: tuple[PolygonPoint, ...]
    confidence: float | None
    source_target: Literal["eye_muscle", "skin", "derived_fat"]
    strategy: Literal["single_model", "boundary_derived"]
    model_id: str | None = None
    artifact_id: str | None = None
    shape_type: Literal["polygon"] = "polygon"


@dataclass(frozen=True)
class ProcessedSegmentation:
    """Cleaned original-image mask paired with its editable annotation."""

    annotation: AnnotationPrediction
    mask: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ExistingSegmentation:
    """An editable Labelme polygon rasterized in the opened image space."""

    label: LabelmeUltrasoundLabel
    mask: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Calibration:
    """Per-image physical scale derived from the cropped image and CSV depth."""

    depth_setting_mm: float
    image_height_px: int
    pixel_size_x_mm: float
    pixel_size_y_mm: float
    source: Literal["csv"] = "csv"


@dataclass(frozen=True)
class MeasurementResult:
    """Framework-independent measurement value and optional editable segment."""

    code: MeasurementCode
    value_mm: float | None
    segment: tuple[PolygonPoint, PolygonPoint] | None
    valid: bool
    method_version: str
    error: str | None = None
    diagnostics: dict[str, object] | None = None
