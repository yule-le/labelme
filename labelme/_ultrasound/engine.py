"""Qt-independent orchestration of one ultrasound segmentation request."""

from __future__ import annotations

from pathlib import Path

from .adapters.pytorch import PyTorchSegmentationAdapter
from .config import load_model_spec
from .contracts import AnnotationPrediction
from .contracts import ExistingSegmentation
from .contracts import InferenceRequest
from .contracts import InferenceResult
from .contracts import ProcessedSegmentation
from .contracts import UltrasoundTarget
from .measurement import MEASUREMENT_CODES
from .measurement import MeasurementEngine
from .postprocessing import probability_to_segmentation
from .preprocessing import prepare_grayscale_input
from .strategies import derive_fat_segmentation


class UltrasoundInferenceEngine:
    """Cache independent target adapters and return framework-free predictions."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config_path = config_path
        self._adapters: dict[UltrasoundTarget, PyTorchSegmentationAdapter] = {}
        self._measurement_engine = MeasurementEngine()

    def infer(self, request: InferenceRequest) -> InferenceResult:
        requested = set(request.tasks)
        if not requested:
            raise ValueError("Select at least one ultrasound annotation task.")
        unknown = requested - {"EMA", "fat", "skin", *MEASUREMENT_CODES}
        if unknown:
            raise ValueError(f"Unsupported ultrasound tasks: {sorted(unknown)}.")

        measurement_codes = tuple(
            code for code in MEASUREMENT_CODES if code in requested
        )
        calibration = None
        if measurement_codes:
            if request.depth_setting_mm is None:
                raise ValueError(
                    "Selected measurements require depth_ocr_mm from an imported CSV."
                )
            calibration = self._measurement_engine.calibration(
                request.depth_setting_mm, int(request.image.shape[0])
            )
        existing = {
            item.label: _existing_to_processed(item)
            for item in request.existing_segmentations
        }
        prefer_existing = bool(measurement_codes)
        needs_eye = bool(requested & {"EMA", "fat", *MEASUREMENT_CODES})
        needs_skin = bool(requested & {"skin", "fat", "FD", "SD"})
        eye_muscle = existing.get("EMA") if prefer_existing else None
        if eye_muscle is None and needs_eye:
            eye_muscle = self._infer_target(request, target="eye_muscle")
        skin = existing.get("skin") if prefer_existing else None
        if skin is None and needs_skin:
            skin = self._infer_target(request, target="skin")

        predictions = []
        if "EMA" in requested and "EMA" not in existing:
            if eye_muscle is None:
                raise ValueError("The eye-muscle model found no EMA region.")
            predictions.append(eye_muscle.annotation)
        fat = existing.get("fat") if prefer_existing else None
        if ("fat" in requested or "FD" in requested) and fat is None:
            if eye_muscle is None:
                raise ValueError("The eye-muscle model found no EMA boundary for fat.")
            if skin is None:
                raise ValueError("The skin model found no skin boundary for fat.")
            fat = derive_fat_segmentation(
                skin=skin,
                eye_muscle=eye_muscle,
            )
            if fat is None:
                raise ValueError(
                    "Could not derive fat between the skin and EMA boundaries."
                )
        if "fat" in requested and "fat" not in existing:
            assert fat is not None
            predictions.append(fat.annotation)
        if "skin" in requested and "skin" not in existing:
            if skin is None:
                raise ValueError("The skin model found no skin region.")
            predictions.append(skin.annotation)

        measurements = ()
        if measurement_codes:
            if eye_muscle is None:
                raise ValueError("EMA segmentation is required for measurement.")
            assert calibration is not None
            masks = {"EMA": eye_muscle.mask}
            if fat is not None:
                masks["fat"] = fat.mask
            if skin is not None:
                masks["skin"] = skin.mask
            measurements = self._measurement_engine.measure(
                codes=measurement_codes,
                masks=masks,
                calibration=calibration,
            )

        return InferenceResult(
            request_id=request.request_id,
            image_path=request.image_path,
            predictions=tuple(predictions),
            measurements=measurements,
            calibration=calibration,
            depth_setting_mm=request.depth_setting_mm,
            original_image_path=request.original_image_path,
        )

    def _infer_target(
        self,
        request: InferenceRequest,
        *,
        target: UltrasoundTarget,
    ) -> ProcessedSegmentation | None:
        adapter = self._adapter(target)
        prepared = prepare_grayscale_input(
            request.image,
            model_hw=adapter.model_hw,
            roi_policy=adapter.spec.roi,
        )
        probability = adapter.predict_probability(prepared.tensor)
        return probability_to_segmentation(
            probability,
            spec=adapter.spec,
            transform=prepared.transform,
        )

    def _adapter(self, target: UltrasoundTarget) -> PyTorchSegmentationAdapter:
        if target not in self._adapters:
            spec = load_model_spec(target=target, config_path=self._config_path)
            self._adapters[target] = PyTorchSegmentationAdapter(spec)
        return self._adapters[target]


def _existing_to_processed(item: ExistingSegmentation) -> ProcessedSegmentation:
    source_target = (
        "eye_muscle"
        if item.label == "EMA"
        else "skin"
        if item.label == "skin"
        else "derived_fat"
    )
    strategy = "boundary_derived" if item.label == "fat" else "single_model"
    return ProcessedSegmentation(
        annotation=AnnotationPrediction(
            label=item.label,
            points=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            confidence=None,
            source_target=source_target,
            strategy=strategy,
        ),
        mask=item.mask,
    )
