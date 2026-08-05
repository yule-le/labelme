"""Qt-independent orchestration of one ultrasound segmentation request."""

from __future__ import annotations

from pathlib import Path

from .adapters.pytorch import PyTorchSegmentationAdapter
from .config import load_model_spec
from .contracts import InferenceRequest
from .contracts import InferenceResult
from .contracts import ProcessedSegmentation
from .contracts import UltrasoundTarget
from .postprocessing import probability_to_segmentation
from .preprocessing import prepare_grayscale_input
from .strategies import derive_fat_segmentation


class UltrasoundInferenceEngine:
    """Cache independent target adapters and return framework-free predictions."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config_path = config_path
        self._adapters: dict[UltrasoundTarget, PyTorchSegmentationAdapter] = {}

    def infer(self, request: InferenceRequest) -> InferenceResult:
        requested = set(request.tasks)
        if not requested:
            raise ValueError("Select at least one ultrasound annotation task.")
        unknown = requested - {"EMA", "fat", "skin"}
        if unknown:
            raise ValueError(f"Unsupported ultrasound tasks: {sorted(unknown)}.")

        eye_muscle = (
            self._infer_target(request, target="eye_muscle")
            if requested & {"EMA", "fat"}
            else None
        )
        skin = (
            self._infer_target(request, target="skin")
            if requested & {"skin", "fat"}
            else None
        )

        predictions = []
        if "EMA" in requested:
            if eye_muscle is None:
                raise ValueError("The eye-muscle model found no EMA region.")
            predictions.append(eye_muscle.annotation)
        if "fat" in requested:
            if eye_muscle is None:
                raise ValueError(
                    "The eye-muscle model found no EMA boundary for fat."
                )
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
            predictions.append(fat.annotation)
        if "skin" in requested:
            if skin is None:
                raise ValueError("The skin model found no skin region.")
            predictions.append(skin.annotation)

        return InferenceResult(
            request_id=request.request_id,
            image_path=request.image_path,
            predictions=tuple(predictions),
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

    def _adapter(
        self, target: UltrasoundTarget
    ) -> PyTorchSegmentationAdapter:
        if target not in self._adapters:
            spec = load_model_spec(
                target=target, config_path=self._config_path
            )
            self._adapters[target] = PyTorchSegmentationAdapter(spec)
        return self._adapters[target]
