"""Framework-independent contracts for ultrasound auto-annotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type PolygonPoint = tuple[float, float]


@dataclass(frozen=True)
class AnnotationPrediction:
    """A model-agnostic polygon ready to be adapted into a Labelme shape."""

    label: Literal["eye_muscle", "fat", "skin"]
    points: tuple[PolygonPoint, ...]
    confidence: float | None
    strategy: Literal["boundary_derived", "multilabel"]
    shape_type: Literal["polygon"] = "polygon"
