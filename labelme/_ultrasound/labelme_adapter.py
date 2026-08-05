"""Convert framework-independent predictions into native Labelme shapes."""

from __future__ import annotations

import numpy as np

from labelme._shape import Shape

from .contracts import AnnotationPrediction


def prediction_to_shape(prediction: AnnotationPrediction) -> Shape:
    """Create an ordinary editable Labelme polygon."""

    if prediction.shape_type != "polygon" or len(prediction.points) < 3:
        raise ValueError(
            "Ultrasound annotations must be polygons with at least 3 points."
        )
    return Shape(
        label=prediction.label,
        shape_type="polygon",
        points=np.asarray(prediction.points, dtype=np.float64),
        flags={},
        description="",
        closed=True,
    )
