"""Registry-driven orchestration for versioned ultrasound measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import Calibration
from ..contracts import MeasurementCode
from ..contracts import MeasurementResult
from .eye_muscle import EyeMuscleGeometry
from .eye_muscle import measure_eye_muscle
from .layer_depth import measure_layer_depth
from .protocols import MeasurementProtocols

MEASUREMENT_CODES: tuple[MeasurementCode, ...] = ("EMW", "EMD", "FD", "SD")


@dataclass(frozen=True)
class MeasurementDefinition:
    code: MeasurementCode
    required_labels: tuple[str, ...]
    method_version: str


class MeasurementEngine:
    """Measure selected codes without depending on Qt or Labelme shapes."""

    def __init__(self, protocols: MeasurementProtocols | None = None) -> None:
        self.protocols = protocols or MeasurementProtocols()
        self.definitions = {
            "EMW": MeasurementDefinition(
                "EMW", ("EMA",), self.protocols.eye_muscle.method_version
            ),
            "EMD": MeasurementDefinition(
                "EMD", ("EMA",), self.protocols.eye_muscle.method_version
            ),
            "FD": MeasurementDefinition(
                "FD", ("EMA", "fat"), self.protocols.fat_depth.method_version
            ),
            "SD": MeasurementDefinition(
                "SD", ("EMA", "skin"), self.protocols.skin_depth.method_version
            ),
        }

    @staticmethod
    def calibration(depth_setting_mm: float, image_height_px: int) -> Calibration:
        if depth_setting_mm <= 0.0:
            raise ValueError("depth_setting must be greater than zero millimetres.")
        if image_height_px <= 0:
            raise ValueError("Cropped image height must be greater than zero.")
        pixel_size = float(depth_setting_mm) / float(image_height_px)
        return Calibration(
            depth_setting_mm=float(depth_setting_mm),
            image_height_px=int(image_height_px),
            pixel_size_x_mm=pixel_size,
            pixel_size_y_mm=pixel_size,
        )

    def measure(
        self,
        *,
        codes: tuple[MeasurementCode, ...],
        masks: dict[str, np.ndarray],
        calibration: Calibration,
    ) -> tuple[MeasurementResult, ...]:
        selected = set(codes)
        unknown = selected - set(MEASUREMENT_CODES)
        if unknown:
            raise ValueError(f"Unsupported measurements: {sorted(unknown)}")
        results: dict[MeasurementCode, MeasurementResult] = {}
        eye_geometry: EyeMuscleGeometry | None = None
        if selected & {"EMW", "EMD", "FD", "SD"}:
            try:
                eye_geometry = measure_eye_muscle(
                    masks["EMA"], calibration, self.protocols.eye_muscle
                )
            except Exception as exc:
                for code in selected & {"EMW", "EMD", "FD", "SD"}:
                    definition = self.definitions[code]
                    results[code] = MeasurementResult(
                        code=code,
                        value_mm=None,
                        segment=None,
                        valid=False,
                        method_version=definition.method_version,
                        error=str(exc),
                    )

        if eye_geometry is not None:
            if "EMW" in selected:
                results["EMW"] = MeasurementResult(
                    code="EMW",
                    value_mm=eye_geometry.width_mm,
                    segment=eye_geometry.width_segment,
                    valid=True,
                    method_version=self.definitions["EMW"].method_version,
                    diagnostics=eye_geometry.diagnostics,
                )
            if "EMD" in selected:
                results["EMD"] = MeasurementResult(
                    code="EMD",
                    value_mm=eye_geometry.depth_mm,
                    segment=eye_geometry.depth_segment,
                    valid=True,
                    method_version=self.definitions["EMD"].method_version,
                    diagnostics=eye_geometry.diagnostics,
                )
            eye_depth_segment = eye_geometry.depth_segment
            for code, label, protocol in (
                ("FD", "fat", self.protocols.fat_depth),
                ("SD", "skin", self.protocols.skin_depth),
            ):
                if code not in selected:
                    continue
                try:
                    value, segment, layer_diagnostics = measure_layer_depth(
                        masks[label],
                        eye_depth_segment,
                        calibration,
                        axis_offset_mm=protocol.axis_offset_mm,
                    )
                    results[code] = MeasurementResult(
                        code=code,
                        value_mm=value,
                        segment=segment,
                        valid=True,
                        method_version=self.definitions[code].method_version,
                        diagnostics=layer_diagnostics,
                    )
                except Exception as exc:
                    results[code] = MeasurementResult(
                        code=code,
                        value_mm=None,
                        segment=None,
                        valid=False,
                        method_version=self.definitions[code].method_version,
                        error=str(exc),
                    )
        return tuple(results[code] for code in MEASUREMENT_CODES if code in selected)
