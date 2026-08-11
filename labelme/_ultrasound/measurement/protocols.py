"""Protocol parameters kept separate from measurement orchestration and UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EyeMuscleProtocol:
    method_version: str = "geom-depth-width-v1.1"
    contour_num_points: int = 256
    top_region_ratio: float = 0.38
    top_margin_ratio: float = 0.05
    center_band_ratio: float = 0.18


@dataclass(frozen=True)
class LayerDepthProtocol:
    method_version: str = "fat-skin-v2.0"
    axis_offset_mm: float = 0.0


@dataclass(frozen=True)
class MeasurementProtocols:
    eye_muscle: EyeMuscleProtocol = EyeMuscleProtocol()
    fat_depth: LayerDepthProtocol = LayerDepthProtocol()
    skin_depth: LayerDepthProtocol = LayerDepthProtocol()
