from __future__ import annotations

import numpy as np

from labelme._ultrasound.contracts import Calibration
from labelme._ultrasound.measurement import MeasurementEngine
from labelme._ultrasound.measurement.layer_depth import measure_layer_depth


def _eye_mask() -> np.ndarray:
    yy, xx = np.ogrid[:220, :320]
    return ((xx - 160) / 105) ** 2 + ((yy - 115) / 65) ** 2 <= 1.0


def test_measurement_engine_returns_four_versioned_segments() -> None:
    eye = _eye_mask()
    fat = np.zeros_like(eye)
    skin = np.zeros_like(eye)
    fat[30:55, 140:181] = True
    skin[15:28, 140:181] = True
    engine = MeasurementEngine()
    calibration = engine.calibration(depth_setting_mm=100.0, image_height_px=220)

    results = engine.measure(
        codes=("EMW", "EMD", "FD", "SD"),
        masks={"EMA": eye, "fat": fat, "skin": skin},
        calibration=calibration,
    )

    assert [result.code for result in results] == ["EMW", "EMD", "FD", "SD"]
    assert all(result.valid for result in results)
    assert all(result.segment is not None for result in results)
    assert all(
        result.value_mm is not None and result.value_mm > 0 for result in results
    )
    assert results[0].method_version == "geom-depth-width-v1.1"
    assert results[2].method_version == "fat-skin-v2.0"


def test_calibration_uses_cropped_image_height_and_depth_in_mm() -> None:
    calibration = MeasurementEngine.calibration(100.0, 536)

    assert calibration.depth_setting_mm == 100.0
    assert calibration.image_height_px == 536
    assert calibration.pixel_size_y_mm == 100.0 / 536.0
    assert calibration.pixel_size_x_mm == calibration.pixel_size_y_mm


def test_layer_depth_is_vertical_when_emd_axis_is_slanted() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:36, 48:53] = True
    calibration = Calibration(
        depth_setting_mm=100.0,
        image_height_px=100,
        pixel_size_x_mm=1.0,
        pixel_size_y_mm=1.0,
    )

    value, segment, diagnostics = measure_layer_depth(
        mask,
        ((60.0, 80.0), (50.0, 40.0)),
        calibration,
    )

    assert 15.0 <= value <= 16.0
    assert abs(segment[0][0] - 50.0) < 0.1
    assert abs(segment[1][0] - 50.0) < 0.1
    assert diagnostics["axisMode"] == "image_vertical_from_eye_top"
