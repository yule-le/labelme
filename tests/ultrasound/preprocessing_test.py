from __future__ import annotations

import numpy as np
import pytest

from labelme._ultrasound import RoiPolicy
from labelme._ultrasound.preprocessing import prepare_grayscale_input
from labelme._ultrasound.preprocessing import restore_mask_to_original

_ROI = RoiPolicy(
    raw_size_wh=(800, 600),
    rectangle_xywh=(92, 31, 536, 536),
)
_VARIABLE_CROPPED_ROI = RoiPolicy(
    raw_size_wh=(800, 600),
    rectangle_xywh=(92, 31, 536, 536),
    accept_variable_cropped_size=True,
)


def test_raw_image_uses_fixed_roi_and_restores_offset() -> None:
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image[31:567, 92:628] = 255

    prepared = prepare_grayscale_input(
        image, model_hw=(512, 512), roi_policy=_ROI
    )
    assert prepared.tensor.shape == (1, 1, 512, 512)
    assert prepared.tensor.dtype == np.float32
    assert np.allclose(prepared.tensor, 1.0)
    assert prepared.transform.roi_xywh == (92, 31, 536, 536)

    model_mask = np.zeros((512, 512), dtype=np.bool_)
    model_mask[100:200, 120:220] = True
    restored = restore_mask_to_original(
        model_mask, transform=prepared.transform
    )
    assert restored.shape == (600, 800)
    rows, columns = np.nonzero(restored)
    assert rows.min() >= 31
    assert columns.min() >= 92
    assert rows.max() < 567
    assert columns.max() < 628


def test_already_cropped_grayscale_image_has_zero_offset() -> None:
    image = np.full((536, 536), 127, dtype=np.uint8)
    prepared = prepare_grayscale_input(
        image, model_hw=(512, 512), roi_policy=_ROI
    )
    assert prepared.transform.roi_xywh == (0, 0, 536, 536)
    assert prepared.tensor.shape == (1, 1, 512, 512)


def test_variable_size_prepared_roi_uses_full_image_and_restores_shape() -> None:
    image = np.full((538, 599), 127, dtype=np.uint8)
    prepared = prepare_grayscale_input(
        image,
        model_hw=(512, 512),
        roi_policy=_VARIABLE_CROPPED_ROI,
    )

    assert prepared.transform.original_hw == (538, 599)
    assert prepared.transform.roi_xywh == (0, 0, 599, 538)
    assert prepared.tensor.shape == (1, 1, 512, 512)

    model_mask = np.ones((512, 512), dtype=np.bool_)
    restored = restore_mask_to_original(
        model_mask,
        transform=prepared.transform,
    )
    assert restored.shape == (538, 599)
    assert restored.all()


def test_raw_size_takes_precedence_over_variable_cropped_roi() -> None:
    image = np.zeros((600, 800), dtype=np.uint8)
    prepared = prepare_grayscale_input(
        image,
        model_hw=(512, 512),
        roi_policy=_VARIABLE_CROPPED_ROI,
    )
    assert prepared.transform.roi_xywh == (92, 31, 536, 536)


def test_unknown_image_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        prepare_grayscale_input(
            np.zeros((400, 400), dtype=np.uint8),
            model_hw=(512, 512),
            roi_policy=_ROI,
        )
