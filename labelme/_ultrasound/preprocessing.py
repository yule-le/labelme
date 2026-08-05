"""Ultrasound image preprocessing with an explicit inverse geometry contract."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

from .contracts import GeometryTransform
from .contracts import PreparedInput
from .contracts import RoiPolicy


def prepare_grayscale_input(
    image: npt.NDArray[np.uint8],
    *,
    model_hw: tuple[int, int],
    roi_policy: RoiPolicy,
) -> PreparedInput:
    """Crop, resize, and normalize an image to ``[1, 1, H, W]`` float32."""

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {array.dtype}.")
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected grayscale or color image, got shape {array.shape}.")
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA image, got shape {array.shape}.")

    original_hw = (int(array.shape[0]), int(array.shape[1]))
    roi_xywh = _resolve_roi(original_hw=original_hw, policy=roi_policy)
    x, y, width, height = roi_xywh
    cropped = array[y : y + height, x : x + width]
    grayscale = _to_grayscale(cropped)

    model_h, model_w = model_hw
    resized = Image.fromarray(grayscale, mode="L").resize(
        (model_w, model_h), resample=Image.Resampling.BILINEAR
    )
    normalized = np.asarray(resized, dtype=np.float32) / np.float32(127.5) - 1.0
    tensor = np.ascontiguousarray(normalized[None, None, :, :], dtype=np.float32)
    return PreparedInput(
        tensor=tensor,
        transform=GeometryTransform(
            original_hw=original_hw,
            roi_xywh=roi_xywh,
            model_hw=model_hw,
        ),
    )


def restore_mask_to_original(
    mask: npt.NDArray[np.bool_],
    *,
    transform: GeometryTransform,
) -> npt.NDArray[np.bool_]:
    """Inverse-resize a model mask and place it in original-image coordinates."""

    model_mask = np.asarray(mask, dtype=np.bool_)
    if model_mask.shape != transform.model_hw:
        raise ValueError(
            f"Mask shape {model_mask.shape} does not match "
            f"model shape {transform.model_hw}."
        )
    x, y, roi_w, roi_h = transform.roi_xywh
    resized = Image.fromarray(model_mask.astype(np.uint8) * 255, mode="L").resize(
        (roi_w, roi_h), resample=Image.Resampling.NEAREST
    )
    roi_mask = np.asarray(resized, dtype=np.uint8) > 0
    original = np.zeros(transform.original_hw, dtype=np.bool_)
    original[y : y + roi_h, x : x + roi_w] = roi_mask
    return original


def _resolve_roi(
    *,
    original_hw: tuple[int, int],
    policy: RoiPolicy,
) -> tuple[int, int, int, int]:
    height, width = original_hw
    raw_width, raw_height = policy.raw_size_wh
    x, y, roi_width, roi_height = policy.rectangle_xywh

    if (width, height) == (raw_width, raw_height):
        if x < 0 or y < 0 or x + roi_width > width or y + roi_height > height:
            raise ValueError(
                f"ROI {policy.rectangle_xywh} exceeds raw image {(width, height)}."
            )
        return policy.rectangle_xywh
    if policy.accept_cropped_size and (width, height) == (roi_width, roi_height):
        return (0, 0, roi_width, roi_height)
    if policy.accept_variable_cropped_size:
        return (0, 0, width, height)

    raise ValueError(
        "Image size does not match the configured raw or cropped ROI size: "
        f"got {(width, height)}, expected {(raw_width, raw_height)} or "
        f"{(roi_width, roi_height)}."
    )


def _to_grayscale(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if image.ndim == 2:
        return image
    rgb = image[:, :, :3].astype(np.float32)
    # Match Pillow/OpenCV-style luminance closely while keeping the operation
    # explicit and independent of either optional library.
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)
