"""Lazy PyTorch adapter for published single-target segmentation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from labelme._yaml import safe_load

from ..contracts import ModelSpec


class ModelDependencyError(RuntimeError):
    """Raised when the optional PyTorch runtime is unavailable."""


class PyTorchSegmentationAdapter:
    """Load one state-dict artifact once and reuse it for inference."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model: Any | None = None
        self._device: Any | None = None
        self._model_hw: tuple[int, int] | None = None

    @property
    def model_hw(self) -> tuple[int, int]:
        if self._model_hw is None:
            self._read_and_validate_config()
        assert self._model_hw is not None
        return self._model_hw

    def predict_probability(
        self, tensor: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        self._ensure_loaded()
        torch = _import_torch()
        assert self._device is not None
        model = self._model
        assert callable(model)
        with torch.no_grad():
            output = model(torch.from_numpy(tensor).to(self._device))
        if isinstance(output, dict):
            output = output.get("out")
        if output is None or tuple(output.shape[:2]) != (1, 1):
            shape = None if output is None else tuple(output.shape)
            raise ValueError(f"Expected model output [1, 1, H, W], got {shape}.")
        probability = torch.sigmoid(output[0, 0]).detach().cpu().numpy()
        return np.asarray(probability, dtype=np.float32)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        config = self._read_and_validate_config()
        torch = _import_torch()
        arch = config["arch"]
        if arch == "deeplabv3_resnet50":
            try:
                from torchvision.models.segmentation import deeplabv3_resnet50
            except Exception as exc:
                raise ModelDependencyError(
                    "Ultrasound EMA inference requires torchvision."
                ) from exc
            model = deeplabv3_resnet50(
                weights=None, weights_backbone=None, aux_loss=False
            )
            model.classifier[4] = torch.nn.Conv2d(
                256, int(config["out_channels"]), kernel_size=1
            )
            built_model = _deep_lab_wrapper(torch=torch, model=model)
        elif arch == "unet_standard":
            built_model = _standard_unet(
                torch=torch,
                in_channels=int(config["in_channels"]),
                out_channels=int(config["out_channels"]),
            )
        else:
            raise ValueError(
                f"Unsupported ultrasound model architecture: {arch!r}."
            )

        loaded = torch.load(self.spec.weights_path, map_location="cpu")
        state_dict = _extract_state_dict(loaded, torch=torch)
        if arch == "deeplabv3_resnet50" and not any(
            key.startswith("model.") for key in state_dict
        ):
            state_dict = {f"model.{key}": value for key, value in state_dict.items()}
        built_model.load_state_dict(state_dict)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        built_model.to(self._device)
        built_model.eval()
        self._model = built_model

    def _read_and_validate_config(self) -> dict[str, Any]:
        config_path = Path(self.spec.config_path)
        config = safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Model config must be a mapping: {config_path}")
        expected = {
            "schema_version": "model_config.v1",
            "in_channels": 1,
            "out_channels": 1,
            "output_semantics": "single_channel_logits",
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(
                    f"Model config {key} must be {value!r}, got {config.get(key)!r}."
                )
        input_config = config.get("input")
        if not isinstance(input_config, dict):
            raise ValueError("Model config input must be a mapping.")
        if input_config.get("plane") != "transverse":
            raise ValueError("Only transverse ultrasound models are supported.")
        if input_config.get("target") != self.spec.target:
            raise ValueError(
                f"Configured target {self.spec.target!r} does not match artifact."
            )
        fixed_size = input_config.get("fixed_size_hw")
        if (
            not isinstance(fixed_size, list)
            or len(fixed_size) != 2
            or any(not isinstance(value, int) for value in fixed_size)
        ):
            raise ValueError("Model input.fixed_size_hw must contain two integers.")
        self._model_hw = (fixed_size[0], fixed_size[1])

        preprocess = config.get("preprocess")
        if not isinstance(preprocess, dict) or preprocess.get("grayscale") is not True:
            raise ValueError("Model preprocess.grayscale must be true.")
        normalize = preprocess.get("normalize")
        if not isinstance(normalize, dict) or (
            normalize.get("type"),
            normalize.get("mean"),
            normalize.get("std"),
        ) != ("mean_std", [0.5], [0.5]):
            raise ValueError(
                "Model normalization must be mean_std with mean/std [0.5]/[0.5]."
            )
        return config


def _import_torch() -> Any:  # noqa: ANN401
    try:
        import torch
    except ImportError as exc:
        raise ModelDependencyError(
            "Ultrasound auto-annotation requires PyTorch and torchvision. "
            "Install the ultrasound optional dependencies first."
        ) from exc
    return torch


def _extract_state_dict(
    loaded: object, *, torch: Any  # noqa: ANN401
) -> dict[str, Any]:
    if isinstance(loaded, torch.nn.Module):
        raise ValueError(
            "Full model objects are not accepted; expected a state_dict artifact."
        )
    if not isinstance(loaded, dict):
        raise ValueError("weights.pth must contain a state_dict.")
    candidate = loaded.get("model", loaded)
    if not isinstance(candidate, dict) or not candidate:
        raise ValueError("weights.pth contains an empty or invalid state_dict.")
    if any(not isinstance(value, torch.Tensor) for value in candidate.values()):
        raise ValueError("Every state_dict value must be a torch.Tensor.")
    state_dict = dict(candidate)
    for prefix in ("model.", "module."):
        if all(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key[len(prefix) :]: value for key, value in state_dict.items()
            }
    return state_dict


def _deep_lab_wrapper(*, torch: Any, model: Any) -> Any:  # noqa: ANN401
    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, tensor: Any) -> Any:  # noqa: ANN401
            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            output = self.model(tensor)
            return output["out"] if isinstance(output, dict) else output

    return Wrapper()


def _standard_unet(
    *,
    torch: Any,  # noqa: ANN401
    in_channels: int,
    out_channels: int,
) -> Any:  # noqa: ANN401
    class DoubleConv(torch.nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.double_conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    input_channels, output_channels, kernel_size=3, padding=1
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(
                    output_channels, output_channels, kernel_size=3, padding=1
                ),
                torch.nn.ReLU(inplace=True),
            )

        def forward(self, tensor: Any) -> Any:  # noqa: ANN401
            return self.double_conv(tensor)

    class StandardUNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1 = DoubleConv(in_channels, 64)
            self.pool1 = torch.nn.MaxPool2d(2)
            self.down2 = DoubleConv(64, 128)
            self.pool2 = torch.nn.MaxPool2d(2)
            self.down3 = DoubleConv(128, 256)
            self.pool3 = torch.nn.MaxPool2d(2)
            self.down4 = DoubleConv(256, 512)
            self.pool4 = torch.nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(512, 1024)
            self.up4 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.conv4 = DoubleConv(1024, 512)
            self.up3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.conv3 = DoubleConv(512, 256)
            self.up2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.conv2 = DoubleConv(256, 128)
            self.up1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.conv1 = DoubleConv(128, 64)
            self.out_conv = torch.nn.Conv2d(64, out_channels, kernel_size=1)

        @staticmethod
        def _match_hw(tensor: Any, reference: Any) -> Any:  # noqa: ANN401
            _, _, height, width = tensor.shape
            _, _, ref_height, ref_width = reference.shape
            if height > ref_height:
                difference = height - ref_height
                top = difference // 2
                tensor = tensor[:, :, top : top + ref_height, :]
            if width > ref_width:
                difference = width - ref_width
                left = difference // 2
                tensor = tensor[:, :, :, left : left + ref_width]
            _, _, height, width = tensor.shape
            pad_height = max(0, ref_height - height)
            pad_width = max(0, ref_width - width)
            if pad_height or pad_width:
                top = pad_height // 2
                left = pad_width // 2
                tensor = torch.nn.functional.pad(
                    tensor,
                    (
                        left,
                        pad_width - left,
                        top,
                        pad_height - top,
                    ),
                )
            return tensor

        def forward(self, tensor: Any) -> Any:  # noqa: ANN401
            level1 = self.down1(tensor)
            level2 = self.pool1(level1)
            level3 = self.down2(level2)
            level4 = self.pool2(level3)
            level5 = self.down3(level4)
            level6 = self.pool3(level5)
            level7 = self.down4(level6)
            level8 = self.pool4(level7)
            tensor = self.bottleneck(level8)
            tensor = self._decode(tensor, level7, self.up4, self.conv4)
            tensor = self._decode(tensor, level5, self.up3, self.conv3)
            tensor = self._decode(tensor, level3, self.up2, self.conv2)
            tensor = self._decode(tensor, level1, self.up1, self.conv1)
            return self.out_conv(tensor)

        def _decode(
            self,
            tensor: Any,  # noqa: ANN401
            skip: Any,  # noqa: ANN401
            upsample: Any,  # noqa: ANN401
            convolution: Any,  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            tensor = upsample(tensor)
            tensor = self._match_hw(tensor, skip)
            return convolution(torch.cat([skip, tensor], dim=1))

    return StandardUNet()
