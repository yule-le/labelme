"""Configuration loading for ultrasound model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import cast

from labelme._yaml import safe_load

from .contracts import LabelmeUltrasoundLabel
from .contracts import ModelSpec
from .contracts import RoiPolicy
from .contracts import UltrasoundTarget

_SCHEMA_VERSION = "ultrasound_models.v1"


class UltrasoundConfigError(ValueError):
    """Raised when the local ultrasound model contract is invalid."""


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs/ultrasound/models.local.yaml"


def load_annotation_root(
    config_path: str | Path | None = None,
) -> Path | None:
    """Return the optional local directory shown first by Open Folder."""

    path = Path(config_path) if config_path is not None else default_config_path()
    if not path.is_file():
        return None
    payload = safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UltrasoundConfigError("Ultrasound configuration must be a mapping.")
    value = payload.get("annotation_root")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise UltrasoundConfigError(
            "annotation_root must be a non-empty directory path or null."
        )
    root = Path(value)
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise UltrasoundConfigError(
            f"annotation_root directory does not exist: {root}"
        )
    return root


def load_model_spec(
    *,
    target: UltrasoundTarget,
    config_path: str | Path | None = None,
) -> ModelSpec:
    path = Path(config_path) if config_path is not None else default_config_path()
    if not path.is_file():
        raise UltrasoundConfigError(
            f"Ultrasound model configuration not found: {path}. "
            "Copy configs/ultrasound/models.example.yaml to models.local.yaml "
            "and set the artifact paths."
        )

    payload = safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UltrasoundConfigError("Ultrasound configuration must be a mapping.")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise UltrasoundConfigError(
            f"Expected schema_version {_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}."
        )

    models = payload.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(target), dict):
        raise UltrasoundConfigError(f"Missing models.{target} configuration.")
    raw = cast(dict[str, Any], models[target])
    roi_raw = raw.get("roi")
    if not isinstance(roi_raw, dict):
        raise UltrasoundConfigError(f"models.{target}.roi must be a mapping.")

    weights_path = _resolve_path(path, raw.get("weights_path"), "weights_path")
    model_config_path = _resolve_path(
        path, raw.get("config_path"), "config_path"
    )
    threshold = float(raw.get("threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        raise UltrasoundConfigError(
            f"models.{target}.threshold must be between 0 and 1."
        )

    raw_size_wh = _int_tuple(
        roi_raw.get("raw_size_wh"), 2, f"models.{target}.roi.raw_size_wh"
    )
    rectangle_xywh = _int_tuple(
        roi_raw.get("rectangle_xywh"),
        4,
        f"models.{target}.roi.rectangle_xywh",
    )
    if any(value <= 0 for value in (*raw_size_wh, *rectangle_xywh[2:])):
        raise UltrasoundConfigError("ROI dimensions must be positive.")

    export_label = raw.get("export_label")
    expected_label = "EMA" if target == "eye_muscle" else "skin"
    if export_label != expected_label:
        raise UltrasoundConfigError(
            f"models.{target}.export_label must be {expected_label!r}."
        )

    return ModelSpec(
        model_id=_required_str(raw, "model_id", target),
        artifact_id=_required_str(raw, "artifact_id", target),
        target=target,
        export_label=cast(LabelmeUltrasoundLabel, export_label),
        weights_path=str(weights_path),
        config_path=str(model_config_path),
        threshold=threshold,
        roi=RoiPolicy(
            raw_size_wh=cast(tuple[int, int], raw_size_wh),
            rectangle_xywh=cast(tuple[int, int, int, int], rectangle_xywh),
            accept_cropped_size=_bool_value(
                roi_raw.get("accept_cropped_size", True),
                f"models.{target}.roi.accept_cropped_size",
            ),
            accept_variable_cropped_size=_bool_value(
                roi_raw.get("accept_variable_cropped_size", False),
                f"models.{target}.roi.accept_variable_cropped_size",
            ),
        ),
        min_component_area=int(raw.get("min_component_area", 64)),
        polygon_tolerance=float(raw.get("polygon_tolerance", 1.5)),
    )


def _resolve_path(config_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise UltrasoundConfigError(f"{field} must be a non-empty path.")
    result = Path(value)
    if not result.is_absolute():
        result = config_path.parent / result
    result = result.resolve()
    if not result.is_file():
        raise UltrasoundConfigError(f"{field} does not exist: {result}")
    return result


def _required_str(raw: dict[str, Any], field: str, target: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise UltrasoundConfigError(
            f"models.{target}.{field} must be a non-empty string."
        )
    return value


def _int_tuple(value: object, length: int, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list | tuple)
        or len(value) != length
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise UltrasoundConfigError(f"{field} must contain {length} integers.")
    return tuple(cast(int, item) for item in value)


def _bool_value(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise UltrasoundConfigError(f"{field} must be a boolean.")
    return value
