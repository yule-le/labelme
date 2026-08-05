from __future__ import annotations

from pathlib import Path

import pytest

from labelme._ultrasound.config import UltrasoundConfigError
from labelme._ultrasound.config import load_annotation_root
from labelme._ultrasound.config import load_model_spec


def _write_config(path: Path, weights: Path, model_config: Path) -> None:
    path.write_text(
        f"""
schema_version: ultrasound_models.v1
models:
  eye_muscle:
    model_id: seg_eye_muscle
    artifact_id: seg_eye_muscle__transverse__v1
    export_label: EMA
    weights_path: {weights.as_posix()}
    config_path: {model_config.as_posix()}
    threshold: 0.5
    roi:
      raw_size_wh: [800, 600]
      rectangle_xywh: [92, 31, 536, 536]
      accept_variable_cropped_size: true
""",
        encoding="utf-8",
    )


def test_load_independent_eye_muscle_model_spec(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pth"
    model_config = tmp_path / "model_config.yaml"
    weights.touch()
    model_config.touch()
    local_config = tmp_path / "models.local.yaml"
    _write_config(local_config, weights, model_config)

    spec = load_model_spec(target="eye_muscle", config_path=local_config)

    assert spec.model_id == "seg_eye_muscle"
    assert spec.artifact_id == "seg_eye_muscle__transverse__v1"
    assert spec.export_label == "EMA"
    assert spec.roi.rectangle_xywh == (92, 31, 536, 536)
    assert spec.roi.accept_variable_cropped_size is True


def test_multilabel_or_wrong_export_label_is_rejected(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pth"
    model_config = tmp_path / "model_config.yaml"
    weights.touch()
    model_config.touch()
    local_config = tmp_path / "models.local.yaml"
    _write_config(local_config, weights, model_config)
    text = local_config.read_text(encoding="utf-8")
    local_config.write_text(
        text.replace("export_label: EMA", "export_label: eye_muscle")
    )

    with pytest.raises(UltrasoundConfigError, match="must be 'EMA'"):
        load_model_spec(target="eye_muscle", config_path=local_config)


def test_non_boolean_variable_cropped_size_setting_is_rejected(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights.pth"
    model_config = tmp_path / "model_config.yaml"
    weights.touch()
    model_config.touch()
    local_config = tmp_path / "models.local.yaml"
    _write_config(local_config, weights, model_config)
    text = local_config.read_text(encoding="utf-8")
    local_config.write_text(
        text.replace(
            "accept_variable_cropped_size: true",
            "accept_variable_cropped_size: yes-please",
        ),
        encoding="utf-8",
    )

    with pytest.raises(UltrasoundConfigError, match="must be a boolean"):
        load_model_spec(target="eye_muscle", config_path=local_config)


def test_load_optional_annotation_root(tmp_path: Path) -> None:
    annotation_root = tmp_path / "processed" / "transverse"
    annotation_root.mkdir(parents=True)
    local_config = tmp_path / "models.local.yaml"
    local_config.write_text(
        "schema_version: ultrasound_models.v1\n"
        f"annotation_root: {annotation_root.as_posix()}\n",
        encoding="utf-8",
    )

    assert load_annotation_root(local_config) == annotation_root.resolve()


def test_missing_annotation_root_falls_back_to_default_behavior(
    tmp_path: Path,
) -> None:
    local_config = tmp_path / "models.local.yaml"
    local_config.write_text(
        "schema_version: ultrasound_models.v1\nannotation_root: null\n",
        encoding="utf-8",
    )

    assert load_annotation_root(local_config) is None
