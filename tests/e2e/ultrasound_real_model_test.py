from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot
from skimage.draw import polygon

from labelme._ultrasound.config import default_config_path

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_IMAGE = _REPO_ROOT / "test_data/ultrasound/transverse_ema_golden.jpg"
_HAS_LOCAL_RUNTIME = (
    default_config_path().is_file()
    and _GOLDEN_IMAGE.is_file()
    and importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchvision") is not None
)


@pytest.mark.gui
@pytest.mark.ultrasound_model
@pytest.mark.skipif(
    not _HAS_LOCAL_RUNTIME,
    reason="requires models.local.yaml plus PyTorch and torchvision",
)
def test_real_ema_model_runs_in_worker_and_adds_shape(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    pause: bool,
) -> None:
    win = main_win(
        file_or_dir=str(_GOLDEN_IMAGE),
        config_overrides={"auto_save": False},
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    heartbeat: list[bool] = []
    QtCore.QTimer.singleShot(50, lambda: heartbeat.append(True))
    win._start_ultrasound_auto_label()

    qtbot.waitUntil(lambda: bool(heartbeat), timeout=2_000)
    qtbot.waitUntil(
        lambda: len(win._canvas_widgets.canvas.shapes) == 2,
        timeout=60_000,
    )
    assert [
        shape.label for shape in win._canvas_widgets.canvas.shapes
    ] == ["EMA", "skin"]
    win._generate_fat_from_edited_shapes()
    shapes = win._canvas_widgets.canvas.shapes
    assert [shape.label for shape in shapes] == ["EMA", "skin", "fat"]
    assert all(shape.shape_type == "polygon" for shape in shapes)
    assert all(len(shape.points) >= 3 for shape in shapes)
    assert all(shape.flags == {} for shape in shapes)
    assert all(shape.description == "" for shape in shapes)

    expected_payload = json.loads(
        _GOLDEN_IMAGE.with_suffix(".json").read_text(encoding="utf-8")
    )
    expected_by_label = {
        shape["label"]: shape for shape in expected_payload["shapes"]
    }
    minimum_iou = {"EMA": 0.8, "fat": 0.7, "skin": 0.45}
    for shape in shapes:
        assert shape.label is not None
        expected_points = np.asarray(
            expected_by_label[shape.label]["points"], dtype=np.float64
        )
        expected_mask = np.zeros((536, 536), dtype=np.bool_)
        rows, columns = polygon(
            expected_points[:, 1],
            expected_points[:, 0],
            shape=expected_mask.shape,
        )
        expected_mask[rows, columns] = True
        predicted_mask = np.zeros_like(expected_mask)
        rows, columns = polygon(
            shape.points[:, 1], shape.points[:, 0], shape=predicted_mask.shape
        )
        predicted_mask[rows, columns] = True
        intersection = np.logical_and(expected_mask, predicted_mask).sum()
        union = np.logical_or(expected_mask, predicted_mask).sum()
        assert intersection / union > minimum_iou[shape.label]

    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)
