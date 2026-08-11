from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6 import QtCore
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

import labelme._app
from labelme._app import MainWindow
from labelme._shape import Shape
from labelme._ultrasound import AnnotationPrediction
from labelme._ultrasound import BatchInferenceRequest
from labelme._ultrasound import Calibration
from labelme._ultrasound import InferenceResult
from labelme._ultrasound import MeasurementResult

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata


def _ema_result(win: MainWindow, request_id: str) -> InferenceResult:
    assert win._image_path is not None
    ema = AnnotationPrediction(
        label="EMA",
        points=((40.0, 30.0), (180.0, 28.0), (210.0, 90.0), (90.0, 110.0)),
        confidence=0.92,
        source_target="eye_muscle",
        strategy="single_model",
        model_id="seg_eye_muscle",
        artifact_id="seg_eye_muscle__transverse__v1",
    )
    fat = AnnotationPrediction(
        label="fat",
        points=((35.0, 12.0), (215.0, 12.0), (180.0, 28.0), (40.0, 30.0)),
        confidence=0.88,
        source_target="derived_fat",
        strategy="boundary_derived",
    )
    skin = AnnotationPrediction(
        label="skin",
        points=((30.0, 5.0), (220.0, 5.0), (215.0, 12.0), (35.0, 12.0)),
        confidence=0.94,
        source_target="skin",
        strategy="single_model",
        model_id="seg_skin",
        artifact_id="seg_skin__transverse__v1",
    )
    return InferenceResult(
        request_id=request_id,
        image_path=os.path.abspath(win._image_path),
        predictions=(ema, fat, skin),
    )


@pytest.mark.gui
def test_ema_result_is_one_undoable_operation_and_round_trips(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "ultrasound-test.png"
    Image.new("RGB", (320, 240), color=(80, 80, 80)).save(image_path)
    win = main_win(
        file_or_dir=str(image_path),
        config_overrides={"auto_save": False},
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    request_id = "ema-test-request"
    win._ultrasound_request_id = request_id

    win._on_ultrasound_inference_completed(_ema_result(win, request_id))

    canvas = win._canvas_widgets.canvas
    assert [shape.label for shape in canvas.shapes] == ["EMA", "fat", "skin"]
    assert canvas.can_restore_shape
    assert win._is_changed
    assert all(shape.flags == {} for shape in canvas.shapes)
    assert all(shape.description == "" for shape in canvas.shapes)

    label_path = tmp_path / "ultrasound-test.json"
    assert win.save_labels(label_path=str(label_path))
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert [shape["label"] for shape in payload["shapes"]] == [
        "EMA",
        "fat",
        "skin",
    ]
    assert all(shape["shape_type"] == "polygon" for shape in payload["shapes"])
    assert all(shape["flags"] == {} for shape in payload["shapes"])
    assert all(shape["description"] == "" for shape in payload["shapes"])

    win.undo_shape_edit()
    assert canvas.shapes == []
    win.mark_clean()
    win.close()

    reopened = main_win(file_or_dir=str(label_path), output_dir=str(tmp_path))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=reopened)
    assert [shape.label for shape in reopened._canvas_widgets.canvas.shapes] == [
        "EMA",
        "fat",
        "skin",
    ]
    assert all(shape.flags == {} for shape in reopened._canvas_widgets.canvas.shapes)
    assert all(
        shape.description == "" for shape in reopened._canvas_widgets.canvas.shapes
    )

    close_or_pause(qtbot=qtbot, widget=reopened, pause=pause)


@pytest.mark.gui
def test_measurement_lines_and_metadata_round_trip_and_follow_manual_edit(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "measurement.png"
    Image.new("RGB", (320, 240), color=(80, 80, 80)).save(image_path)
    win = main_win(
        file_or_dir=str(image_path),
        config_overrides={"auto_save": False},
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    calibration = Calibration(
        depth_setting_mm=100.0,
        image_height_px=240,
        pixel_size_x_mm=100.0 / 240.0,
        pixel_size_y_mm=100.0 / 240.0,
    )
    measurements = tuple(
        MeasurementResult(
            code=code,
            value_mm=10.0,
            segment=segment,
            valid=True,
            method_version=method,
        )
        for code, segment, method in (
            ("EMW", ((10.0, 20.0), (100.0, 20.0)), "geom-depth-width-v1.1"),
            ("EMD", ((50.0, 100.0), (50.0, 40.0)), "geom-depth-width-v1.1"),
            ("FD", ((50.0, 35.0), (50.0, 25.0)), "eye-axis-layer-depth-v1"),
            ("SD", ((50.0, 20.0), (50.0, 15.0)), "eye-axis-layer-depth-v1"),
        )
    )
    request_id = "measurement-result"
    win._ultrasound_request_id = request_id
    win._on_ultrasound_inference_completed(
        InferenceResult(
            request_id=request_id,
            image_path=os.path.abspath(image_path),
            predictions=(),
            measurements=measurements,
            calibration=calibration,
            depth_setting_mm=100.0,
            original_image_path="backend/data/raw/batch/measurement.png",
        )
    )

    canvas = win._canvas_widgets.canvas
    assert [shape.label for shape in canvas.shapes] == ["EMW", "EMD", "FD", "SD"]
    assert all(shape.shape_type == "line" for shape in canvas.shapes)
    emw = canvas.shapes[0]
    emw.move_vertex(1, np.array([130.0, 20.0]))

    label_path = image_path.with_suffix(".json")
    assert win.save_labels(label_path=str(label_path))
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    metadata = payload["ultrasoundMetadata"]
    assert metadata["depthSettingMm"] == 100.0
    assert metadata["originalImagePath"].endswith("measurement.png")
    assert metadata["calibration"]["roiHeightPx"] == 240
    assert metadata["measurements"]["EMW"]["valueMm"] == pytest.approx(50.0)
    assert payload["shapes"][0]["ultrasoundMeasurement"]["valueMm"] == pytest.approx(
        50.0
    )

    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_import_depth_csv_matches_cropped_image_filename(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pause: bool,
) -> None:
    image_path = tmp_path / "2026Jul29-10.50.59.jpg"
    Image.new("RGB", (320, 240), color=(80, 80, 80)).save(image_path)
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(
        "sample_id,filename,depth_ocr_text,depth_ocr_mm,ocr_status\n"
        "s_123,2026Jul29-10.50.59.jpg,10,100,ok\n",
        encoding="utf-8",
    )
    win = main_win(file_or_dir=str(image_path), output_dir=str(tmp_path))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(csv_path), "CSV files (*.csv)"),
    )

    win._import_depth_csv()

    entry = win._depth_entry(str(image_path))
    assert entry is not None
    assert entry.depth_setting_mm == 100.0
    assert win._ai_annotation._depth_status.text() == "Depth: 100.00 mm (CSV)"
    label_path = image_path.with_suffix(".json")
    assert win.save_labels(label_path=str(label_path))
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert payload["ultrasoundMetadata"]["depthSettingMm"] == 100.0
    assert (
        payload["ultrasoundMetadata"]["sourceFilename"]
        == "2026Jul29-10.50.59.jpg"
    )
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_result_for_previous_image_is_ignored(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "current.png"
    Image.new("RGB", (320, 240), color=(80, 80, 80)).save(image_path)
    win = main_win(file_or_dir=str(image_path), output_dir=str(tmp_path))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    request_id = "stale-request"
    win._ultrasound_request_id = request_id
    result = _ema_result(win, request_id)
    stale = InferenceResult(
        request_id=result.request_id,
        image_path=str(tmp_path / "previous.png"),
        predictions=result.predictions,
    )

    win._on_ultrasound_inference_completed(stale)

    assert win._canvas_widgets.canvas.shapes == []
    assert win._ultrasound_request_id is None
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_rerun_replaces_generated_set_as_one_undoable_operation(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "replace.png"
    Image.new("RGB", (320, 240), color=(80, 80, 80)).save(image_path)
    win = main_win(
        file_or_dir=str(image_path),
        config_overrides={"auto_save": False},
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    first_id = "first-request"
    win._ultrasound_request_id = first_id
    win._on_ultrasound_inference_completed(_ema_result(win, first_id))
    first_points = [shape.points.copy() for shape in win._canvas_widgets.canvas.shapes]

    second_id = "second-request"
    win._ultrasound_request_id = second_id
    second_result = _ema_result(win, second_id)
    shifted_predictions = tuple(
        AnnotationPrediction(
            label=prediction.label,
            points=tuple((x + 3.0, y + 2.0) for x, y in prediction.points),
            confidence=prediction.confidence,
            source_target=prediction.source_target,
            strategy=prediction.strategy,
            model_id=prediction.model_id,
            artifact_id=prediction.artifact_id,
        )
        for prediction in second_result.predictions
    )
    win._on_ultrasound_inference_completed(
        InferenceResult(
            request_id=second_id,
            image_path=second_result.image_path,
            predictions=shifted_predictions,
        )
    )

    canvas = win._canvas_widgets.canvas
    assert [shape.label for shape in canvas.shapes] == ["EMA", "fat", "skin"]
    assert all(
        not np.array_equal(shape.points, original)
        for shape, original in zip(canvas.shapes, first_points, strict=True)
    )

    win.undo_shape_edit()
    assert [shape.label for shape in canvas.shapes] == ["EMA", "fat", "skin"]
    assert all(
        np.array_equal(shape.points, original)
        for shape, original in zip(canvas.shapes, first_points, strict=True)
    )
    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_folder_run_submits_all_images_and_worker_skips_existing_json(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pause: bool,
) -> None:
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    Image.new("RGB", (320, 240), color=(80, 80, 80)).save(first_image)
    Image.new("RGB", (320, 240), color=(80, 80, 80)).save(second_image)
    second_image.with_suffix(".json").write_text(
        "existing annotation",
        encoding="utf-8",
    )
    win = main_win(
        file_or_dir=str(tmp_path),
        config_overrides={"auto_save": False},
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(win, "_ensure_ultrasound_worker", lambda: None)
    submitted: list[BatchInferenceRequest] = []
    win.ultrasound_batch_requested.connect(submitted.append)

    win._start_ultrasound_batch()

    assert len(submitted) == 1
    request = submitted[0]
    assert request.tasks == ("EMA", "skin")
    assert [Path(item.image_path).name for item in request.items] == [
        "first.png",
        "second.png",
    ]
    assert request.items[1].label_path == str(second_image.with_suffix(".json"))

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_generate_fat_uses_current_edited_ema_and_skin_as_one_operation(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "derive-fat.png"
    Image.new("RGB", (320, 240), color=(80, 80, 80)).save(image_path)
    win = main_win(
        file_or_dir=str(image_path),
        config_overrides={"auto_save": False},
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    request_id = "ema-skin-only"
    result = _ema_result(win, request_id)
    win._ultrasound_request_id = request_id
    win._on_ultrasound_inference_completed(
        InferenceResult(
            request_id=request_id,
            image_path=result.image_path,
            predictions=(result.predictions[0], result.predictions[2]),
        )
    )
    canvas = win._canvas_widgets.canvas
    assert [shape.label for shape in canvas.shapes] == ["EMA", "skin"]
    boundary_points = {shape.label: shape.points.copy() for shape in canvas.shapes}

    win._generate_fat_from_edited_shapes()

    assert [shape.label for shape in canvas.shapes] == ["EMA", "skin", "fat"]
    assert np.array_equal(canvas.shapes[0].points, boundary_points["EMA"])
    assert np.array_equal(canvas.shapes[1].points, boundary_points["skin"])
    win.undo_shape_edit()
    assert [shape.label for shape in canvas.shapes] == ["EMA", "skin"]
    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_review_statuses_default_off_and_round_trip_as_image_metadata(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "review.png"
    Image.new("RGB", (40, 30), color=(80, 80, 80)).save(image_path)
    win = main_win(
        file_or_dir=str(image_path),
        config_overrides={"auto_save": False},
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    toolbar_widgets = [
        action.defaultWidget()
        for toolbar in win.findChildren(QtWidgets.QToolBar)
        for action in toolbar.actions()
        if isinstance(action, QtWidgets.QWidgetAction)
    ]
    assert win._ultrasound_review in toolbar_widgets
    assert win._ai_text not in toolbar_widgets
    assert win._ai_text.isHidden()
    win._switch_canvas_mode(edit=False, create_mode="polygon")
    assert win._ai_text.isHidden()
    win._switch_canvas_mode(edit=True)
    assert win.menuBar().font().pointSizeF() >= 11.0
    assert all(menu.font().pointSizeF() >= 11.0 for menu in win._menus)
    assert win._ultrasound_review.review_required is False
    assert win._ultrasound_review.model_failure is False

    win._ultrasound_review._model_failure_checkbox.click()
    assert win._ultrasound_review.review_required is False
    assert win._ultrasound_review.model_failure is True
    label_path = image_path.with_suffix(".json")
    assert win.save_labels(label_path=str(label_path))
    win.mark_clean()
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert payload["ultrasoundReview"] == {
        "required": False,
        "modelFailure": True,
        "reasons": [],
        "note": "",
    }
    win.close()

    reopened = main_win(file_or_dir=str(label_path), output_dir=str(tmp_path))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=reopened)
    assert reopened._ultrasound_review.review_required is False
    assert reopened._ultrasound_review.model_failure is True
    reopened._ultrasound_review._needs_expert_review_checkbox.click()
    assert reopened.save_labels(label_path=str(label_path))
    reopened.mark_clean()
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert payload["ultrasoundReview"] == {
        "required": True,
        "modelFailure": True,
        "reasons": [],
        "note": "",
    }

    reopened._ultrasound_review._needs_expert_review_checkbox.click()
    reopened._ultrasound_review._model_failure_checkbox.click()
    assert reopened.save_labels(label_path=str(label_path))
    reopened.mark_clean()
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert "ultrasoundReview" not in payload

    close_or_pause(qtbot=qtbot, widget=reopened, pause=pause)


@pytest.mark.gui
def test_delete_shape_action_prefers_marquee_selected_vertices(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pause: bool,
) -> None:
    image_path = tmp_path / "marquee.png"
    Image.new("RGB", (120, 100), color=(80, 80, 80)).save(image_path)
    win = main_win(
        file_or_dir=str(image_path),
        config_overrides={"auto_save": False},
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    shape = Shape(
        label="EMA",
        shape_type="polygon",
        points=np.array([(10, 10), (50, 10), (90, 10), (90, 80), (50, 80), (10, 80)]),
        closed=True,
    )
    win._load_shapes([shape])
    canvas = win._canvas_widgets.canvas
    canvas.select_shapes([shape])
    canvas._begin_vertex_marquee(pos=QtCore.QPointF(40, 0))
    canvas._update_vertex_marquee(pos=QtCore.QPointF(100, 30))
    canvas._finish_vertex_marquee()
    monkeypatch.setattr(
        win,
        "_confirm_deletion",
        lambda **_kwargs: pytest.fail("Shape deletion confirmation was shown"),
    )

    win.delete_selected_shapes()

    assert len(canvas.shapes) == 1
    assert len(canvas.shapes[0].points) == 4
    assert canvas.shapes[0].points[:2].tolist() == [
        [10.0, 10.0],
        [90.0, 80.0],
    ]
    win.undo_shape_edit()
    assert len(canvas.shapes[0].points) == 6
    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_open_folder_starts_at_configured_ultrasound_root(
    main_win: MainWinFactory,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pause: bool,
) -> None:
    image_path = tmp_path / "current.png"
    Image.new("RGB", (40, 30), color=(80, 80, 80)).save(image_path)
    annotation_root = tmp_path / "processed" / "transverse"
    annotation_root.mkdir(parents=True)
    win = main_win(file_or_dir=str(image_path))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    monkeypatch.setattr(
        labelme._app,
        "load_annotation_root",
        lambda: annotation_root,
    )
    opened_at: list[str] = []

    def _capture_directory(
        parent: QtWidgets.QWidget,
        caption: str,
        directory: str,
        options: QtWidgets.QFileDialog.Option,
    ) -> str:
        del parent, caption, options
        opened_at.append(directory)
        return ""

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        _capture_directory,
    )

    win._open_dir_with_dialog()

    assert opened_at == [str(annotation_root)]
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)
