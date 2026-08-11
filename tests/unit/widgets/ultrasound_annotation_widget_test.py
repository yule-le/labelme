from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from labelme._widgets import AiAssistedAnnotationWidget
from labelme._widgets import UltrasoundReviewWidget


def test_ultrasound_toolbar_defaults_and_callbacks(qtbot: QtBot) -> None:
    current_calls: list[bool] = []
    folder_calls: list[bool] = []
    fat_calls: list[bool] = []
    csv_calls: list[bool] = []
    widget = AiAssistedAnnotationWidget(
        on_run_segmentation_current=lambda: current_calls.append(True),
        on_run_segmentation_folder=lambda: folder_calls.append(True),
        on_run_measurement_current=lambda: current_calls.append(False),
        on_run_measurement_folder=lambda: folder_calls.append(False),
        on_generate_fat=lambda: fat_calls.append(True),
        on_import_depth_csv=lambda: csv_calls.append(True),
    )
    qtbot.addWidget(widget)

    assert widget.selected_tasks == ("EMA", "skin")
    widget.set_task_checked("skin", False)
    assert widget.selected_tasks == ("EMA",)
    with pytest.raises(ValueError, match="edited EMA and skin"):
        widget.set_task_checked("fat", True)

    widget._run_segmentation_current_button.click()
    widget._run_segmentation_folder_button.click()
    widget._run_measurement_current_button.click()
    widget._run_measurement_folder_button.click()
    widget._generate_fat_button.click()
    widget._import_depth_button.click()
    assert current_calls == [True, False]
    assert folder_calls == [True, False]
    assert fat_calls == [True]
    assert csv_calls == [True]

    widget.set_task_checked("EMD", True)
    widget.set_task_checked("EMW", True)
    widget.set_task_checked("FD", True)
    assert widget.selected_tasks == ("EMA", "EMD", "EMW", "FD")
    assert widget.selected_segmentation_tasks == ("EMA",)
    assert widget.selected_measurement_tasks == ("EMD", "EMW", "FD")

    widget.set_depth_status(100.0)
    assert widget._depth_status.text() == "Depth: 100.00 mm (CSV)"


def test_review_status_defaults_off_and_checkboxes_are_independent(
    qtbot: QtBot,
) -> None:
    values: list[tuple[bool, bool]] = []
    widget = UltrasoundReviewWidget(
        on_status_changed=lambda review, failure: values.append((review, failure))
    )
    qtbot.addWidget(widget)

    assert widget.review_required is False
    assert widget.model_failure is False

    widget._model_failure_checkbox.click()
    assert widget.review_required is False
    assert widget.model_failure is True
    assert values == [(False, True)]

    widget._needs_expert_review_checkbox.click()
    assert widget.review_required is True
    assert widget.model_failure is True
    assert values[-1] == (True, True)

    widget.set_status(review_required=False, model_failure=False)
    assert widget.review_required is False
    assert widget.model_failure is False
    assert values == [(False, True), (True, True)]
