from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from labelme._widgets import AiAssistedAnnotationWidget
from labelme._widgets import UltrasoundReviewWidget


def test_ultrasound_toolbar_defaults_and_callbacks(qtbot: QtBot) -> None:
    current_calls: list[bool] = []
    folder_calls: list[bool] = []
    fat_calls: list[bool] = []
    widget = AiAssistedAnnotationWidget(
        on_run_current=lambda: current_calls.append(True),
        on_run_folder=lambda: folder_calls.append(True),
        on_generate_fat=lambda: fat_calls.append(True),
    )
    qtbot.addWidget(widget)

    assert widget.selected_tasks == ("EMA", "skin")
    widget.set_task_checked("skin", False)
    assert widget.selected_tasks == ("EMA",)
    with pytest.raises(ValueError, match="edited EMA and skin"):
        widget.set_task_checked("fat", True)

    widget._run_current_button.click()
    widget._run_folder_button.click()
    widget._generate_fat_button.click()
    assert current_calls == [True]
    assert folder_calls == [True]
    assert fat_calls == [True]


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
