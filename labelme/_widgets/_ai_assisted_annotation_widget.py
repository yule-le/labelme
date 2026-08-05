from __future__ import annotations

from collections.abc import Callable
from typing import cast

from PySide6 import QtCore
from PySide6 import QtWidgets

from .._automation import AiOutputFormat
from .._ultrasound import UltrasoundTask
from ._info_button import InfoButton


class AiAssistedAnnotationWidget(QtWidgets.QWidget):
    """Compact transverse-ultrasound controls in the existing AI toolbar slot."""

    hover_highlight_requested = QtCore.Signal(bool)

    def __init__(
        self,
        on_run_current: Callable[[], None],
        on_run_folder: Callable[[], None],
        on_generate_fat: Callable[[], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._on_run_current = on_run_current
        self._on_run_folder = on_run_folder
        self._on_generate_fat = on_generate_fat
        self._task_checkboxes: dict[UltrasoundTask, QtWidgets.QCheckBox] = {}
        self._init_ui()

    @property
    def output_format(self) -> AiOutputFormat:
        """Keep AI text/point tools producing polygons after this UI is repurposed."""

        return cast(AiOutputFormat, "polygon")

    @property
    def selected_tasks(self) -> tuple[UltrasoundTask, ...]:
        return tuple(
            task for task in ("EMA", "skin") if self._task_checkboxes[task].isChecked()
        )

    def set_task_checked(self, task: UltrasoundTask, checked: bool) -> None:
        if task == "fat":
            raise ValueError("Fat is generated from edited EMA and skin polygons.")
        self._task_checkboxes[task].setChecked(checked)

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addStretch()
        header_layout.addWidget(QtWidgets.QLabel(self.tr("AI-Assisted Annotation")))
        header_layout.addWidget(
            InfoButton(
                tooltip=self.tr(
                    "Run transverse ultrasound EMA and skin models. "
                    "Fat is derived from their anatomical boundaries."
                )
            )
        )
        header_layout.addStretch()
        layout.addLayout(header_layout)

        model_combo = QtWidgets.QComboBox()
        model_combo.addItem(self.tr("Transverse"), "transverse")
        model_combo.setEnabled(False)
        layout.addWidget(model_combo)

        tasks_layout = QtWidgets.QHBoxLayout()
        tasks_layout.setContentsMargins(0, 0, 0, 0)
        tasks_layout.setSpacing(6)
        for task, display, checked in (
            ("EMA", "EMA", True),
            ("skin", "skin", True),
        ):
            checkbox = QtWidgets.QCheckBox(display)
            checkbox.setChecked(checked)
            self._task_checkboxes[task] = checkbox
            tasks_layout.addWidget(checkbox)
        layout.addLayout(tasks_layout)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._run_current_button = QtWidgets.QPushButton(self.tr("Run Current"))
        self._run_current_button.clicked.connect(self._on_run_current)
        buttons_layout.addWidget(self._run_current_button)
        self._run_folder_button = QtWidgets.QPushButton(self.tr("Run Folder"))
        self._run_folder_button.clicked.connect(self._on_run_folder)
        buttons_layout.addWidget(self._run_folder_button)
        layout.addLayout(buttons_layout)

        self._generate_fat_button = QtWidgets.QPushButton(self.tr("Generate Fat"))
        self._generate_fat_button.setToolTip(
            self.tr("Use the edited skin lower boundary and EMA upper boundary")
        )
        self._generate_fat_button.clicked.connect(self._on_generate_fat)
        layout.addWidget(self._generate_fat_button)

        self.setMaximumWidth(230)

    def set_busy(self, *, current: bool = False, folder: bool = False) -> None:
        busy = current or folder
        for checkbox in self._task_checkboxes.values():
            checkbox.setEnabled(not busy)
        self._generate_fat_button.setEnabled(not busy and self.isEnabled())
        self._run_current_button.setEnabled(not busy and self.isEnabled())
        self._run_folder_button.setEnabled(self.isEnabled())
        self._run_folder_button.setText(
            self.tr("Cancel") if folder else self.tr("Run Folder")
        )

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        if not enabled:
            self.hover_highlight_requested.emit(False)

    def set_disabled_models(self, disabled_models: tuple[str, ...]) -> None:
        """Compatibility no-op for the retained AI-Points/AI-Box canvas modes."""

        del disabled_models
