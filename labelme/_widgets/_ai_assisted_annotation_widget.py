from __future__ import annotations

from collections.abc import Callable
from typing import cast

from PySide6 import QtCore
from PySide6 import QtWidgets

from .._automation import AiOutputFormat
from .._ultrasound import MeasurementCode
from .._ultrasound import UltrasoundTask
from ._info_button import InfoButton


class AiAssistedAnnotationWidget(QtWidgets.QWidget):
    """Compact transverse-ultrasound controls in the existing AI toolbar slot."""

    hover_highlight_requested = QtCore.Signal(bool)

    def __init__(
        self,
        on_run_segmentation_current: Callable[[], None],
        on_run_segmentation_folder: Callable[[], None],
        on_run_measurement_current: Callable[[], None],
        on_run_measurement_folder: Callable[[], None],
        on_generate_fat: Callable[[], None],
        on_import_depth_csv: Callable[[], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._on_run_segmentation_current = on_run_segmentation_current
        self._on_run_segmentation_folder = on_run_segmentation_folder
        self._on_run_measurement_current = on_run_measurement_current
        self._on_run_measurement_folder = on_run_measurement_folder
        self._on_generate_fat = on_generate_fat
        self._on_import_depth_csv = on_import_depth_csv
        self._task_checkboxes: dict[UltrasoundTask, QtWidgets.QCheckBox] = {}
        self._init_ui()

    @property
    def output_format(self) -> AiOutputFormat:
        """Keep AI text/point tools producing polygons after this UI is repurposed."""

        return cast(AiOutputFormat, "polygon")

    @property
    def selected_tasks(self) -> tuple[UltrasoundTask, ...]:
        return self.selected_segmentation_tasks + self.selected_measurement_tasks

    @property
    def selected_segmentation_tasks(self) -> tuple[UltrasoundTask, ...]:
        return tuple(
            task for task in ("EMA", "skin") if self._task_checkboxes[task].isChecked()
        )

    @property
    def selected_measurement_tasks(self) -> tuple[UltrasoundTask, ...]:
        return tuple(
            task
            for task in ("EMD", "EMW", "FD", "SD")
            if self._task_checkboxes[task].isChecked()
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

        sections_layout = QtWidgets.QHBoxLayout()
        sections_layout.setContentsMargins(0, 0, 0, 0)
        sections_layout.setSpacing(6)

        segmentation_group = QtWidgets.QGroupBox(self.tr("Segmentation"))
        segmentation_layout = QtWidgets.QVBoxLayout(segmentation_group)
        segmentation_layout.setContentsMargins(6, 6, 6, 6)
        segmentation_layout.setSpacing(4)

        segmentation_controls_layout = QtWidgets.QHBoxLayout()
        segmentation_controls_layout.setContentsMargins(0, 0, 0, 0)
        segmentation_controls_layout.setSpacing(8)

        model_combo = QtWidgets.QComboBox()
        model_combo.addItem(self.tr("Transverse"), "transverse")
        model_combo.setEnabled(False)
        model_combo.setMinimumWidth(110)
        model_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        segmentation_controls_layout.addWidget(model_combo, 1)
        for task, display, checked in (("EMA", "EMA", True), ("skin", "skin", True)):
            checkbox = QtWidgets.QCheckBox(display)
            checkbox.setChecked(checked)
            self._task_checkboxes[task] = checkbox
            segmentation_controls_layout.addWidget(checkbox)
        segmentation_layout.addLayout(segmentation_controls_layout)

        segmentation_buttons_layout = QtWidgets.QHBoxLayout()
        segmentation_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._run_segmentation_current_button = QtWidgets.QPushButton(
            self.tr("Run Current")
        )
        self._allow_horizontal_shrink(self._run_segmentation_current_button)
        self._run_segmentation_current_button.clicked.connect(
            self._on_run_segmentation_current
        )
        segmentation_buttons_layout.addWidget(self._run_segmentation_current_button)
        self._run_segmentation_folder_button = QtWidgets.QPushButton(
            self.tr("Run Folder")
        )
        self._allow_horizontal_shrink(self._run_segmentation_folder_button)
        self._run_segmentation_folder_button.clicked.connect(
            self._on_run_segmentation_folder
        )
        segmentation_buttons_layout.addWidget(self._run_segmentation_folder_button)
        segmentation_layout.addLayout(segmentation_buttons_layout)

        self._generate_fat_button = QtWidgets.QPushButton(self.tr("Generate Fat"))
        self._allow_horizontal_shrink(self._generate_fat_button)
        self._generate_fat_button.setToolTip(
            self.tr("Use the edited skin lower boundary and EMA upper boundary")
        )
        self._generate_fat_button.clicked.connect(self._on_generate_fat)
        segmentation_layout.addWidget(self._generate_fat_button)
        sections_layout.addWidget(segmentation_group, 1)

        measurement_group = QtWidgets.QGroupBox(self.tr("Measurement"))
        measurement_layout = QtWidgets.QVBoxLayout(measurement_group)
        measurement_layout.setContentsMargins(6, 6, 6, 6)
        measurement_layout.setSpacing(4)

        measurement_tasks_layout = QtWidgets.QHBoxLayout()
        measurement_tasks_layout.setContentsMargins(0, 0, 0, 0)
        measurement_tasks_layout.setSpacing(8)
        measurement_tasks: tuple[MeasurementCode, ...] = ("EMD", "EMW", "FD", "SD")
        for task in measurement_tasks:
            checkbox = QtWidgets.QCheckBox(task)
            checkbox.setToolTip(
                self.tr(
                    {
                        "EMW": "Eye Muscle Width",
                        "EMD": "Eye Muscle Depth",
                        "FD": "Fat Depth",
                        "SD": "Skin Depth",
                    }[task]
                )
            )
            self._task_checkboxes[task] = checkbox
            measurement_tasks_layout.addWidget(checkbox)
        measurement_layout.addLayout(measurement_tasks_layout)

        self._import_depth_button = QtWidgets.QPushButton(self.tr("Import Depth CSV"))
        self._allow_horizontal_shrink(self._import_depth_button)
        self._import_depth_button.clicked.connect(self._on_import_depth_csv)
        measurement_layout.addWidget(self._import_depth_button)
        self._depth_status = QtWidgets.QLabel(self.tr("Depth: CSV not imported"))
        self._depth_status.setWordWrap(True)
        measurement_layout.addWidget(self._depth_status)

        measurement_buttons_layout = QtWidgets.QHBoxLayout()
        measurement_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._run_measurement_current_button = QtWidgets.QPushButton(
            self.tr("Run Current")
        )
        self._allow_horizontal_shrink(self._run_measurement_current_button)
        self._run_measurement_current_button.clicked.connect(
            self._on_run_measurement_current
        )
        measurement_buttons_layout.addWidget(self._run_measurement_current_button)
        self._run_measurement_folder_button = QtWidgets.QPushButton(
            self.tr("Run Folder")
        )
        self._allow_horizontal_shrink(self._run_measurement_folder_button)
        self._run_measurement_folder_button.clicked.connect(
            self._on_run_measurement_folder
        )
        measurement_buttons_layout.addWidget(self._run_measurement_folder_button)
        measurement_layout.addLayout(measurement_buttons_layout)
        sections_layout.addWidget(measurement_group, 1)

        layout.addLayout(sections_layout)

        self.setMaximumWidth(500)

    @staticmethod
    def _allow_horizontal_shrink(widget: QtWidgets.QWidget) -> None:
        widget.setMinimumWidth(0)
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def set_depth_status(self, depth_mm: float | None) -> None:
        self._depth_status.setText(
            self.tr("Depth: Missing")
            if depth_mm is None
            else self.tr("Depth: %.2f mm (CSV)") % depth_mm
        )

    def set_manifest_summary(self, *, matched: int, total: int) -> None:
        self._depth_status.setToolTip(
            self.tr("Depth CSV matched %d of %d folder images.") % (matched, total)
        )

    def set_busy(self, *, current: bool = False, folder: bool = False) -> None:
        busy = current or folder
        for checkbox in self._task_checkboxes.values():
            checkbox.setEnabled(not busy)
        self._generate_fat_button.setEnabled(not busy and self.isEnabled())
        self._import_depth_button.setEnabled(not busy and self.isEnabled())
        for button in (
            self._run_segmentation_current_button,
            self._run_measurement_current_button,
        ):
            button.setEnabled(not busy and self.isEnabled())
        for button in (
            self._run_segmentation_folder_button,
            self._run_measurement_folder_button,
        ):
            button.setEnabled(self.isEnabled())
            button.setText(self.tr("Cancel") if folder else self.tr("Run Folder"))

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        if not enabled:
            self.hover_highlight_requested.emit(False)

    def set_disabled_models(self, disabled_models: tuple[str, ...]) -> None:
        """Compatibility no-op for the retained AI-Points/AI-Box canvas modes."""

        del disabled_models
