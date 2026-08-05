from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore
from PySide6 import QtWidgets

from ._info_button import InfoButton


class UltrasoundReviewWidget(QtWidgets.QWidget):
    """Independent image-level status controls for ultrasound annotations."""

    def __init__(
        self,
        on_status_changed: Callable[[bool, bool], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._on_status_changed = on_status_changed
        self._init_ui()

    @property
    def review_required(self) -> bool:
        return self._needs_expert_review_checkbox.isChecked()

    @property
    def model_failure(self) -> bool:
        return self._model_failure_checkbox.isChecked()

    def set_status(self, *, review_required: bool, model_failure: bool) -> None:
        review_blocker = QtCore.QSignalBlocker(self._needs_expert_review_checkbox)
        failure_blocker = QtCore.QSignalBlocker(self._model_failure_checkbox)
        self._needs_expert_review_checkbox.setChecked(review_required)
        self._model_failure_checkbox.setChecked(model_failure)
        del failure_blocker
        del review_blocker

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addStretch()
        header_layout.addWidget(QtWidgets.QLabel(self.tr("Review")))
        header_layout.addWidget(
            InfoButton(
                tooltip=self.tr(
                    "Optional image-level status flags. Both are off by "
                    "default and can be selected independently."
                )
            )
        )
        header_layout.addStretch()
        layout.addLayout(header_layout)

        self._needs_expert_review_checkbox = QtWidgets.QCheckBox(
            self.tr("Needs expert review")
        )
        self._needs_expert_review_checkbox.setChecked(False)
        self._needs_expert_review_checkbox.setToolTip(
            self.tr("Mark an uncertain annotation for later expert review")
        )
        self._needs_expert_review_checkbox.toggled.connect(self._emit_status_changed)
        layout.addWidget(self._needs_expert_review_checkbox)

        self._model_failure_checkbox = QtWidgets.QCheckBox(self.tr("Model failure"))
        self._model_failure_checkbox.setChecked(False)
        self._model_failure_checkbox.setToolTip(
            self.tr("Mark the model result as unusable or clearly failed")
        )
        self._model_failure_checkbox.toggled.connect(self._emit_status_changed)
        layout.addWidget(self._model_failure_checkbox)

        layout.addStretch()
        self.setMaximumWidth(220)

    def _emit_status_changed(self, _checked: bool) -> None:
        self._on_status_changed(self.review_required, self.model_failure)
