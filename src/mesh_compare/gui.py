"""Drag-and-drop window: drop the original and the scan, trim them, read the numbers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QGuiApplication, QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mesh_compare.comparison import (
    SUPPORTED_SUFFIXES,
    ComparisonResult,
    compare_meshes,
    format_size,
    load_mesh,
)
from mesh_compare.trim_dialog import TrimDialog
from mesh_compare.trimming import trim_files

T = TypeVar("T")

EXPLANATION = (
    "<b>Scan → Original</b>: how accurate the scanned surface is "
    "(leftover untrimmed material makes this worse).<br>"
    "<b>Original → Scan</b>: how completely the original was captured "
    "(holes or missing areas make this worse).<br>"
    "Signed values: <b>+</b> = scan larger than original, <b>−</b> = scan smaller."
)


class DropZone(QFrame):
    """A box that accepts one dropped mesh file, or opens a file picker when clicked."""

    path_changed = Signal(Path)

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.path: Path | None = None
        self.trimmed_path: Path | None = None
        self.setAcceptDrops(True)
        self.setMinimumSize(260, 140)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter, wordWrap=True)
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        self._set_highlight(False)
        self._refresh_label()

    @property
    def active_path(self) -> Path | None:
        """The file to compare: the trimmed version if there is one, else the original."""
        return self.trimmed_path or self.path

    def set_path(self, path: Path) -> None:
        """Select ``path`` as this zone's file and notify listeners."""
        self.path = path
        self.trimmed_path = None
        self._refresh_label()
        self.path_changed.emit(path)

    def set_trimmed(self, path: Path) -> None:
        """Use the trimmed copy at ``path`` for comparisons."""
        self.trimmed_path = path
        self._refresh_label()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 (Qt API)
        if self._mesh_path_from(event) is not None:
            event.acceptProposedAction()
            self._set_highlight(True)

    def dragLeaveEvent(self, event: object) -> None:  # noqa: N802 (Qt API)
        self._set_highlight(False)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 (Qt API)
        self._set_highlight(False)
        path = self._mesh_path_from(event)
        if path is not None:
            event.acceptProposedAction()
            self.set_path(path)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        patterns = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_SUFFIXES))
        chosen, _ = QFileDialog.getOpenFileName(
            self, f"Choose {self.title}", "", f"3D models ({patterns})"
        )
        if chosen:
            self.set_path(Path(chosen))

    @staticmethod
    def _mesh_path_from(event: QDragEnterEvent | QDropEvent) -> Path | None:
        urls = event.mimeData().urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = Path(urls[0].toLocalFile())
        return path if path.suffix.lower() in SUPPORTED_SUFFIXES else None

    def _refresh_label(self) -> None:
        detail = self.path.name if self.path else "Drop a file here<br>or click to browse"
        if self.trimmed_path:
            detail += f"<br><span style='color:#16a34a'>✂ using {self.trimmed_path.name}</span>"
        self._label.setText(f"<b>{self.title}</b><br><br>{detail}")

    def _set_highlight(self, active: bool) -> None:
        colour = "#3b82f6" if active else "#9ca3af"
        self.setStyleSheet(f"DropZone {{ border: 2px dashed {colour}; border-radius: 10px; }}")


class TaskWorker(QThread):
    """Runs a slow task off the UI thread so the window stays responsive."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, task: Callable[[], object]) -> None:
        super().__init__()
        self.task = task

    def run(self) -> None:
        try:
            result = self.task()
        except Exception as error:  # report any failure in the window instead of crashing
            self.failed.emit(str(error))
        else:
            self.succeeded.emit(result)


class MainWindow(QMainWindow):
    """Top-level window holding the drop zones, options and results table."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mesh Compare")
        self.resize(720, 620)
        self._worker: TaskWorker | None = None
        self._result: ComparisonResult | None = None

        self.original_zone = DropZone("Original model")
        self.scan_zone = DropZone("New scan")
        self.trim_buttons: dict[DropZone, QPushButton] = {}
        for zone in (self.original_zone, self.scan_zone):
            button = QPushButton(f"Trim {zone.title.lower()}…")
            button.setEnabled(False)
            button.clicked.connect(lambda _=False, z=zone: self.open_trim(z))
            self.trim_buttons[zone] = button
            zone.path_changed.connect(lambda _path, z=zone: self._model_added(z))
        self.auto_trim = QCheckBox("Open trimming when a model is added")
        self.auto_trim.setChecked(True)

        self.tolerance = QDoubleSpinBox(
            minimum=0.01, maximum=5.0, singleStep=0.05, value=0.1, decimals=2, suffix=" mm"
        )
        self.align_box = QCheckBox("Auto-align scan to original")
        self.align_box.setChecked(True)
        self.compare_button = QPushButton("Compare")
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self.start_comparison)
        self.copy_button = QPushButton("Copy results")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_results)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Metric", "Scan → Original", "Original → Scan"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.info = QLabel(
            wordWrap=True, textInteractionFlags=Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.status = QLabel("Drop both models to begin.", wordWrap=True)

        zones = QHBoxLayout()
        for zone in (self.original_zone, self.scan_zone):
            column = QVBoxLayout()
            column.addWidget(zone)
            column.addWidget(self.trim_buttons[zone])
            zones.addLayout(column)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Tolerance:"))
        controls.addWidget(self.tolerance)
        controls.addWidget(self.align_box)
        controls.addStretch()
        controls.addWidget(self.copy_button)
        controls.addWidget(self.compare_button)

        layout = QVBoxLayout()
        layout.addLayout(zones)
        layout.addWidget(self.auto_trim)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.info)
        layout.addWidget(QLabel(EXPLANATION, wordWrap=True))
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def start_comparison(self) -> None:
        """Run the comparison for the two selected files in the background."""
        original, scan = self.original_zone.active_path, self.scan_zone.active_path
        if original is None or scan is None:
            return
        tolerance, align = self.tolerance.value(), self.align_box.isChecked()
        self._run_task(
            "Comparing… (aligning and measuring can take a few seconds)",
            lambda: compare_meshes(
                load_mesh(original), load_mesh(scan), tolerance=tolerance, align=align
            ),
            self.show_result,
        )

    def open_trim(self, zone: DropZone) -> None:
        """Open the trimming window for ``zone``'s model, then trim and save in the background."""
        if zone.path is None or self._worker is not None:
            return
        other = self.scan_zone if zone is self.original_zone else self.original_zone
        try:
            mesh = load_mesh(zone.path)
        except Exception as error:  # unreadable file: say so instead of crashing
            self.show_error(str(error))
            return
        dialog = TrimDialog(mesh, zone.path.name, other.path.name if other.path else None)
        if dialog.exec() != TrimDialog.DialogCode.Accepted or (plane := dialog.plane) is None:
            return

        model_path = zone.path
        other_path = other.path if dialog.copy_to_other else None
        targets = [zone] + ([other] if other_path else [])

        def finished(saved: list[Path]) -> None:
            for target, path in zip(targets, saved, strict=True):
                target.set_trimmed(path)
            self.status.setStyleSheet("")
            self.status.setText("Saved " + ", ".join(path.name for path in saved))

        message = "Trimming…" if other_path is None else "Aligning and trimming both models…"
        self._run_task(message, lambda: trim_files(plane, model_path, other_path), finished)

    def _model_added(self, zone: DropZone) -> None:
        self.trim_buttons[zone].setEnabled(True)
        self._update_compare_button()
        if self.auto_trim.isChecked():
            # Let the drop finish before opening a modal window.
            QTimer.singleShot(0, lambda: self.open_trim(zone))

    def _run_task(
        self, message: str, task: Callable[[], T], on_success: Callable[[T], None]
    ) -> None:
        if self._worker is not None:
            return
        self.status.setStyleSheet("")
        self.status.setText(message)
        self._worker = TaskWorker(task)
        self._worker.succeeded.connect(on_success)
        self._worker.failed.connect(self.show_error)
        self._worker.finished.connect(self._worker_finished)
        self._set_busy(True)
        self._worker.start()

    def show_result(self, result: ComparisonResult) -> None:
        """Fill the table and info area from a finished comparison."""
        self._result = result
        rows = result.metric_rows()
        self.table.setRowCount(len(rows))
        for row, cells in enumerate(rows):
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)

        info = [
            f"Original size: {format_size(result.original_size)}",
            f"Scan size: {format_size(result.scan_size)}",
            f"Points sampled per surface: {result.sample_count:,}",
        ]
        if result.alignment_rmse is not None:
            info.append(f"Alignment fit (best 90% of points): {result.alignment_rmse:.4f} mm")
        self.info.setText("<br>".join(info))

        if result.warnings:
            self.status.setStyleSheet("color: #dc2626;")
            self.status.setText("⚠ " + " ".join(result.warnings))
        else:
            self.status.setStyleSheet("")
            self.status.setText("Done.")
        self.copy_button.setEnabled(True)

    def show_error(self, message: str) -> None:
        """Display a failure message."""
        self.status.setStyleSheet("color: #dc2626;")
        self.status.setText(f"Error: {message}")

    def copy_results(self) -> None:
        """Copy results as tab-separated text (pastes cleanly into Excel/Sheets)."""
        if self._result is not None:
            QGuiApplication.clipboard().setText(self._result.as_text())
            self.status.setStyleSheet("")
            self.status.setText("Results copied to clipboard.")

    def _worker_finished(self) -> None:
        self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        for zone, button in self.trim_buttons.items():
            button.setEnabled(not busy and zone.path is not None)
        self._update_compare_button()

    def _update_compare_button(self) -> None:
        ready = self.original_zone.path is not None and self.scan_zone.path is not None
        self.compare_button.setEnabled(ready and self._worker is None)
