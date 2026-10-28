"""Match-scale window: both models side by side at the same mm-per-pixel, scale the second."""

from __future__ import annotations

import math

import numpy as np
import trimesh
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from mesh_compare.comparison import estimate_scale
from mesh_compare.scaling import grid_step
from mesh_compare.trimming import (
    UP_AXES,
    VIEW_NAMES,
    Camera,
    RenderedView,
    SurfaceSamples,
    render_view,
    side_cameras,
)
from mesh_compare.widgets import ScrollArea, fit_to_screen
from mesh_compare.workers import TaskWorker

MIN_SCALE = 1e-4
MAX_SCALE = 1e4
# Slider positions per factor of ten (the slider is logarithmic).
SLIDER_STEPS_PER_DECADE = 1000
BACKGROUND = QColor(245, 245, 245)
GRID_COLOUR = QColor(200, 205, 215)
# Size difference beyond which the window points it out.
SIZE_HINT_RATIO = 5.0
MODEL_TINTS = (np.array([1.0, 1.0, 1.0]), np.array([0.85, 0.92, 1.0]))


class ModelCanvas(QWidget):
    """Draws a rendered model at a given mm-per-pixel over a millimetre grid."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.setFixedSize(size, size)
        self.image = QImage()
        self.image_span_mm = 1.0
        self.mm_per_pixel = 1.0
        self.grid_mm = 1.0

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        painter.setPen(QPen(GRID_COLOUR, 1))
        centre = self.width() / 2
        spacing = self.grid_mm / self.mm_per_pixel
        lines = int(centre / spacing) + 1
        for i in range(-lines, lines + 1):
            offset = centre + i * spacing
            painter.drawLine(int(offset), 0, int(offset), self.height())
            painter.drawLine(0, int(offset), self.width(), int(offset))

        side = self.image_span_mm / self.mm_per_pixel
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRectF(centre - side / 2, centre - side / 2, side, side), self.image)
        painter.setPen(QColor(90, 90, 90))
        painter.drawText(8, self.height() - 8, f"grid {self.grid_mm:g} mm")
        painter.end()


class ScaleDialog(QDialog):
    """Shows the original and the scan side by side and lets the user scale the scan.

    After ``exec()`` returns ``Accepted``, ``scale`` holds the chosen factor and
    ``compare_after`` says whether the user also wants to run the comparison.
    """

    def __init__(
        self,
        original: trimesh.Trimesh,
        scan: trimesh.Trimesh,
        original_name: str,
        scan_name: str,
        canvas_size: int = 380,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Match scale")
        self.meshes = (original, scan)
        self.canvas_size = canvas_size
        self.scale = 1.0
        self.compare_after = False
        self.cameras: list[Camera] = []
        self.canvases = (ModelCanvas(canvas_size), ModelCanvas(canvas_size))
        self.size_labels = (QLabel(), QLabel())
        self._samples: dict[tuple[int, str], SurfaceSamples] = {}
        self._worker: TaskWorker | None = None

        self.view = QComboBox()
        self.view.addItems(list(VIEW_NAMES))
        self.up_axes = (QComboBox(), QComboBox())
        for combo in self.up_axes:
            combo.addItems(list(UP_AXES))
            combo.currentTextChanged.connect(self._render)
        self.view.currentTextChanged.connect(self._render)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(
            round(math.log10(MIN_SCALE) * SLIDER_STEPS_PER_DECADE),
            round(math.log10(MAX_SCALE) * SLIDER_STEPS_PER_DECADE),
        )
        self.slider.valueChanged.connect(
            lambda value: self.set_scale(10 ** (value / SLIDER_STEPS_PER_DECADE))
        )
        self.factor = QDoubleSpinBox(decimals=4, minimum=MIN_SCALE, maximum=MAX_SCALE, prefix="× ")
        self.factor.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        self.factor.valueChanged.connect(self.set_scale)

        quick = QHBoxLayout()
        for label, multiplier in (("× 1000  (m → mm)", 1000.0), ("× 10  (cm → mm)", 10.0)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, m=multiplier: self.set_scale(self.scale * m))
            quick.addWidget(button)
        self.auto_button = QPushButton("Auto-fit")
        self.auto_button.setToolTip(
            "Estimate the scale by fitting the scan's shape to the original"
        )
        self.auto_button.clicked.connect(self.auto_fit)
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: self.set_scale(1.0))
        quick.addWidget(self.auto_button)
        quick.addWidget(reset)
        quick.addStretch()

        self.status = QLabel(wordWrap=True)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        apply = QPushButton("Apply")
        apply.clicked.connect(lambda: self._finish(compare=False))
        self.compare_button = QPushButton("Apply && compare")
        self.compare_button.setDefault(True)
        self.compare_button.clicked.connect(lambda: self._finish(compare=True))

        top = QHBoxLayout()
        top.addWidget(QLabel("View:"))
        top.addWidget(self.view)
        top.addStretch()
        models = QGridLayout()
        for column, name in enumerate((original_name, f"{scan_name} (scaled)")):
            models.addWidget(QLabel(f"<b>{name}</b>"), 0, column)
            models.addWidget(self.canvases[column], 1, column)
            models.addWidget(self.size_labels[column], 2, column)
            up_row = QHBoxLayout()
            up_row.addWidget(QLabel("Up axis:"))
            up_row.addWidget(self.up_axes[column])
            up_row.addStretch()
            models.addLayout(up_row, 3, column)
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale:"))
        scale_row.addWidget(self.slider, stretch=1)
        scale_row.addWidget(self.factor)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status, stretch=1)
        buttons.addWidget(cancel)
        buttons.addWidget(apply)
        buttons.addWidget(self.compare_button)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Both models are drawn at the same scale over a millimetre grid. "
                "Scale the scan until it matches the original, or try Auto-fit.",
                wordWrap=True,
            )
        )
        layout.addLayout(top)
        # The models scroll on small screens so the controls below stay reachable.
        layout.addWidget(ScrollArea(models), stretch=1)
        layout.addLayout(scale_row)
        layout.addLayout(quick)
        layout.addLayout(buttons)
        self._render()
        self.set_scale(1.0)
        fit_to_screen(self)

    def set_scale(self, factor: float) -> None:
        """Set the scan's scale factor, keeping the slider and number box in sync."""
        factor = float(np.clip(factor, MIN_SCALE, MAX_SCALE))
        self.scale = factor
        self.slider.blockSignals(True)
        self.slider.setValue(round(math.log10(factor) * SLIDER_STEPS_PER_DECADE))
        self.slider.blockSignals(False)
        self.factor.blockSignals(True)
        self.factor.setValue(factor)
        self.factor.blockSignals(False)
        self._update_display()

    def auto_fit(self) -> None:
        """Estimate the scale in the background and apply it when done."""
        if self._worker is not None:
            return
        original, scan = self.meshes
        self.auto_button.setEnabled(False)
        self.status.setText("Estimating scale… (a few seconds)")
        self._worker = TaskWorker(lambda: estimate_scale(scan, original))
        self._worker.succeeded.connect(self._auto_fit_done)
        self._worker.failed.connect(
            lambda message: self.status.setText(f"Auto-fit failed: {message}")
        )
        self._worker.finished.connect(self._auto_fit_finished)
        self._worker.start()

    def _auto_fit_done(self, factor: float) -> None:
        self.set_scale(factor)
        self.status.setText(
            f"Auto-fit suggests × {factor:.4g}. Check the two look the same size, then apply."
        )

    def _auto_fit_finished(self) -> None:
        self._worker = None
        self.auto_button.setEnabled(True)

    def _finish(self, compare: bool) -> None:
        self.compare_after = compare
        self.accept()

    def _render(self) -> None:
        view_index = VIEW_NAMES.index(self.view.currentText())
        self.cameras = []
        for index, (mesh, canvas, combo) in enumerate(
            zip(self.meshes, self.canvases, self.up_axes, strict=True)
        ):
            up = combo.currentText()
            cameras = side_cameras(mesh, up, self.canvas_size)
            key = (index, up)
            if key not in self._samples:
                self._samples[key] = SurfaceSamples.from_mesh(mesh, cameras[0])
            camera = cameras[view_index]
            canvas.image = _model_image(render_view(self._samples[key], camera), MODEL_TINTS[index])
            self.cameras.append(camera)
        self._update_display()

    def _update_display(self) -> None:
        spans = [
            2 * camera.half_size * factor
            for camera, factor in zip(self.cameras, (1.0, self.scale), strict=True)
        ]
        mm_per_pixel = max(spans) / self.canvas_size
        ratio = spans[1] / spans[0]
        if self._worker is None:
            if ratio < 1 / SIZE_HINT_RATIO:
                self.status.setText(
                    f"The scan is {1 / ratio:.0f}× smaller than the original — "
                    "try × 1000 or Auto-fit."
                )
            elif ratio > SIZE_HINT_RATIO:
                self.status.setText(f"The scan is {ratio:.0f}× larger than the original.")
            elif self.status.text().startswith("The scan is"):
                self.status.setText("")
        step = grid_step(max(spans))
        for canvas, span in zip(self.canvases, spans, strict=True):
            canvas.image_span_mm = span
            canvas.mm_per_pixel = mm_per_pixel
            canvas.grid_mm = step
            canvas.update()

        for label, mesh, camera, factor in zip(
            self.size_labels, self.meshes, self.cameras, (1.0, self.scale), strict=True
        ):
            vertices = np.asarray(mesh.vertices) * factor
            width = np.ptp(vertices @ camera.right)
            height = np.ptp(vertices @ camera.up)
            box = " × ".join(f"{v:.2f}" for v in sorted(np.asarray(mesh.extents) * factor))
            label.setText(
                f"In this view: {width:.2f} × {height:.2f} mm (w × h)<br>" f"Bounding box: {box} mm"
            )


def _model_image(view: RenderedView, tint: np.ndarray) -> QImage:
    rgb = view.shade[..., None].astype(np.float64) * tint
    alpha = np.where(view.mask, 255, 0)[..., None]
    pixels = np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2).astype(np.uint8))
    height, width, _ = pixels.shape
    return QImage(pixels.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()
