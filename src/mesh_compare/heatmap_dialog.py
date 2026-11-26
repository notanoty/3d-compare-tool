"""Heat map window: the coloured original from four sides, with a legend and Save image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from mesh_compare.heatmap import (
    MISSING_COLOUR,
    DetailRange,
    HeatMap,
    deviation_colours,
    detail_colours,
)
from mesh_compare.trimming import UP_AXES
from mesh_compare.widgets import FitImage, fit_to_screen
from mesh_compare.workers import TaskWorker

MARGIN = 16
# Narrowest image that still fits the legend; smaller views are centred.
MIN_WIDTH = 600
TITLE_HEIGHT = 30
LABEL_HEIGHT = 20
LEGEND_HEIGHT = 76
CAPTION_HEIGHT = 24
BAR_HEIGHT = 16
SWATCH_WIDTH = 28
SAVE_FILTER = "PNG image (*.png);;JPEG image (*.jpg *.jpeg)"
# Colour scale choices in the window: standard (fixed tolerance band) or detail.
SCALES = {"Tolerance": False, "Detail (relative)": True}
DETAIL_SUFFIX = "_detail"
DETAIL_PERCENTILE_LABEL = "99th percentile (not missing)"


def _rgba_image(pixels: np.ndarray) -> QImage:
    pixels = np.ascontiguousarray(pixels)
    height, width, _ = pixels.shape
    return QImage(pixels.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()


def _colour(rgb: np.ndarray) -> QColor:
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def heatmap_image(
    heatmap: HeatMap, missing_threshold: float, title: str, detail: bool = False
) -> QImage:
    """Compose the four views, a colour legend and a summary line into one image.

    Args:
        heatmap: Heat map to draw.
        missing_threshold: Deviation beyond which the surface counts as missing.
        title: Text drawn at the top (e.g. the two file names).
        detail: Use the detail colour scale (green at the best-matching point,
            red at the largest non-missing deviation) instead of the tolerance band.

    Returns:
        An opaque image on a white background, ready to save.
    """
    size = heatmap.views[0].view.camera.size
    width = max(2 * size + 3 * MARGIN, MIN_WIDTH)
    grid_left = (width - 2 * size - MARGIN) / 2
    cell = LABEL_HEIGHT + size
    height = MARGIN + TITLE_HEIGHT + 2 * cell + MARGIN + LEGEND_HEIGHT + CAPTION_HEIGHT + MARGIN
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    bold = QFont(painter.font())
    bold.setBold(True)
    bold.setPointSize(12)
    painter.setFont(bold)
    painter.setPen(QColor("#111827"))
    painter.drawText(
        QRectF(MARGIN, MARGIN, width - 2 * MARGIN, TITLE_HEIGHT),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        title,
    )
    normal = QFont(painter.font())
    normal.setBold(False)
    normal.setPointSize(10)
    painter.setFont(normal)

    top = MARGIN + TITLE_HEIGHT
    for index, item in enumerate(heatmap.views):
        x = grid_left + (index % 2) * (size + MARGIN)
        y = top + (index // 2) * cell
        painter.setPen(QColor("#374151"))
        painter.drawText(QRectF(x, y, size, LABEL_HEIGHT), Qt.AlignmentFlag.AlignCenter, item.name)
        painter.setPen(QColor("#e5e7eb"))
        painter.drawRect(QRectF(x, y + LABEL_HEIGHT, size, size))
        painter.drawImage(
            QPointF(x, y + LABEL_HEIGHT),
            _rgba_image(heatmap.rgba(index, missing_threshold, detail)),
        )

    legend_area = QRectF(MARGIN, top + 2 * cell + MARGIN, width - 2 * MARGIN, LEGEND_HEIGHT)
    if detail:
        _draw_detail_legend(painter, heatmap.detail_range(missing_threshold), missing_threshold, legend_area)
    else:
        _draw_legend(painter, heatmap.tolerance, missing_threshold, legend_area)
    caption = (
        f"Missing in scan (|deviation| > {missing_threshold:g} mm): "
        f"{heatmap.missing_pct(missing_threshold):.1f} % of the visible original surface"
    )
    if detail:
        colour_range = heatmap.detail_range(missing_threshold)
        caption += f"  ·  detail scale {colour_range.low:.3f}–{colour_range.high:.3f} mm"
    painter.setPen(QColor("#111827"))
    painter.drawText(
        QRectF(MARGIN, height - MARGIN - CAPTION_HEIGHT, width - 2 * MARGIN, CAPTION_HEIGHT),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        caption,
    )
    painter.end()
    return image


def _bar(colours: np.ndarray) -> QImage:
    row = np.concatenate([colours, np.full((len(colours), 1), 255, dtype=np.uint8)], axis=1)
    return _rgba_image(np.ascontiguousarray(np.repeat(row[None], BAR_HEIGHT, axis=0)))


def _draw_missing_swatch(painter: QPainter, threshold: float, left: float, top: float) -> None:
    painter.fillRect(QRectF(left, top, SWATCH_WIDTH, BAR_HEIGHT), _colour(MISSING_COLOUR))
    painter.drawText(
        QRectF(left + SWATCH_WIDTH + 6, top - 2, 140, BAR_HEIGHT + 4),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        "Missing in scan",
    )
    painter.drawText(
        QRectF(left, top + BAR_HEIGHT + 4, 170, 16),
        Qt.AlignmentFlag.AlignLeft,
        f"beyond ±{threshold:g}",
    )


def _draw_detail_legend(
    painter: QPainter, colour_range: DetailRange, threshold: float, area: QRectF
) -> None:
    bar_width = int(area.width() - 2 * SWATCH_WIDTH - 160)
    bar_left = area.left()
    bar_top = area.top() + 18
    values = np.linspace(colour_range.low, colour_range.high, bar_width)
    painter.drawImage(int(bar_left), int(bar_top), _bar(detail_colours(values, colour_range, threshold)))

    painter.setPen(QColor("#374151"))
    painter.drawText(
        QRectF(bar_left, area.top(), bar_width, 16),
        Qt.AlignmentFlag.AlignCenter,
        "|Deviation| of the scan from the original (mm), relative scale",
    )
    span = colour_range.high - colour_range.low
    for k in range(5):
        value = colour_range.low + span * k / 4
        x = bar_left + k / 4 * (bar_width - 1)
        painter.drawLine(int(x), int(bar_top + BAR_HEIGHT), int(x), int(bar_top + BAR_HEIGHT + 4))
        if k == 0:
            rect, align = QRectF(x, 0, 60, 16), Qt.AlignmentFlag.AlignLeft
        elif k == 4:
            rect, align = QRectF(x - 60, 0, 60, 16), Qt.AlignmentFlag.AlignRight
        else:
            rect, align = QRectF(x - 30, 0, 60, 16), Qt.AlignmentFlag.AlignHCenter
        rect.moveTop(bar_top + BAR_HEIGHT + 4)
        painter.drawText(rect, align, f"{value:.3f}")
    hint_top = bar_top + BAR_HEIGHT + 22
    painter.drawText(
        QRectF(bar_left, hint_top, bar_width / 2, 16),
        Qt.AlignmentFlag.AlignLeft,
        "← best-matching point",
    )
    painter.drawText(
        QRectF(bar_left + bar_width / 2, hint_top, bar_width / 2, 16),
        Qt.AlignmentFlag.AlignRight,
        f"{DETAIL_PERCENTILE_LABEL} →",
    )
    _draw_missing_swatch(painter, threshold, bar_left + bar_width + SWATCH_WIDTH, bar_top)


def _draw_legend(painter: QPainter, tolerance: float, threshold: float, area: QRectF) -> None:
    bar_width = int(area.width() - 2 * SWATCH_WIDTH - 160)
    bar_left = area.left()
    bar_top = area.top() + 18

    values = np.linspace(-threshold, threshold, bar_width)
    painter.drawImage(int(bar_left), int(bar_top), _bar(deviation_colours(values, tolerance, threshold)))

    painter.setPen(QColor("#374151"))
    painter.drawText(
        QRectF(bar_left, area.top(), bar_width, 16),
        Qt.AlignmentFlag.AlignCenter,
        "Deviation of the scan from the original (mm)",
    )
    for value in (-threshold, -tolerance, 0.0, tolerance, threshold):
        x = bar_left + (value + threshold) / (2 * threshold) * (bar_width - 1)
        painter.drawLine(int(x), int(bar_top + BAR_HEIGHT), int(x), int(bar_top + BAR_HEIGHT + 4))
        # The end labels hang inward so they stay inside the image.
        if value == -threshold:
            rect, align = QRectF(x, 0, 60, 16), Qt.AlignmentFlag.AlignLeft
        elif value == threshold:
            rect, align = QRectF(x - 60, 0, 60, 16), Qt.AlignmentFlag.AlignRight
        else:
            rect, align = QRectF(x - 30, 0, 60, 16), Qt.AlignmentFlag.AlignHCenter
        rect.moveTop(bar_top + BAR_HEIGHT + 4)
        painter.drawText(rect, align, f"{value:+g}" if value else "0")
    hint_top = bar_top + BAR_HEIGHT + 22
    painter.drawText(
        QRectF(bar_left, hint_top, bar_width / 2, 16),
        Qt.AlignmentFlag.AlignLeft,
        "← scan smaller",
    )
    painter.drawText(
        QRectF(bar_left + bar_width / 2, hint_top, bar_width / 2, 16),
        Qt.AlignmentFlag.AlignRight,
        "scan larger →",
    )

    _draw_missing_swatch(painter, threshold, bar_left + bar_width + SWATCH_WIDTH, bar_top)


class HeatMapDialog(QDialog):
    """Shows the deviation heat map and saves it as an image."""

    def __init__(
        self, heatmap: HeatMap, title: str, default_path: Path, missing_threshold: float
    ) -> None:
        super().__init__()
        self.setWindowTitle("Deviation heat map")
        self.heatmap = heatmap
        self.title = title
        self.default_path = default_path
        self.image = QImage()
        self._cache: dict[str, HeatMap] = {heatmap.up_axis: heatmap}
        self._worker: TaskWorker | None = None
        # Holds the last finished worker: dropping it inside its own ``finished``
        # signal would delete the thread object while Qt is still using it.
        self._finished_worker: TaskWorker | None = None

        # Scaled to fit the window; Save image always writes the full-size image.
        self.picture = FitImage()
        self.up_axis = QComboBox()
        self.up_axis.addItems(list(UP_AXES))
        self.up_axis.setCurrentText(heatmap.up_axis)
        self.up_axis.currentTextChanged.connect(self.set_up_axis)
        self.threshold = QDoubleSpinBox(
            minimum=round(heatmap.tolerance + 0.01, 2),
            maximum=50.0,
            singleStep=0.1,
            decimals=2,
            suffix=" mm",
        )
        self.threshold.setValue(missing_threshold)
        self.threshold.setToolTip("Deviations larger than this are shown as missing (magenta)")
        self.threshold.valueChanged.connect(self._redraw)
        self.scale = QComboBox()
        self.scale.addItems(list(SCALES))
        self.scale.setToolTip(
            "Tolerance: green inside ±tolerance. Detail: green at the best-matching point, "
            "red at the largest deviation, so differences inside the tolerance show up"
        )
        self.scale.currentTextChanged.connect(lambda _: self._redraw())
        self.save_button = QPushButton("Save image…")
        self.save_button.clicked.connect(lambda: self.save_image())
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        self.status = QLabel(wordWrap=True)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Up axis:"))
        controls.addWidget(self.up_axis)
        controls.addWidget(QLabel("Missing beyond:"))
        controls.addWidget(self.threshold)
        controls.addWidget(QLabel("Colours:"))
        controls.addWidget(self.scale)
        controls.addStretch()
        controls.addWidget(self.save_button)
        controls.addWidget(close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.picture, stretch=1)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        self._redraw()
        fit_to_screen(self)

    def set_up_axis(self, up_axis: str) -> None:
        """Show the heat map around ``up_axis``, rendering it in the background if needed."""
        if up_axis in self._cache:
            self.heatmap = self._cache[up_axis]
            self._redraw()
            return
        if self._worker is not None:
            return
        current = self.heatmap
        self.up_axis.setEnabled(False)
        self.status.setText("Rendering…")
        self._worker = TaskWorker(lambda: current.with_up_axis(up_axis))
        self._worker.succeeded.connect(self._rendered)
        self._worker.failed.connect(lambda message: self.status.setText(f"Error: {message}"))
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    @property
    def detail(self) -> bool:
        """Whether the detail colour scale is shown."""
        return SCALES[self.scale.currentText()]

    def save_image(self, path: Path | None = None) -> bool:
        """Save the heat map image, asking where when ``path`` is not given."""
        if path is None:
            default = self.default_path
            if self.detail:
                default = default.with_name(default.stem + DETAIL_SUFFIX + default.suffix)
            chosen, _ = QFileDialog.getSaveFileName(self, "Save heat map", str(default), SAVE_FILTER)
            if not chosen:
                return False
            path = Path(chosen)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            path = path.with_suffix(".png")
        if self.image.save(str(path)):
            self.status.setText(f"Saved {path.name}")
            return True
        self.status.setText(f"Error: could not save {path}")
        return False

    def done(self, result: int) -> None:
        if self._worker is not None:
            self._worker.wait()
        super().done(result)

    def _rendered(self, heatmap: HeatMap) -> None:
        self._cache[heatmap.up_axis] = heatmap
        self.heatmap = heatmap
        self.status.setText("")
        self._redraw()

    def _worker_finished(self) -> None:
        self._finished_worker, self._worker = self._worker, None
        self.up_axis.setEnabled(True)

    def _redraw(self) -> None:
        self.image = heatmap_image(self.heatmap, self.threshold.value(), self.title, self.detail)
        self.picture.set_image(self.image)
