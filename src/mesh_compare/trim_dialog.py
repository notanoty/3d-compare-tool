"""Trimming window: four side views, a draggable cut line, and top/bottom removal."""

from __future__ import annotations

import numpy as np
import trimesh
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mesh_compare.trimming import (
    UP_AXES,
    VIEW_NAMES,
    RemoveSide,
    RenderedView,
    SurfaceSamples,
    TrimPlane,
    default_line,
    line_on_view,
    plane_from_line,
    render_view,
    side_cameras,
)

BACKGROUND = np.array([245, 245, 245], dtype=np.float64)
REMOVED_TINT = np.array([230, 40, 40], dtype=np.float64)
LINE_COLOUR = QColor("#2563eb")
HANDLE_RADIUS = 6.0
GRAB_DISTANCE = 12.0
# Mouse travel (pixels) before a press on a view starts drawing a new line.
DRAW_THRESHOLD = 5.0

Point = tuple[float, float]


class ViewCanvas(QWidget):
    """Shows one rendered view with the cut line drawn on top and reports mouse input."""

    pressed = Signal(QPointF)
    dragged = Signal(QPointF)
    released = Signal()

    def __init__(self, size: int) -> None:
        super().__init__()
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.image = QImage()
        self.lines: list[tuple[QPointF, QPointF]] = []
        self.handles: list[QPointF] = []

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(0, 0, self.image)
        painter.setPen(QPen(LINE_COLOUR, 2, Qt.PenStyle.DashLine))
        for start, end in self.lines:
            painter.drawLine(start, end)
        painter.setPen(QPen(LINE_COLOUR, 2))
        painter.setBrush(QColor("white"))
        for handle in self.handles:
            painter.drawEllipse(handle, HANDLE_RADIUS, HANDLE_RADIUS)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        self.pressed.emit(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        self.dragged.emit(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        self.released.emit()


class TrimDialog(QDialog):
    """Lets the user place one cut plane by drawing a line on any of four side views.

    After ``exec()`` returns ``Accepted``, ``plane`` holds the cut (in the mesh's own
    coordinates) and ``copy_to_other`` says whether to repeat it on the other model.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        title: str,
        other_name: str | None = None,
        view_size: int = 360,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"Trim — {title}")
        self.mesh = mesh
        self.view_size = view_size
        self.remove: RemoveSide = "bottom"
        self.source_view = 0
        self.line: list[Point] = []
        self.views: list[RenderedView] = []
        self.samples: SurfaceSamples | None = None
        self._grabbed_end: int | None = None
        self._press: tuple[int, Point] | None = None

        self.canvases = [ViewCanvas(view_size) for _ in VIEW_NAMES]
        grid = QGridLayout()
        for index, (name, canvas) in enumerate(zip(VIEW_NAMES, self.canvases, strict=True)):
            canvas.pressed.connect(lambda pos, i=index: self._on_press(i, pos))
            canvas.dragged.connect(lambda pos, i=index: self._on_drag(i, pos))
            canvas.released.connect(self._on_release)
            cell = QVBoxLayout()
            cell.addWidget(QLabel(f"<b>{name}</b>", alignment=Qt.AlignmentFlag.AlignCenter))
            cell.addWidget(canvas)
            grid.addLayout(cell, index // 2, index % 2)

        self.up_axis = QComboBox()
        self.up_axis.addItems(list(UP_AXES))
        self.up_axis.currentTextChanged.connect(self._render_views)
        self.remove_top = QPushButton("Remove top")
        self.remove_bottom = QPushButton("Remove bottom")
        side_group = QButtonGroup(self)
        for button in (self.remove_top, self.remove_bottom):
            button.setCheckable(True)
            side_group.addButton(button)
        self.remove_bottom.setChecked(True)
        self.remove_top.clicked.connect(lambda: self.set_remove("top"))
        self.remove_bottom.clicked.connect(lambda: self.set_remove("bottom"))

        self.copy_box = QCheckBox(
            f"Also apply this cut to “{other_name}”" if other_name else "Also apply to other model"
        )
        self.copy_box.setChecked(other_name is not None)
        self.copy_box.setEnabled(other_name is not None)
        self.info = QLabel()
        skip = QPushButton("Skip")
        skip.clicked.connect(self.reject)
        self.trim_button = QPushButton("Trim && save")
        self.trim_button.setDefault(True)
        self.trim_button.clicked.connect(self.accept)

        options = QHBoxLayout()
        options.addWidget(QLabel("Up axis:"))
        options.addWidget(self.up_axis)
        options.addSpacing(16)
        options.addWidget(self.remove_top)
        options.addWidget(self.remove_bottom)
        options.addStretch()
        buttons = QHBoxLayout()
        buttons.addWidget(self.copy_box)
        buttons.addStretch()
        buttons.addWidget(skip)
        buttons.addWidget(self.trim_button)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Drag on any view to draw the cut line; drag its round ends to adjust. "
                "Red shows what will be removed.",
                wordWrap=True,
            )
        )
        layout.addLayout(options)
        layout.addLayout(grid)
        layout.addWidget(self.info)
        layout.addLayout(buttons)
        self._render_views()

    @property
    def plane(self) -> TrimPlane | None:
        """The current cut, or ``None`` if the line is too short to define one."""
        try:
            return plane_from_line(
                self.views[self.source_view].camera, self.line[0], self.line[1], self.remove
            )
        except ValueError:
            return None

    @property
    def copy_to_other(self) -> bool:
        """Whether the user asked to apply the same cut to the other model."""
        return self.copy_box.isEnabled() and self.copy_box.isChecked()

    def set_remove(self, side: RemoveSide) -> None:
        """Choose which side of the line is cut away."""
        self.remove = side
        (self.remove_top if side == "top" else self.remove_bottom).setChecked(True)
        self._refresh()

    def set_line(self, view: int, start: Point, end: Point) -> None:
        """Place the cut line on ``view`` (pixel coordinates)."""
        self.source_view = view
        self.line = [start, end]
        self._refresh()

    def _render_views(self) -> None:
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            cameras = side_cameras(self.mesh, self.up_axis.currentText(), self.view_size)
            self.samples = SurfaceSamples.from_mesh(self.mesh, cameras[0])
            self.views = [render_view(self.samples, camera) for camera in cameras]
        finally:
            QGuiApplication.restoreOverrideCursor()
        start, end = default_line(cameras[0], self.mesh)
        self.set_line(0, start, end)

    def _refresh(self) -> None:
        plane = self.plane
        for index, (view, canvas) in enumerate(zip(self.views, self.canvases, strict=True)):
            canvas.image = _view_image(view, plane)
            canvas.lines = []
            canvas.handles = []
            if plane is not None and (edge_on := line_on_view(plane, view.camera)) is not None:
                point, direction = edge_on
                reach = 2.0 * self.view_size
                canvas.lines.append(
                    (QPointF(*(point - reach * direction)), QPointF(*(point + reach * direction)))
                )
            if index == self.source_view:
                canvas.handles = [QPointF(*end) for end in self.line]
            canvas.update()

        if plane is None or self.samples is None:
            self.info.setText("Draw a longer line to define the cut.")
            self.trim_button.setEnabled(False)
            return
        removed = self.samples.removed_fraction(plane)
        self.trim_button.setEnabled(removed < 0.999)
        warning = "  — this removes almost everything!" if removed >= 0.999 else ""
        self.info.setText(f"Removes about {removed * 100:.1f}% of the surface.{warning}")

    def _on_press(self, view: int, pos: QPointF) -> None:
        point = (pos.x(), pos.y())
        self._press = (view, point)
        self._grabbed_end = None
        if view == self.source_view:
            distances = [np.hypot(point[0] - x, point[1] - y) for x, y in self.line]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= GRAB_DISTANCE:
                self._grabbed_end = nearest

    def _on_drag(self, view: int, pos: QPointF) -> None:
        point = (pos.x(), pos.y())
        if self._grabbed_end is not None:
            self.line[self._grabbed_end] = point
            self._refresh()
        elif self._press is not None:
            press_view, start = self._press
            if np.hypot(point[0] - start[0], point[1] - start[1]) >= DRAW_THRESHOLD:
                self.set_line(press_view, start, point)
                self._grabbed_end = 1

    def _on_release(self) -> None:
        self._press = None
        self._grabbed_end = None


def _view_image(view: RenderedView, plane: TrimPlane | None) -> QImage:
    rgb = np.repeat(view.shade[..., None].astype(np.float64), 3, axis=2)
    rgb[~view.mask] = BACKGROUND
    if plane is not None:
        removed = view.removed(plane)
        rgb[removed] = rgb[removed] * 0.45 + REMOVED_TINT * 0.55
    pixels = np.ascontiguousarray(rgb.astype(np.uint8))
    height, width, _ = pixels.shape
    return QImage(pixels.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
