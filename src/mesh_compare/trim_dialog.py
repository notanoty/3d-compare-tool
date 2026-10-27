"""Trimming window: four side views, any number of draggable cut lines, top/bottom per line."""

from __future__ import annotations

from dataclasses import dataclass

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
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mesh_compare.trimming import (
    UP_AXES,
    VIEW_NAMES,
    Camera,
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
CUT_COLOURS = tuple(QColor(c) for c in ("#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2"))
HANDLE_RADIUS = 6.0
GRAB_DISTANCE = 12.0
# Mouse travel (pixels) before a press on a view starts drawing a new line.
DRAW_THRESHOLD = 5.0
# Where a newly added cut starts, as a fraction of the model's height.
NEW_CUT_HEIGHTS = {"bottom": 0.1, "top": 0.9}

Point = tuple[float, float]


@dataclass
class CutLine:
    """One cut: a line drawn on ``view`` and which side of it is removed."""

    view: int
    start: Point
    end: Point
    remove: RemoveSide

    def plane(self, cameras: list[Camera]) -> TrimPlane | None:
        """The cutting plane, or ``None`` if the line is too short to define one."""
        try:
            return plane_from_line(cameras[self.view], self.start, self.end, self.remove)
        except ValueError:
            return None

    def describe(self, number: int) -> str:
        return f"Cut {number}: remove {self.remove} (drawn on {VIEW_NAMES[self.view]})"


@dataclass(frozen=True)
class CanvasLine:
    start: QPointF
    end: QPointF
    colour: QColor
    active: bool


class ViewCanvas(QWidget):
    """Shows one rendered view with the cut lines drawn on top and reports mouse input."""

    pressed = Signal(QPointF)
    dragged = Signal(QPointF)
    released = Signal()

    def __init__(self, size: int) -> None:
        super().__init__()
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.image = QImage()
        self.lines: list[CanvasLine] = []
        self.handles: list[tuple[QPointF, QColor]] = []

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(0, 0, self.image)
        for line in self.lines:
            width = 2.5 if line.active else 1.5
            painter.setPen(QPen(line.colour, width, Qt.PenStyle.DashLine))
            painter.drawLine(line.start, line.end)
        painter.setBrush(QColor("white"))
        for handle, colour in self.handles:
            painter.setPen(QPen(colour, 2))
            painter.drawEllipse(handle, HANDLE_RADIUS, HANDLE_RADIUS)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        self.pressed.emit(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        self.dragged.emit(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        self.released.emit()


class TrimDialog(QDialog):
    """Lets the user place cut planes by drawing lines on any of four side views.

    After ``exec()`` returns ``Accepted``, ``planes`` holds the cuts (in the mesh's
    own coordinates) and ``copy_to_other`` says whether to repeat them on the other
    model.
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
        self.cuts: list[CutLine] = []
        self.active = 0
        self.views: list[RenderedView] = []
        self.samples: SurfaceSamples | None = None
        self._grabbed: tuple[int, int] | None = None  # (cut index, line end)
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

        self.cut_list = QListWidget()
        self.cut_list.setMaximumHeight(90)
        self.cut_list.currentRowChanged.connect(self.select_cut)
        self.add_button = QPushButton("Add cut")
        self.add_button.clicked.connect(lambda: self.add_cut())
        self.delete_button = QPushButton("Delete cut")
        self.delete_button.clicked.connect(self.delete_cut)
        self.remove_top = QPushButton("Remove top")
        self.remove_bottom = QPushButton("Remove bottom")
        side_group = QButtonGroup(self)
        for button in (self.remove_top, self.remove_bottom):
            button.setCheckable(True)
            side_group.addButton(button)
        self.remove_top.clicked.connect(lambda: self.set_remove("top"))
        self.remove_bottom.clicked.connect(lambda: self.set_remove("bottom"))

        self.copy_box = QCheckBox(
            f"Also apply these cuts to “{other_name}”"
            if other_name
            else "Also apply to other model"
        )
        self.copy_box.setChecked(False)
        self.copy_box.setEnabled(other_name is not None)
        self.info = QLabel()
        skip = QPushButton("Skip")
        skip.clicked.connect(self.reject)
        self.trim_button = QPushButton("Trim && save")
        self.trim_button.setDefault(True)
        self.trim_button.clicked.connect(self.accept)

        cut_buttons = QVBoxLayout()
        cut_buttons.addWidget(self.add_button)
        cut_buttons.addWidget(self.delete_button)
        cut_buttons.addStretch()
        side_buttons = QVBoxLayout()
        side_buttons.addWidget(QLabel("Selected cut:"))
        side_buttons.addWidget(self.remove_top)
        side_buttons.addWidget(self.remove_bottom)
        side_buttons.addStretch()
        up_row = QVBoxLayout()
        up_row.addWidget(QLabel("Up axis:"))
        up_row.addWidget(self.up_axis)
        up_row.addStretch()
        cuts_row = QHBoxLayout()
        cuts_row.addWidget(self.cut_list, stretch=1)
        cuts_row.addLayout(cut_buttons)
        cuts_row.addLayout(side_buttons)
        cuts_row.addLayout(up_row)
        buttons = QHBoxLayout()
        buttons.addWidget(self.copy_box)
        buttons.addStretch()
        buttons.addWidget(skip)
        buttons.addWidget(self.trim_button)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Drag on any view to redraw the selected cut; drag a line's round ends to "
                "adjust it. Red shows everything that will be removed.",
                wordWrap=True,
            )
        )
        layout.addLayout(cuts_row)
        layout.addLayout(grid)
        layout.addWidget(self.info)
        layout.addLayout(buttons)
        self._render_views()

    @property
    def cameras(self) -> list[Camera]:
        return [view.camera for view in self.views]

    @property
    def planes(self) -> list[TrimPlane] | None:
        """All cutting planes, or ``None`` if any line is too short to define one."""
        planes = [cut.plane(self.cameras) for cut in self.cuts]
        valid = [plane for plane in planes if plane is not None]
        return valid if len(valid) == len(planes) else None

    @property
    def copy_to_other(self) -> bool:
        """Whether the user asked to apply the same cuts to the other model."""
        return self.copy_box.isEnabled() and self.copy_box.isChecked()

    def add_cut(self, remove: RemoveSide | None = None) -> None:
        """Add a level cut on the front view and select it.

        By default the first cut removes the bottom and later ones remove the top,
        each starting near that end of the model.
        """
        side: RemoveSide = remove or ("bottom" if not self.cuts else "top")
        start, end = default_line(self.cameras[0], self.mesh, NEW_CUT_HEIGHTS[side])
        self.cuts.append(CutLine(0, start, end, side))
        self.select_cut(len(self.cuts) - 1)

    def delete_cut(self) -> None:
        """Delete the selected cut (the last remaining cut cannot be deleted)."""
        if len(self.cuts) <= 1:
            return
        del self.cuts[self.active]
        self.select_cut(min(self.active, len(self.cuts) - 1))

    def select_cut(self, index: int) -> None:
        """Make cut ``index`` the one that the side buttons and drawing act on."""
        if not 0 <= index < len(self.cuts):
            return
        self.active = index
        self._refresh()

    def set_remove(self, side: RemoveSide) -> None:
        """Choose which side of the selected cut's line is cut away."""
        self.cuts[self.active].remove = side
        self._refresh()

    def set_line(self, view: int, start: Point, end: Point) -> None:
        """Place the selected cut's line on ``view`` (pixel coordinates)."""
        cut = self.cuts[self.active]
        cut.view, cut.start, cut.end = view, start, end
        self._refresh()

    def _render_views(self) -> None:
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            cameras = side_cameras(self.mesh, self.up_axis.currentText(), self.view_size)
            self.samples = SurfaceSamples.from_mesh(self.mesh, cameras[0])
            self.views = [render_view(self.samples, camera) for camera in cameras]
        finally:
            QGuiApplication.restoreOverrideCursor()
        # Lines are drawn in screen space, so a new "up" invalidates them.
        self.cuts = []
        self.add_cut("bottom")

    def _refresh(self) -> None:
        cameras = self.cameras
        planes = [cut.plane(cameras) for cut in self.cuts]
        valid = [plane for plane in planes if plane is not None]

        for view_index, (view, canvas) in enumerate(zip(self.views, self.canvases, strict=True)):
            canvas.image = _view_image(view, valid)
            canvas.lines = []
            canvas.handles = []
            for cut_index, (cut, plane) in enumerate(zip(self.cuts, planes, strict=True)):
                colour = CUT_COLOURS[cut_index % len(CUT_COLOURS)]
                active = cut_index == self.active
                if plane is not None and (edge_on := line_on_view(plane, view.camera)):
                    point, direction = edge_on
                    reach = 2.0 * self.view_size
                    canvas.lines.append(
                        CanvasLine(
                            QPointF(*(point - reach * direction)),
                            QPointF(*(point + reach * direction)),
                            colour,
                            active,
                        )
                    )
                if cut.view == view_index:
                    canvas.handles += [(QPointF(*cut.start), colour), (QPointF(*cut.end), colour)]
            canvas.update()

        self.cut_list.blockSignals(True)
        self.cut_list.clear()
        for number, cut in enumerate(self.cuts, start=1):
            self.cut_list.addItem(cut.describe(number))
            item = self.cut_list.item(number - 1)
            item.setForeground(CUT_COLOURS[(number - 1) % len(CUT_COLOURS)])
        self.cut_list.setCurrentRow(self.active)
        self.cut_list.blockSignals(False)
        side = self.cuts[self.active].remove
        (self.remove_top if side == "top" else self.remove_bottom).setChecked(True)
        self.delete_button.setEnabled(len(self.cuts) > 1)

        if len(valid) < len(planes) or self.samples is None:
            self.info.setText("Draw a longer line to define every cut.")
            self.trim_button.setEnabled(False)
            return
        removed = self.samples.removed_fraction(valid)
        self.trim_button.setEnabled(removed < 0.999)
        warning = "  — this removes almost everything!" if removed >= 0.999 else ""
        self.info.setText(f"Removes about {removed * 100:.1f}% of the surface.{warning}")

    def _on_press(self, view: int, pos: QPointF) -> None:
        point = (pos.x(), pos.y())
        self._press = (view, point)
        self._grabbed = None
        best = GRAB_DISTANCE
        # Prefer the selected cut when handles overlap.
        order = [self.active] + [i for i in range(len(self.cuts)) if i != self.active]
        for cut_index in order:
            cut = self.cuts[cut_index]
            if cut.view != view:
                continue
            for end_index, (x, y) in enumerate((cut.start, cut.end)):
                distance = float(np.hypot(point[0] - x, point[1] - y))
                if distance < best:
                    best, self._grabbed = distance, (cut_index, end_index)
        if self._grabbed is not None and self._grabbed[0] != self.active:
            self.select_cut(self._grabbed[0])

    def _on_drag(self, view: int, pos: QPointF) -> None:
        point = (pos.x(), pos.y())
        if self._grabbed is not None:
            cut = self.cuts[self._grabbed[0]]
            if self._grabbed[1] == 0:
                cut.start = point
            else:
                cut.end = point
            self._refresh()
        elif self._press is not None:
            press_view, start = self._press
            if np.hypot(point[0] - start[0], point[1] - start[1]) >= DRAW_THRESHOLD:
                self.set_line(press_view, start, point)
                self._grabbed = (self.active, 1)

    def _on_release(self) -> None:
        self._press = None
        self._grabbed = None


def _view_image(view: RenderedView, planes: list[TrimPlane]) -> QImage:
    rgb = np.repeat(view.shade[..., None].astype(np.float64), 3, axis=2)
    rgb[~view.mask] = BACKGROUND
    removed = view.removed(planes)
    rgb[removed] = rgb[removed] * 0.45 + REMOVED_TINT * 0.55
    pixels = np.ascontiguousarray(rgb.astype(np.uint8))
    height, width, _ = pixels.shape
    return QImage(pixels.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
