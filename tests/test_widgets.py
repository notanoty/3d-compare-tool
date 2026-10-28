import os
from pathlib import Path

import pytest
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect, QSize  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mesh_compare.heatmap import build_heatmap  # noqa: E402
from mesh_compare.heatmap_dialog import HeatMapDialog  # noqa: E402
from mesh_compare.scale_dialog import ScaleDialog  # noqa: E402
from mesh_compare.trim_dialog import TrimDialog  # noqa: E402
from mesh_compare.widgets import MIN_AREA, FitImage, ScrollArea, fit_to_screen  # noqa: E402

# A small laptop screen once the menu bar and dock are taken off.
SMALL = QSize(640, 520)


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


def fully_inside(window: QWidget, widget: QWidget) -> bool:
    corner = widget.mapTo(window, QPoint(0, 0))
    return window.rect().contains(QRect(corner, widget.size()))


def buttons(window: QWidget, *labels: str) -> list[QPushButton]:
    found = [b for b in window.findChildren(QPushButton) if b.text() in labels]
    assert len(found) == len(labels)
    return found


@pytest.mark.parametrize("kind", ["trim", "scale", "heatmap"])
def test_dialogs_shrink_and_keep_buttons_reachable(
    app: QApplication, lumpy_mesh: trimesh.Trimesh, kind: str
) -> None:
    window: QDialog
    labels: tuple[str, ...]
    if kind == "trim":
        window = TrimDialog(lumpy_mesh, "model.stl", "scan.stl")
        labels = ("Skip", "Trim && save")
    elif kind == "scale":
        window = ScaleDialog(lumpy_mesh, lumpy_mesh, "original.stl", "scan.stl")
        labels = ("Cancel", "Apply", "Apply && compare")
    else:
        heatmap = build_heatmap(lumpy_mesh, lumpy_mesh, None, 0.1)
        window = HeatMapDialog(heatmap, "a vs b", Path("unused.png"), 0.5)
        labels = ("Save image…", "Close")

    assert window.minimumSizeHint().width() <= SMALL.width()
    assert window.minimumSizeHint().height() <= SMALL.height()
    window.show()
    window.resize(SMALL)
    app.processEvents()
    assert window.size() == SMALL
    for button in buttons(window, *labels):
        assert fully_inside(window, button), button.text()
    window.close()


def test_scroll_area_asks_for_full_content_but_can_shrink(app: QApplication) -> None:
    grid = QGridLayout()
    big = QLabel()
    big.setFixedSize(900, 1200)
    grid.addWidget(big, 0, 0)
    area = ScrollArea(grid)
    assert area.sizeHint().width() >= 900 and area.sizeHint().height() >= 1200
    assert area.minimumSizeHint() == MIN_AREA


def test_fit_to_screen_caps_window_to_screen(app: QApplication) -> None:
    window = QWidget()
    layout = QVBoxLayout(window)
    area_content = QGridLayout()
    huge = QLabel()
    huge.setFixedSize(5000, 5000)
    area_content.addWidget(huge, 0, 0)
    layout.addWidget(ScrollArea(area_content))
    fit_to_screen(window)
    screen = window.screen() or QApplication.primaryScreen()
    assert screen is not None
    free = screen.availableGeometry().size()
    assert window.width() <= free.width() and window.height() <= free.height()

    small = QWidget()
    small.setFixedSize(200, 150)
    fit_to_screen(small)
    assert small.size() == QSize(200, 150)


def test_fit_image_scales_down_not_up(app: QApplication) -> None:
    image = QImage(400, 200, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    widget = FitImage()
    assert widget.sizeHint() == MIN_AREA  # nothing to show yet
    widget.set_image(image)
    assert widget.sizeHint() == QSize(400, 200)

    widget.resize(200, 200)  # half width: drawn at 200 x 100, centred
    shown = widget.grab().toImage()
    assert shown.pixelColor(100, 100) == QColor("red")
    assert shown.pixelColor(100, 20) != QColor("red")

    widget.resize(800, 400)  # bigger than the image: not enlarged
    shown = widget.grab().toImage()
    assert shown.pixelColor(400, 200) == QColor("red")
    assert shown.pixelColor(150, 200) != QColor("red")
