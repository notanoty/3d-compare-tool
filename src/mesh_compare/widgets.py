"""Small shared widgets that let the windows shrink to fit small screens."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import QLayout, QScrollArea, QSizePolicy, QWidget

# Smallest size a scrolling or scaling area may be squeezed to.
MIN_AREA = QSize(240, 180)
# Share of the screen's free area a window may take when it opens.
SCREEN_FRACTION = 0.9


class ScrollArea(QScrollArea):
    """Scrolls ``layout`` when the window is too small to show all of it.

    Asks for the full size of its contents, so a window opens showing
    everything when the screen is big enough, yet can shrink to ``MIN_AREA``.
    """

    def __init__(self, layout: QLayout) -> None:
        super().__init__()
        content = QWidget()
        content.setLayout(layout)
        self.setWidget(content)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        frame = 2 * self.frameWidth()
        content = self.widget()
        return (content.sizeHint() if content else MIN_AREA) + QSize(frame, frame)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return MIN_AREA


class FitImage(QWidget):
    """Shows an image scaled down (never up) to fit, keeping its aspect ratio."""

    def __init__(self) -> None:
        super().__init__()
        self.image = QImage()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image: QImage) -> None:
        self.image = image
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return self.image.size() if not self.image.isNull() else MIN_AREA

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return MIN_AREA

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt API)
        if self.image.isNull():
            return
        scale = min(1.0, self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        target = QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self.image)
        painter.end()


def fit_to_screen(window: QWidget) -> None:
    """Resize ``window`` to its preferred size, but no larger than the screen allows."""
    screen = window.screen() or QGuiApplication.primaryScreen()
    wanted = window.sizeHint()
    if screen is None:
        window.resize(wanted)
        return
    free = screen.availableGeometry().size() * SCREEN_FRACTION
    window.resize(wanted.boundedTo(free).expandedTo(window.minimumSizeHint()))
