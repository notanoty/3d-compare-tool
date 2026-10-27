import os
from collections.abc import Iterator

import numpy as np
import pytest
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mesh_compare.trim_dialog import TrimDialog  # noqa: E402

SIZE = 160


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


@pytest.fixture
def dialog(app: QApplication, lumpy_mesh: trimesh.Trimesh) -> Iterator[TrimDialog]:
    dlg = TrimDialog(lumpy_mesh, "model.stl", other_name="scan.stl", view_size=SIZE)
    yield dlg
    dlg.close()


def drag(
    dialog: TrimDialog, view: int, start: tuple[float, float], end: tuple[float, float]
) -> None:
    canvas = dialog.canvases[view]
    canvas.pressed.emit(QPointF(*start))
    canvas.dragged.emit(QPointF(*end))
    canvas.released.emit()


def test_opens_with_default_cut(dialog: TrimDialog) -> None:
    assert dialog.plane is not None
    assert dialog.source_view == 0
    assert "Removes about" in dialog.info.text()
    assert dialog.trim_button.isEnabled()
    assert dialog.copy_to_other
    for canvas in dialog.canvases:
        assert not canvas.image.isNull()
        assert len(canvas.lines) == 1  # a level cut is edge-on in every side view
    assert len(dialog.canvases[0].handles) == 2
    assert dialog.canvases[1].handles == []
    dialog.canvases[0].grab()  # exercises painting


def test_remove_side_flips_plane(dialog: TrimDialog) -> None:
    bottom = dialog.plane
    dialog.remove_top.click()
    top = dialog.plane
    assert bottom is not None and top is not None
    assert dialog.remove == "top"
    assert np.allclose(bottom.normal, -top.normal)
    dialog.set_remove("bottom")
    assert dialog.remove_bottom.isChecked()


def test_drawing_on_another_view_moves_the_line_there(dialog: TrimDialog) -> None:
    drag(dialog, 2, (10.0, 100.0), (150.0, 80.0))
    assert dialog.source_view == 2
    assert dialog.line == [(10.0, 100.0), (150.0, 80.0)]
    assert len(dialog.canvases[2].handles) == 2
    assert dialog.canvases[0].handles == []
    # Tilted cut: drawn as a line on the views facing it, shading only on the others.
    assert [len(c.lines) for c in dialog.canvases] == [1, 0, 1, 0]


def test_dragging_a_handle_moves_only_that_end(dialog: TrimDialog) -> None:
    start, end = dialog.line
    drag(dialog, 0, start, (start[0] + 3, start[1] - 30))
    assert dialog.line == [(start[0] + 3, start[1] - 30), end]


def test_tiny_drag_keeps_existing_line(dialog: TrimDialog) -> None:
    before = list(dialog.line)
    drag(dialog, 1, (80.0, 80.0), (82.0, 81.0))
    assert dialog.line == before
    assert dialog.source_view == 0


def test_cutting_everything_disables_trim(dialog: TrimDialog) -> None:
    dialog.set_line(0, (0.0, 2.0), (SIZE, 2.0))
    assert not dialog.trim_button.isEnabled()
    assert "almost everything" in dialog.info.text()


def test_degenerate_line_has_no_plane(dialog: TrimDialog) -> None:
    dialog.set_line(0, (50.0, 50.0), (50.0, 50.0))
    assert dialog.plane is None
    assert not dialog.trim_button.isEnabled()


def test_changing_up_axis_rerenders(dialog: TrimDialog) -> None:
    drag(dialog, 3, (10.0, 100.0), (150.0, 80.0))
    dialog.up_axis.setCurrentText("+X")
    assert dialog.source_view == 0
    assert np.allclose(dialog.views[0].camera.up, [1, 0, 0])


def test_no_other_model_disables_copy(app: QApplication, lumpy_mesh: trimesh.Trimesh) -> None:
    dlg = TrimDialog(lumpy_mesh, "model.stl", view_size=SIZE)
    assert not dlg.copy_box.isEnabled()
    assert not dlg.copy_to_other
    dlg.close()
