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


def removed_percent(dialog: TrimDialog) -> float:
    return float(dialog.info.text().split("about ")[1].split("%")[0])


def test_opens_with_one_bottom_cut(dialog: TrimDialog) -> None:
    planes = dialog.planes
    assert planes is not None and len(planes) == 1
    assert dialog.cuts[0].view == 0 and dialog.cuts[0].remove == "bottom"
    assert dialog.remove_bottom.isChecked()
    assert dialog.cut_list.count() == 1
    assert not dialog.delete_button.isEnabled()
    assert "Removes about" in dialog.info.text()
    assert dialog.trim_button.isEnabled()
    assert dialog.copy_box.isEnabled()
    assert not dialog.copy_to_other  # copying the cuts is opt-in
    for canvas in dialog.canvases:
        assert not canvas.image.isNull()
        assert len(canvas.lines) == 1  # a level cut is edge-on in every side view
    assert len(dialog.canvases[0].handles) == 2
    assert dialog.canvases[1].handles == []
    dialog.canvases[0].grab()  # exercises painting


def test_add_cut_removes_top_too(dialog: TrimDialog) -> None:
    before = removed_percent(dialog)
    dialog.add_button.click()
    assert len(dialog.cuts) == 2
    assert dialog.active == 1
    assert dialog.cuts[1].remove == "top"
    assert dialog.remove_top.isChecked()
    assert dialog.cut_list.currentRow() == 1
    assert dialog.delete_button.isEnabled()
    assert removed_percent(dialog) > before + 5
    assert all(len(canvas.lines) == 2 for canvas in dialog.canvases)
    planes = dialog.planes
    assert planes is not None
    assert planes[0].normal @ planes[1].normal < 0  # keep the band between them


def test_side_buttons_change_only_selected_cut(dialog: TrimDialog) -> None:
    dialog.add_cut()
    dialog.remove_bottom.click()
    assert [cut.remove for cut in dialog.cuts] == ["bottom", "bottom"]
    dialog.cut_list.setCurrentRow(0)
    assert dialog.active == 0
    dialog.remove_top.click()
    assert [cut.remove for cut in dialog.cuts] == ["top", "bottom"]
    assert "Cut 1: remove top" in dialog.cut_list.item(0).text()


def test_remove_side_flips_plane(dialog: TrimDialog) -> None:
    bottom = dialog.planes
    dialog.set_remove("top")
    top = dialog.planes
    assert bottom is not None and top is not None
    assert np.allclose(bottom[0].normal, -top[0].normal)


def test_delete_cut(dialog: TrimDialog) -> None:
    dialog.add_cut()
    dialog.add_cut()
    dialog.select_cut(1)
    third = dialog.cuts[2]
    dialog.delete_button.click()
    assert len(dialog.cuts) == 2
    assert dialog.cuts[1] is third
    assert dialog.active == 1
    dialog.delete_cut()
    dialog.delete_cut()  # the last cut stays
    assert len(dialog.cuts) == 1
    assert not dialog.delete_button.isEnabled()


def test_drawing_redraws_selected_cut_on_that_view(dialog: TrimDialog) -> None:
    dialog.add_cut()
    drag(dialog, 2, (10.0, 100.0), (150.0, 80.0))
    second = dialog.cuts[1]
    assert (second.view, second.start, second.end) == (2, (10.0, 100.0), (150.0, 80.0))
    assert dialog.cuts[0].view == 0  # the other cut is untouched
    assert "drawn on Back" in dialog.cut_list.item(1).text()
    # Cut 1 is level (a line everywhere); cut 2 is tilted on Back (a line on Front/Back only).
    assert [len(c.lines) for c in dialog.canvases] == [2, 1, 2, 1]
    assert len(dialog.canvases[2].handles) == 2


def test_grabbing_another_cuts_handle_selects_it(dialog: TrimDialog) -> None:
    dialog.add_cut()
    first = dialog.cuts[0]
    start, end = first.start, first.end
    drag(dialog, 0, start, (start[0] + 3, start[1] - 20))
    assert dialog.active == 0
    assert first.start == (start[0] + 3, start[1] - 20)
    assert first.end == end


def test_tiny_drag_keeps_existing_line(dialog: TrimDialog) -> None:
    cut = dialog.cuts[0]
    before = (cut.view, cut.start, cut.end)
    drag(dialog, 1, (80.0, 80.0), (82.0, 81.0))
    assert (cut.view, cut.start, cut.end) == before


def test_cutting_everything_disables_trim(dialog: TrimDialog) -> None:
    dialog.set_line(0, (0.0, 2.0), (SIZE, 2.0))
    assert not dialog.trim_button.isEnabled()
    assert "almost everything" in dialog.info.text()


def test_degenerate_line_has_no_planes(dialog: TrimDialog) -> None:
    dialog.add_cut()
    dialog.set_line(0, (50.0, 50.0), (50.0, 50.0))
    assert dialog.planes is None
    assert not dialog.trim_button.isEnabled()
    assert "every cut" in dialog.info.text()


def test_changing_up_axis_resets_to_one_cut(dialog: TrimDialog) -> None:
    dialog.add_cut()
    dialog.up_axis.setCurrentText("+X")
    assert len(dialog.cuts) == 1
    assert np.allclose(dialog.views[0].camera.up, [1, 0, 0])


def test_no_other_model_disables_copy(app: QApplication, lumpy_mesh: trimesh.Trimesh) -> None:
    dlg = TrimDialog(lumpy_mesh, "model.stl", view_size=SIZE)
    assert not dlg.copy_box.isEnabled()
    assert not dlg.copy_to_other
    dlg.close()
