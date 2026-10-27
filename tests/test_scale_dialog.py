import os
from collections.abc import Iterator

import pytest
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from mesh_compare.scale_dialog import ScaleDialog  # noqa: E402
from mesh_compare.scaling import scale_mesh  # noqa: E402

TRUE_SCALE = 100.0


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


@pytest.fixture
def dialog(app: QApplication, lumpy_mesh: trimesh.Trimesh) -> Iterator[ScaleDialog]:
    scan = scale_mesh(lumpy_mesh, 1 / TRUE_SCALE)
    dlg = ScaleDialog(lumpy_mesh, scan, "original.stl", "scan.stl", canvas_size=160)
    yield dlg
    dlg.close()


def button(dialog: ScaleDialog, text: str) -> QPushButton:
    matches = [b for b in dialog.findChildren(QPushButton) if b.text().startswith(text)]
    assert len(matches) == 1
    return matches[0]


def test_opens_at_scale_one_with_hint(dialog: ScaleDialog) -> None:
    assert dialog.scale == 1.0
    assert dialog.factor.value() == pytest.approx(1.0)
    assert dialog.slider.value() == 0
    assert "100× smaller" in dialog.status.text()
    for canvas in dialog.canvases:
        assert not canvas.image.isNull()
        canvas.grab()  # exercises painting


def test_scale_controls_stay_in_sync(dialog: ScaleDialog) -> None:
    dialog.factor.setValue(50.0)
    assert dialog.scale == pytest.approx(50.0)
    assert dialog.slider.value() == round(1.69897 * 1000)
    dialog.slider.setValue(2000)
    assert dialog.scale == pytest.approx(100.0)
    assert dialog.factor.value() == pytest.approx(100.0)
    assert dialog.status.text() == ""  # sizes now match, hint cleared


def test_quick_buttons(dialog: ScaleDialog) -> None:
    button(dialog, "× 10 ").click()
    assert dialog.scale == pytest.approx(10.0)
    button(dialog, "× 1000").click()
    assert dialog.scale == pytest.approx(10_000.0)
    assert "larger" in dialog.status.text()
    button(dialog, "Reset").click()
    assert dialog.scale == 1.0


def test_both_models_drawn_at_same_mm_per_pixel(dialog: ScaleDialog) -> None:
    dialog.set_scale(TRUE_SCALE)
    original, scan = dialog.canvases
    assert original.mm_per_pixel == scan.mm_per_pixel
    assert scan.image_span_mm == pytest.approx(original.image_span_mm, rel=1e-6)
    assert "Bounding box" in dialog.size_labels[1].text()


def test_auto_fit_runs_in_background(dialog: ScaleDialog, app: QApplication) -> None:
    dialog.auto_fit()
    assert not dialog.auto_button.isEnabled()
    worker = dialog._worker
    assert worker is not None
    worker.wait()
    while dialog._worker is not None:
        app.processEvents()
    assert dialog.scale == pytest.approx(TRUE_SCALE, rel=0.005)
    assert "Auto-fit suggests" in dialog.status.text()
    assert dialog.auto_button.isEnabled()


def test_changing_view_and_up_axis_rerenders(dialog: ScaleDialog) -> None:
    before = dialog.cameras[1].toward_viewer.copy()
    dialog.view.setCurrentText("Right")
    assert not (dialog.cameras[1].toward_viewer == before).all()
    dialog.up_axes[1].setCurrentText("+X")
    assert dialog.cameras[1].up.tolist() == [1.0, 0.0, 0.0]
    assert dialog.cameras[0].up.tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize(("label", "compare"), [("Apply && compare", True), ("Apply", False)])
def test_apply_buttons(dialog: ScaleDialog, label: str, compare: bool) -> None:
    matches = [b for b in dialog.findChildren(QPushButton) if b.text() == label]
    matches[0].click()
    assert dialog.result() == ScaleDialog.DialogCode.Accepted
    assert dialog.compare_after is compare
