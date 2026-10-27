import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mesh_compare.comparison import compare_meshes, load_mesh  # noqa: E402
from mesh_compare.gui import MainWindow  # noqa: E402
from mesh_compare.scale_dialog import ScaleDialog  # noqa: E402
from mesh_compare.trim_dialog import TrimDialog  # noqa: E402
from mesh_compare.workers import TaskWorker  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


@pytest.fixture
def window(app: QApplication) -> Iterator[MainWindow]:
    win = MainWindow()
    win.auto_trim.setChecked(False)
    yield win
    win.close()


@pytest.fixture
def mesh_files(tmp_path: Path, lumpy_mesh: trimesh.Trimesh) -> tuple[Path, Path]:
    original, scan = tmp_path / "original.stl", tmp_path / "scan.stl"
    lumpy_mesh.export(original)
    lumpy_mesh.export(scan)
    return original, scan


def test_compare_enabled_only_with_both_files(
    window: MainWindow, mesh_files: tuple[Path, Path]
) -> None:
    assert not window.compare_button.isEnabled()
    window.original_zone.set_path(mesh_files[0])
    assert not window.compare_button.isEnabled()
    window.scan_zone.set_path(mesh_files[1])
    assert window.compare_button.isEnabled()


def test_show_result_fills_table_and_copies(
    window: MainWindow, app: QApplication, lumpy_mesh: trimesh.Trimesh
) -> None:
    window.show_result(compare_meshes(lumpy_mesh, lumpy_mesh, samples=1_000))
    assert window.table.rowCount() == 7
    item = window.table.item(0, 0)
    assert item is not None and item.text() == "RMS deviation"
    window.copy_results()
    assert app.clipboard().text().startswith("Metric")


def test_worker_reports_success_and_failure() -> None:
    results: list[object] = []
    errors: list[str] = []

    good = TaskWorker(lambda: 42)
    good.succeeded.connect(results.append)
    good.run()

    def explode() -> None:
        raise ValueError("bad file")

    bad = TaskWorker(explode)
    bad.failed.connect(errors.append)
    bad.run()

    assert results == [42]
    assert errors == ["bad file"]


def test_show_error(window: MainWindow) -> None:
    window.show_error("boom")
    assert window.status.text() == "Error: boom"


# Qt events only borrow the mime data, so keep it alive for the whole test.
_mime_keepalive: list[QMimeData] = []


def _drop_events(paths: list[Path]) -> tuple[QDragEnterEvent, QDropEvent]:
    mime = QMimeData()
    _mime_keepalive.append(mime)
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    enter = QDragEnterEvent(
        QPoint(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    drop = QDropEvent(
        QPointF(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    return enter, drop


def test_dropping_mesh_file_selects_it(window: MainWindow, mesh_files: tuple[Path, Path]) -> None:
    enter, drop = _drop_events([mesh_files[0]])
    window.original_zone.dragEnterEvent(enter)
    assert enter.isAccepted()
    window.original_zone.dragLeaveEvent(None)
    window.original_zone.dropEvent(drop)
    assert window.original_zone.path == mesh_files[0]


@pytest.mark.parametrize("names", [["notes.txt"], ["a.stl", "b.stl"]])
def test_dropping_wrong_files_is_ignored(
    window: MainWindow, tmp_path: Path, names: list[str]
) -> None:
    enter, drop = _drop_events([tmp_path / name for name in names])
    enter.ignore()
    window.scan_zone.dragEnterEvent(enter)
    assert not enter.isAccepted()
    window.scan_zone.dropEvent(drop)
    assert window.scan_zone.path is None


def test_full_comparison_in_background(
    window: MainWindow, app: QApplication, mesh_files: tuple[Path, Path]
) -> None:
    window.original_zone.set_path(mesh_files[0])
    window.scan_zone.set_path(mesh_files[1])
    window.start_comparison()
    assert not window.compare_button.isEnabled()
    worker = window._worker
    assert worker is not None
    worker.wait()
    while window._worker is not None:
        app.processEvents()
    assert window.status.text() == "Done."
    assert window.table.rowCount() == 7
    assert window.compare_button.isEnabled()


def test_warnings_shown_in_status(window: MainWindow, lumpy_mesh: trimesh.Trimesh) -> None:
    scan = lumpy_mesh.copy()
    scan.apply_scale(10.0)
    window.show_result(compare_meshes(lumpy_mesh, scan, samples=1_000, align=False))
    assert window.status.text().startswith("⚠")


def wait_for_worker(window: MainWindow, app: QApplication) -> None:
    worker = window._worker
    assert worker is not None
    worker.wait()
    while window._worker is not None:
        app.processEvents()


@pytest.fixture
def accept_trim(monkeypatch: pytest.MonkeyPatch) -> list[TrimDialog]:
    """Make trim dialogs accept immediately with their default cut; records them."""
    opened: list[TrimDialog] = []

    def fake_exec(dialog: TrimDialog) -> int:
        opened.append(dialog)
        return int(TrimDialog.DialogCode.Accepted)

    monkeypatch.setattr(TrimDialog, "exec", fake_exec)
    return opened


def test_trim_saves_and_is_used_for_comparison(
    window: MainWindow,
    app: QApplication,
    mesh_files: tuple[Path, Path],
    accept_trim: list[TrimDialog],
) -> None:
    window.original_zone.set_path(mesh_files[0])
    assert window.trim_buttons[window.original_zone].isEnabled()
    window.trim_buttons[window.original_zone].click()
    assert len(accept_trim) == 1
    assert not accept_trim[0].copy_to_other  # no scan loaded yet
    wait_for_worker(window, app)

    trimmed = mesh_files[0].with_name("original_trimmed.stl")
    assert trimmed.exists()
    assert window.original_zone.active_path == trimmed
    assert "Saved original_trimmed.stl" in window.status.text()


def test_trim_not_copied_by_default(
    window: MainWindow,
    app: QApplication,
    mesh_files: tuple[Path, Path],
    accept_trim: list[TrimDialog],
) -> None:
    window.original_zone.set_path(mesh_files[0])
    window.scan_zone.set_path(mesh_files[1])
    window.open_trim(window.scan_zone)
    assert not accept_trim[0].copy_to_other
    wait_for_worker(window, app)

    assert window.scan_zone.active_path == mesh_files[1].with_name("scan_trimmed.stl")
    assert window.original_zone.active_path == mesh_files[0]


def test_trim_copied_to_other_model(
    window: MainWindow,
    app: QApplication,
    mesh_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def accept_with_copy(dialog: TrimDialog) -> int:
        dialog.copy_box.setChecked(True)
        return int(TrimDialog.DialogCode.Accepted)

    monkeypatch.setattr(TrimDialog, "exec", accept_with_copy)
    window.original_zone.set_path(mesh_files[0])
    window.scan_zone.set_path(mesh_files[1])
    window.open_trim(window.scan_zone)
    wait_for_worker(window, app)

    assert window.scan_zone.active_path == mesh_files[1].with_name("scan_trimmed.stl")
    assert window.original_zone.active_path == mesh_files[0].with_name("original_trimmed.stl")

    window.start_comparison()
    wait_for_worker(window, app)
    item = window.table.item(0, 1)
    assert item is not None and float(item.text().split()[0]) < 0.01


def test_skipping_trim_changes_nothing(
    window: MainWindow, mesh_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(TrimDialog, "exec", lambda self: int(TrimDialog.DialogCode.Rejected))
    window.original_zone.set_path(mesh_files[0])
    window.open_trim(window.original_zone)
    assert window._worker is None
    assert window.original_zone.active_path == mesh_files[0]


def test_new_file_clears_trimmed_version(window: MainWindow, mesh_files: tuple[Path, Path]) -> None:
    window.original_zone.set_path(mesh_files[0])
    window.original_zone.set_trimmed(mesh_files[1])
    assert "using scan.stl" in window.original_zone._label.text()
    window.original_zone.set_path(mesh_files[0])
    assert window.original_zone.active_path == mesh_files[0]


def test_unreadable_model_shows_error(window: MainWindow, tmp_path: Path) -> None:
    broken = tmp_path / "broken.stl"
    broken.write_text("not a mesh")
    window.original_zone.set_path(broken)
    window.open_trim(window.original_zone)
    assert window.status.text().startswith("Error:")


def test_auto_trim_opens_after_adding_model(
    window: MainWindow,
    app: QApplication,
    mesh_files: tuple[Path, Path],
    accept_trim: list[TrimDialog],
) -> None:
    window.auto_trim.setChecked(True)
    window.original_zone.set_path(mesh_files[0])
    assert accept_trim == []  # opens once the drop has finished
    app.processEvents()
    assert len(accept_trim) == 1
    wait_for_worker(window, app)


@pytest.mark.parametrize("compare_after", [True, False])
def test_match_scale_saves_scaled_scan(
    window: MainWindow,
    app: QApplication,
    tmp_path: Path,
    lumpy_mesh: trimesh.Trimesh,
    accept_trim: list[TrimDialog],
    monkeypatch: pytest.MonkeyPatch,
    compare_after: bool,
) -> None:
    original, scan = tmp_path / "original.stl", tmp_path / "scan.stl"
    lumpy_mesh.export(original)
    small = lumpy_mesh.copy()
    small.vertices = small.vertices / 100.0
    small.export(scan)

    def fake_exec(dialog: ScaleDialog) -> int:
        dialog.set_scale(100.0)
        dialog.compare_after = compare_after
        return int(ScaleDialog.DialogCode.Accepted)

    monkeypatch.setattr(ScaleDialog, "exec", fake_exec)
    window.original_zone.set_path(original)
    window.scan_zone.set_path(scan)
    assert window.scale_button.isEnabled()
    window.auto_trim.setChecked(True)
    window.scale_button.click()
    wait_for_worker(window, app)

    assert window.scan_zone.path == tmp_path / "scan_scaled.stl"
    app.processEvents()
    assert accept_trim == []  # scaling must not pop up the trim window
    if compare_after:  # the comparison ran as a follow-up while waiting above
        item = window.table.item(0, 1)
        assert item is not None and float(item.text().split()[0]) < 0.01
    else:
        assert window._worker is None
        assert "scaled × 100" in window.status.text()


def test_match_scale_cancel_changes_nothing(
    window: MainWindow, mesh_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ScaleDialog, "exec", lambda self: int(ScaleDialog.DialogCode.Rejected))
    window.original_zone.set_path(mesh_files[0])
    window.scan_zone.set_path(mesh_files[1])
    window.open_scale()
    assert window._worker is None
    assert window.scan_zone.path == mesh_files[1]


def test_match_scale_unreadable_file(
    window: MainWindow, tmp_path: Path, mesh_files: tuple[Path, Path]
) -> None:
    broken = tmp_path / "broken.stl"
    broken.write_text("not a mesh")
    window.original_zone.set_path(mesh_files[0])
    window.scan_zone.set_path(broken)
    window.open_scale()
    assert window.status.text().startswith("Error:")


def test_several_cuts_copied_to_other_model(
    window: MainWindow,
    app: QApplication,
    mesh_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def two_cuts(dialog: TrimDialog) -> int:
        dialog.add_cut()
        dialog.copy_box.setChecked(True)
        return int(TrimDialog.DialogCode.Accepted)

    monkeypatch.setattr(TrimDialog, "exec", two_cuts)
    window.original_zone.set_path(mesh_files[0])
    window.scan_zone.set_path(mesh_files[1])
    window.open_trim(window.original_zone)
    wait_for_worker(window, app)

    paths = [window.original_zone.trimmed_path, window.scan_zone.trimmed_path]
    assert None not in paths
    trimmed = [load_mesh(path) for path in paths if path is not None]
    full = load_mesh(mesh_files[0])
    for mesh in trimmed:
        # Both ends cut away: shorter than the full model at the bottom and the top.
        assert mesh.bounds[0, 2] > full.bounds[0, 2] + 1
        assert mesh.bounds[1, 2] < full.bounds[1, 2] - 1
    assert trimmed[1].area == pytest.approx(trimmed[0].area, rel=1e-3)
