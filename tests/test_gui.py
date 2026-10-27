import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mesh_compare.comparison import compare_meshes  # noqa: E402
from mesh_compare.gui import CompareWorker, MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


@pytest.fixture
def window(app: QApplication) -> Iterator[MainWindow]:
    win = MainWindow()
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


def test_worker_reports_success_and_failure(mesh_files: tuple[Path, Path], tmp_path: Path) -> None:
    results: list[object] = []
    errors: list[str] = []

    good = CompareWorker(*mesh_files, tolerance=0.1, align=False)
    good.succeeded.connect(results.append)
    good.run()

    bad = CompareWorker(tmp_path / "missing.stl", mesh_files[1], tolerance=0.1, align=False)
    bad.failed.connect(errors.append)
    bad.run()

    assert len(results) == 1
    assert len(errors) == 1


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
