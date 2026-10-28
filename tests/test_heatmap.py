import os
from pathlib import Path

import numpy as np
import pytest
import trimesh
from conftest import rigid_transform

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from mesh_compare.comparison import compare_meshes  # noqa: E402
from mesh_compare.heatmap import (  # noqa: E402
    MISSING_COLOUR,
    WITHIN_COLOUR,
    HeatMap,
    build_heatmap,
    default_missing_threshold,
    deviation_colours,
)
from mesh_compare.heatmap_dialog import HeatMapDialog, heatmap_image  # noqa: E402

SIZE = 120


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


@pytest.fixture(scope="module")
def topless_scan(lumpy_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """The lumpy mesh with everything above z = 6 missing, like an incomplete scan."""
    scan = lumpy_mesh.submesh([lumpy_mesh.triangles_center[:, 2] < 6.0], append=True)
    assert isinstance(scan, trimesh.Trimesh)
    return scan


@pytest.fixture(scope="module")
def complete(lumpy_mesh: trimesh.Trimesh) -> HeatMap:
    return build_heatmap(lumpy_mesh, lumpy_mesh, None, 0.1, size=SIZE)


@pytest.fixture(scope="module")
def incomplete(lumpy_mesh: trimesh.Trimesh, topless_scan: trimesh.Trimesh) -> HeatMap:
    return build_heatmap(lumpy_mesh, topless_scan, None, 0.1, size=SIZE)


def test_colours_by_deviation() -> None:
    colours = deviation_colours(np.array([0.0, -0.05, 0.4, -0.4, 2.0, -2.0]), 0.1, 0.5)
    assert colours.shape == (6, 3) and colours.dtype == np.uint8
    np.testing.assert_array_equal(colours[0], WITHIN_COLOUR)
    np.testing.assert_array_equal(colours[1], WITHIN_COLOUR)
    larger, smaller = colours[2].astype(int), colours[3].astype(int)
    assert larger[0] > larger[2]  # scan larger: towards red
    assert smaller[2] > smaller[0]  # scan smaller: towards blue
    np.testing.assert_array_equal(colours[4], MISSING_COLOUR)
    np.testing.assert_array_equal(colours[5], MISSING_COLOUR)


def test_threshold_must_exceed_tolerance() -> None:
    with pytest.raises(ValueError, match="larger than"):
        deviation_colours(np.zeros(3), 0.5, 0.5)


def test_default_missing_threshold() -> None:
    assert default_missing_threshold(0.05) == 0.5
    assert default_missing_threshold(0.2) == pytest.approx(1.0)


def test_identical_meshes_have_nothing_missing(complete: HeatMap) -> None:
    assert [view.name for view in complete.views] == ["Front", "Right", "Back", "Left"]
    assert complete.missing_pct(0.5) == 0.0
    for view in complete.views:
        visible = view.deviations[view.view.mask]
        assert visible.size > 0 and np.max(np.abs(visible)) < 0.1
        assert np.all(np.isnan(view.deviations[~view.view.mask]))


def test_missing_area_is_flagged_where_it_is(incomplete: HeatMap) -> None:
    assert incomplete.missing_pct(0.5) > 5.0
    for view in incomplete.views:
        mask = view.view.mask
        missing = mask & (np.abs(np.nan_to_num(view.deviations)) > 0.5)
        heights = view.view.positions[..., 2]
        assert np.all(heights[missing] > 5.0)
        assert not np.any(missing[mask & (heights < 4.0)])


def test_rgba_matches_mask(incomplete: HeatMap) -> None:
    pixels = incomplete.rgba(0, 0.5)
    mask = incomplete.views[0].view.mask
    assert pixels.shape == (SIZE, SIZE, 4)
    np.testing.assert_array_equal(pixels[..., 3] == 255, mask)
    top = mask & (incomplete.views[0].view.positions[..., 2] > 8.0)
    # Missing pixels keep the magenta hue under shading: red and blue well above green.
    assert np.all(pixels[top, 0] > pixels[top, 1]) and np.all(pixels[top, 2] > pixels[top, 1])


def test_transform_from_comparison_is_used(
    lumpy_mesh: trimesh.Trimesh, rigid_scan: trimesh.Trimesh
) -> None:
    result = compare_meshes(lumpy_mesh, rigid_scan, samples=2_000)
    aligned = build_heatmap(lumpy_mesh, rigid_scan, result.scan_transform, 0.1, size=SIZE)
    unaligned = build_heatmap(lumpy_mesh, rigid_scan, None, 0.1, size=SIZE)
    assert aligned.missing_pct(0.5) < 1.0
    assert unaligned.missing_pct(0.5) > 50.0


@pytest.fixture(scope="module")
def rigid_scan(lumpy_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    moved = lumpy_mesh.copy()
    moved.apply_transform(rigid_transform((0.3, -0.2, 0.5), (5.0, -3.0, 2.0)))
    return moved


def test_with_up_axis_rerenders(incomplete: HeatMap) -> None:
    turned = incomplete.with_up_axis("+X")
    assert turned.up_axis == "+X"
    assert turned.views[0].view.camera.size == SIZE
    assert turned.missing_pct(0.5) > 5.0


def test_heatmap_image_size(app: QApplication, incomplete: HeatMap) -> None:
    image = heatmap_image(incomplete, 0.5, "a vs b")
    assert image.width() == 600  # small views: widened so the legend fits
    assert image.height() > 2 * SIZE


def test_dialog_saves_image(
    app: QApplication, incomplete: HeatMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = HeatMapDialog(incomplete, "a vs b", tmp_path / "scan_heatmap.png", 0.5)
    chosen = tmp_path / "picked"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(chosen), ""))
    dialog.save_button.click()
    saved = tmp_path / "picked.png"
    assert saved.exists()
    loaded = QImage(str(saved))
    assert loaded.width() == dialog.image.width()
    assert dialog.status.text() == "Saved picked.png"

    jpeg = tmp_path / "out.jpg"
    assert dialog.save_image(jpeg) and jpeg.exists()


def test_dialog_cancel_save_does_nothing(
    app: QApplication, incomplete: HeatMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = HeatMapDialog(incomplete, "a vs b", tmp_path / "scan_heatmap.png", 0.5)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))
    assert not dialog.save_image()
    assert list(tmp_path.iterdir()) == []


def test_dialog_threshold_and_up_axis(app: QApplication, incomplete: HeatMap) -> None:
    dialog = HeatMapDialog(incomplete, "a vs b", Path("unused.png"), 0.5)
    before = dialog.image.copy()
    dialog.threshold.setValue(3.0)
    assert dialog.image != before

    dialog.up_axis.setCurrentText("+Y")
    worker = dialog._worker
    assert worker is not None and not dialog.up_axis.isEnabled()
    worker.wait()
    while dialog._worker is not None:
        app.processEvents()
    assert dialog.heatmap.up_axis == "+Y"
    assert dialog.up_axis.isEnabled()

    dialog.up_axis.setCurrentText("+Z")  # cached: no new render
    assert dialog._worker is None
    assert dialog.heatmap is incomplete
