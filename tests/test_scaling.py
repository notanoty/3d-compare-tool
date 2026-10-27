from pathlib import Path

import numpy as np
import pytest
import trimesh
from conftest import rigid_transform

from mesh_compare.comparison import estimate_scale, load_mesh
from mesh_compare.scaling import grid_step, scale_file, scale_mesh, scaled_path


def test_scaled_path() -> None:
    assert scaled_path(Path("/scans/10_27.ply")) == Path("/scans/10_27_scaled.stl")


def test_scale_mesh_scales_about_origin(lumpy_mesh: trimesh.Trimesh) -> None:
    scaled = scale_mesh(lumpy_mesh, 2.5)
    assert np.allclose(scaled.vertices, lumpy_mesh.vertices * 2.5)
    assert scaled.area == pytest.approx(lumpy_mesh.area * 2.5**2)


@pytest.mark.parametrize("factor", [0.0, -1.0, float("nan"), float("inf")])
def test_scale_mesh_rejects_bad_factor(lumpy_mesh: trimesh.Trimesh, factor: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        scale_mesh(lumpy_mesh, factor)


def test_scale_file(tmp_path: Path, lumpy_mesh: trimesh.Trimesh) -> None:
    source = tmp_path / "scan.stl"
    lumpy_mesh.export(source)
    saved = scale_file(source, 0.001)
    assert saved == tmp_path / "scan_scaled.stl"
    assert np.allclose(load_mesh(saved).extents, lumpy_mesh.extents * 0.001, rtol=1e-5)


@pytest.mark.parametrize(
    ("span", "step"), [(38.6, 5.0), (100.0, 10.0), (0.23, 0.05), (12.0, 2.0), (1.0, 0.1)]
)
def test_grid_step(span: float, step: float) -> None:
    assert grid_step(span) == pytest.approx(step)


def test_grid_step_rejects_empty_span() -> None:
    with pytest.raises(ValueError):
        grid_step(0.0)


@pytest.mark.parametrize("true_scale", [160.4, 0.001, 1.0])
def test_estimate_scale_recovers_known_factor(
    lumpy_mesh: trimesh.Trimesh, true_scale: float
) -> None:
    scan = scale_mesh(lumpy_mesh, 1 / true_scale)
    scan.apply_transform(rigid_transform((1.0, 2.0, 0.3), (0.1, 0.0, 0.2)))
    assert estimate_scale(scan, lumpy_mesh) == pytest.approx(true_scale, rel=0.005)


def test_estimate_scale_with_partial_scan(lumpy_mesh: trimesh.Trimesh) -> None:
    """A scan missing a fifth of the surface still gets the right scale."""
    keep = lumpy_mesh.triangles_center[:, 2] > np.percentile(lumpy_mesh.triangles_center[:, 2], 20)
    partial = trimesh.Trimesh(lumpy_mesh.vertices, lumpy_mesh.faces[keep])
    scan = scale_mesh(partial, 1 / 50.0)
    assert estimate_scale(scan, lumpy_mesh) == pytest.approx(50.0, rel=0.002)
