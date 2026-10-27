from pathlib import Path

import numpy as np
import pytest
import trimesh
from conftest import rigid_transform

from mesh_compare.comparison import (
    DeviationStats,
    compare_meshes,
    load_mesh,
    signed_distances,
)


class TestDeviationStats:
    def test_known_values(self) -> None:
        stats = DeviationStats.from_distances(np.array([0.1, -0.1, 0.3]), tolerance=0.1)
        assert stats.rms == pytest.approx(np.sqrt(0.11 / 3))
        assert stats.mean_signed == pytest.approx(0.1)
        assert stats.mean_abs == pytest.approx(0.5 / 3)
        assert stats.max_abs == pytest.approx(0.3)
        assert stats.within_tolerance_pct == pytest.approx(200 / 3)

    def test_rms_matches_mean_and_std(self) -> None:
        d = np.random.default_rng(1).normal(0.05, 0.2, 1000)
        stats = DeviationStats.from_distances(d, tolerance=0.1)
        assert stats.rms == pytest.approx(np.hypot(stats.mean_signed, stats.std))

    @pytest.mark.parametrize(
        ("distances", "tolerance"), [(np.array([]), 0.1), (np.array([0.1]), -1.0)]
    )
    def test_invalid_input(self, distances: np.ndarray, tolerance: float) -> None:
        with pytest.raises(ValueError):
            DeviationStats.from_distances(distances, tolerance)


class TestSignedDistances:
    def test_outside_positive_inside_negative(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=5, radius=10.0)
        d = signed_distances(np.array([[0.0, 0.0, 12.0], [0.0, 0.0, 7.0]]), sphere)
        assert d[0] == pytest.approx(2.0, abs=0.01)
        assert d[1] == pytest.approx(-3.0, abs=0.01)


class TestCompareMeshes:
    def test_identical_meshes(self, lumpy_mesh: trimesh.Trimesh) -> None:
        result = compare_meshes(lumpy_mesh, lumpy_mesh.copy(), samples=5_000)
        for stats in (result.scan_to_original, result.original_to_scan):
            assert stats.rms < 1e-3
            assert stats.within_tolerance_pct == 100.0
        assert result.warnings == []

    def test_aligns_moved_scan(self, lumpy_mesh: trimesh.Trimesh) -> None:
        scan = lumpy_mesh.copy()
        scan.apply_transform(rigid_transform((2.0, -0.7, 1.3), (40.0, -15.0, 8.0)))
        result = compare_meshes(lumpy_mesh, scan, samples=5_000)
        assert result.scan_to_original.rms < 0.005
        assert result.original_to_scan.rms < 0.005

    def test_without_alignment_moved_scan_is_far(self, lumpy_mesh: trimesh.Trimesh) -> None:
        scan = lumpy_mesh.copy()
        scan.apply_translation((5.0, 0.0, 0.0))
        result = compare_meshes(lumpy_mesh, scan, samples=2_000, align=False)
        assert result.alignment_rmse is None
        assert result.scan_to_original.rms > 1.0

    def test_inflated_scan_has_positive_bias(self, lumpy_mesh: trimesh.Trimesh) -> None:
        scan = lumpy_mesh.copy()
        scan.vertices = scan.vertices + 0.1 * scan.vertex_normals
        result = compare_meshes(lumpy_mesh, scan, samples=5_000, align=False)
        for stats in (result.scan_to_original, result.original_to_scan):
            assert stats.mean_signed == pytest.approx(0.1, abs=0.01)

    def test_untrimmed_material_only_hurts_scan_to_original(
        self, lumpy_mesh: trimesh.Trimesh
    ) -> None:
        stand = trimesh.creation.box(extents=(6.0, 6.0, 2.0))
        stand.apply_translation((0.0, 0.0, lumpy_mesh.bounds[0, 2] - 1.5))
        scan = trimesh.util.concatenate([lumpy_mesh, stand])
        assert isinstance(scan, trimesh.Trimesh)
        scan.apply_transform(rigid_transform((0.3, 0.2, -0.4), (3.0, 1.0, -2.0)))

        result = compare_meshes(lumpy_mesh, scan, samples=10_000)
        assert result.original_to_scan.rms < 0.005
        assert result.scan_to_original.max_abs > 1.0

    def test_scale_mismatch_warns(self, lumpy_mesh: trimesh.Trimesh) -> None:
        scan = lumpy_mesh.copy()
        scan.apply_scale(10.0)
        result = compare_meshes(lumpy_mesh, scan, samples=1_000, align=False)
        assert any("units" in warning for warning in result.warnings)

    def test_invalid_sample_count(self, lumpy_mesh: trimesh.Trimesh) -> None:
        with pytest.raises(ValueError):
            compare_meshes(lumpy_mesh, lumpy_mesh, samples=0)

    def test_text_export(self, lumpy_mesh: trimesh.Trimesh) -> None:
        text = compare_meshes(lumpy_mesh, lumpy_mesh, samples=1_000, tolerance=0.2).as_text()
        assert text.startswith("Metric\tScan → Original\tOriginal → Scan")
        assert "Within ±0.2 mm\t100.0 %\t100.0 %" in text
        assert "Alignment RMSE" in text


class TestLoadMesh:
    @pytest.mark.parametrize("suffix", [".stl", ".ply", ".obj"])
    def test_round_trip(self, tmp_path: Path, lumpy_mesh: trimesh.Trimesh, suffix: str) -> None:
        path = tmp_path / f"model{suffix}"
        lumpy_mesh.export(path)
        loaded = load_mesh(path)
        assert len(loaded.faces) == len(lumpy_mesh.faces)

    def test_inside_out_mesh_is_fixed(self, tmp_path: Path, lumpy_mesh: trimesh.Trimesh) -> None:
        flipped = lumpy_mesh.copy()
        flipped.invert()
        path = tmp_path / "flipped.stl"
        flipped.export(path)
        assert load_mesh(path).volume > 0

    def test_unsupported_suffix(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            load_mesh(tmp_path / "model.txt")

    def test_point_cloud_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "cloud.ply"
        trimesh.PointCloud(np.random.default_rng(0).random((50, 3))).export(path)
        with pytest.raises(ValueError, match="no triangle mesh"):
            load_mesh(path)
