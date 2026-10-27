from pathlib import Path

import numpy as np
import pytest
import trimesh
from conftest import rigid_transform

from mesh_compare.comparison import compare_meshes, load_mesh
from mesh_compare.trimming import (
    Camera,
    SurfaceSamples,
    TrimPlane,
    apply_cuts,
    copy_cuts,
    default_line,
    kept_by_all,
    line_on_view,
    plane_from_line,
    render_view,
    side_cameras,
    trim_files,
    trimmed_path,
)

RADIUS = 10.0


@pytest.fixture(scope="module")
def sphere() -> trimesh.Trimesh:
    mesh: trimesh.Trimesh = trimesh.creation.icosphere(subdivisions=5, radius=RADIUS)
    return mesh


@pytest.fixture(scope="module")
def cameras(sphere: trimesh.Trimesh) -> list[Camera]:
    return side_cameras(sphere, "+Z", size=200)


def pixel_row(camera: Camera, height: float) -> float:
    return camera.size / 2 - height * camera.size / (2 * camera.half_size)


class TestCameras:
    def test_four_orthonormal_views_around_up(self, cameras: list[Camera]) -> None:
        assert len(cameras) == 4
        for camera in cameras:
            basis = np.stack([camera.right, camera.up, camera.toward_viewer])
            assert np.allclose(basis @ basis.T, np.eye(3))
            assert np.allclose(np.cross(camera.right, camera.up), camera.toward_viewer)
            assert np.allclose(camera.up, [0, 0, 1])
        assert np.allclose(cameras[0].toward_viewer, -cameras[2].toward_viewer)

    @pytest.mark.parametrize("axis", ["+X", "-Y", "-Z"])
    def test_other_up_axes(self, sphere: trimesh.Trimesh, axis: str) -> None:
        for camera in side_cameras(sphere, axis, size=50):
            assert abs(camera.toward_viewer @ camera.up) < 1e-12

    def test_pixel_world_round_trip(self, cameras: list[Camera]) -> None:
        camera = cameras[1]
        world = camera.to_world(37.0, 151.0)
        x, y, depth = camera.to_pixels(world[None, :])
        assert (x[0], y[0], depth[0]) == pytest.approx((37.0, 151.0, 0.0))


class TestTrimPlane:
    def test_apply_keeps_normal_side(self, sphere: trimesh.Trimesh) -> None:
        plane = TrimPlane(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 1.0]))
        kept = plane.apply(sphere)
        assert kept.bounds[0, 2] == pytest.approx(5.0)
        assert kept.bounds[1, 2] == pytest.approx(RADIUS)

    def test_removing_everything_fails(self, sphere: trimesh.Trimesh) -> None:
        plane = TrimPlane(np.array([0.0, 0.0, 50.0]), np.array([0.0, 0.0, 1.0]))
        with pytest.raises(ValueError, match="whole model"):
            plane.apply(sphere)

    def test_transformed_plane_follows_mesh(self, lumpy_mesh: trimesh.Trimesh) -> None:
        plane = TrimPlane(np.array([1.0, 2.0, -3.0]), np.array([0.3, -0.2, 0.93]))
        plane = TrimPlane(plane.origin, plane.normal / np.linalg.norm(plane.normal))
        transform = rigid_transform((0.4, 1.1, -0.6), (12.0, -4.0, 7.0))

        expected = plane.apply(lumpy_mesh)
        expected.apply_transform(transform)
        moved = lumpy_mesh.copy()
        moved.apply_transform(transform)
        actual = plane.transformed(transform).apply(moved)
        assert np.allclose(actual.bounds, expected.bounds, atol=1e-6)


class TestSeveralCuts:
    def test_two_cuts_keep_the_band_between(self, sphere: trimesh.Trimesh) -> None:
        below = TrimPlane(np.array([0.0, 0.0, -4.0]), np.array([0.0, 0.0, 1.0]))
        above = TrimPlane(np.array([0.0, 0.0, 6.0]), np.array([0.0, 0.0, -1.0]))
        band = apply_cuts([below, above], sphere)
        assert band.bounds[:, 2] == pytest.approx([-4.0, 6.0])
        points = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 7.0]])
        assert kept_by_all([below, above], points).tolist() == [False, True, False]

    def test_no_cuts_keeps_everything(self, sphere: trimesh.Trimesh) -> None:
        assert kept_by_all([], np.zeros((3, 3))).all()
        assert apply_cuts([], sphere) is sphere

    def test_cuts_that_remove_everything_fail(self, sphere: trimesh.Trimesh) -> None:
        below = TrimPlane(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0]))
        above = TrimPlane(np.array([0.0, 0.0, -2.0]), np.array([0.0, 0.0, -1.0]))
        with pytest.raises(ValueError, match="whole model"):
            apply_cuts([below, above], sphere)

    @pytest.mark.parametrize("height", [0.1, 0.5, 0.9])
    def test_default_line_height(
        self, sphere: trimesh.Trimesh, cameras: list[Camera], height: float
    ) -> None:
        (_, y), _ = default_line(cameras[0], sphere, height)
        assert y == pytest.approx(pixel_row(cameras[0], -RADIUS + 2 * RADIUS * height), abs=0.5)


class TestPlaneFromLine:
    @pytest.mark.parametrize(("remove", "kept_sign"), [("bottom", 1.0), ("top", -1.0)])
    def test_horizontal_line(self, cameras: list[Camera], remove: str, kept_sign: float) -> None:
        row = pixel_row(cameras[0], 2.0)
        plane = plane_from_line(cameras[0], (20.0, row), (180.0, row), remove)  # type: ignore[arg-type]
        assert np.allclose(plane.normal, [0, 0, kept_sign])
        assert plane.origin[2] == pytest.approx(2.0)

    def test_line_direction_does_not_matter(self, cameras: list[Camera]) -> None:
        a, b = (20.0, 120.0), (180.0, 90.0)
        forward = plane_from_line(cameras[2], a, b, "bottom")
        backward = plane_from_line(cameras[2], b, a, "bottom")
        assert np.allclose(forward.normal, backward.normal)

    def test_vertical_line_still_has_a_side(self, cameras: list[Camera]) -> None:
        plane = plane_from_line(cameras[0], (100.0, 20.0), (100.0, 180.0), "top")
        assert abs(plane.normal @ cameras[0].up) < 1e-9

    def test_zero_length_line(self, cameras: list[Camera]) -> None:
        with pytest.raises(ValueError, match="no length"):
            plane_from_line(cameras[0], (50.0, 50.0), (50.0, 50.0), "top")


class TestLineOnView:
    def test_tilted_line_is_edge_on_only_front_and_back(self, cameras: list[Camera]) -> None:
        plane = plane_from_line(cameras[0], (20.0, 140.0), (180.0, 100.0), "bottom")
        shown = [line_on_view(plane, camera) is not None for camera in cameras]
        assert shown == [True, False, True, False]

    def test_line_passes_through_drawn_points(self, cameras: list[Camera]) -> None:
        a, b = (20.0, 140.0), (180.0, 100.0)
        edge_on = line_on_view(plane_from_line(cameras[0], a, b, "bottom"), cameras[0])
        assert edge_on is not None
        point, direction = edge_on
        to_b = np.array(b) - point
        assert abs(to_b[0] * direction[1] - to_b[1] * direction[0]) < 1e-6

    def test_horizontal_line_is_level_in_every_view(self, cameras: list[Camera]) -> None:
        row = pixel_row(cameras[0], -3.0)
        plane = plane_from_line(cameras[0], (20.0, row), (180.0, row), "bottom")
        for camera in cameras:
            edge_on = line_on_view(plane, camera)
            assert edge_on is not None
            assert edge_on[0][1] == pytest.approx(row)
            assert abs(edge_on[1][1]) < 1e-9


class TestRendering:
    def test_sphere_silhouette_and_positions(
        self, sphere: trimesh.Trimesh, cameras: list[Camera]
    ) -> None:
        view = render_view(SurfaceSamples.from_mesh(sphere, cameras[0]), cameras[0])
        radius_pixels = RADIUS * cameras[0].size / (2 * cameras[0].half_size)
        assert view.mask.sum() == pytest.approx(np.pi * radius_pixels**2, rel=0.03)
        distances = np.linalg.norm(view.positions[view.mask], axis=1)
        assert np.allclose(distances, RADIUS, atol=0.05)
        # Only the near hemisphere is visible.
        assert np.all(view.positions[view.mask] @ cameras[0].toward_viewer > -0.5)

    def test_removed_pixels_are_below_the_cut(
        self, sphere: trimesh.Trimesh, cameras: list[Camera]
    ) -> None:
        view = render_view(SurfaceSamples.from_mesh(sphere, cameras[1]), cameras[1])
        plane = TrimPlane(np.zeros(3), np.array([0.0, 0.0, 1.0]))
        rows = np.nonzero(view.removed([plane]))[0]
        assert np.min(rows) >= cameras[1].size / 2 - 1

    def test_default_line_removes_tenth_of_sphere(
        self, sphere: trimesh.Trimesh, cameras: list[Camera]
    ) -> None:
        # A sphere's surface area is spread evenly over its height (Archimedes).
        start, end = default_line(cameras[0], sphere)
        plane = plane_from_line(cameras[0], start, end, "bottom")
        samples = SurfaceSamples.from_mesh(sphere, cameras[0])
        assert samples.removed_fraction([plane]) == pytest.approx(0.1, abs=0.01)


class TestCopyAndSave:
    def test_copied_cut_matches_on_moved_copy(self, lumpy_mesh: trimesh.Trimesh) -> None:
        cameras = side_cameras(lumpy_mesh, "+Z", size=200)
        plane = plane_from_line(cameras[0], (10.0, 130.0), (190.0, 110.0), "bottom")
        other = lumpy_mesh.copy()
        other.apply_transform(rigid_transform((0.5, -1.0, 2.0), (30.0, 5.0, -10.0)))

        trimmed = plane.apply(lumpy_mesh)
        other_trimmed = apply_cuts(copy_cuts([plane], lumpy_mesh, other), other)
        result = compare_meshes(trimmed, other_trimmed, samples=5_000)
        assert result.scan_to_original.rms < 0.005
        assert other_trimmed.area == pytest.approx(trimmed.area, rel=1e-3)

    def test_trimmed_path(self) -> None:
        assert trimmed_path(Path("/data/ear L.ply")) == Path("/data/ear L_trimmed.stl")

    def test_trim_files(self, tmp_path: Path, lumpy_mesh: trimesh.Trimesh) -> None:
        model, other = tmp_path / "original.stl", tmp_path / "scan.obj"
        lumpy_mesh.export(model)
        lumpy_mesh.export(other)
        plane = TrimPlane(np.array([0.0, 0.0, -2.0]), np.array([0.0, 0.0, 1.0]))

        saved = trim_files([plane], model, other)
        assert saved == [tmp_path / "original_trimmed.stl", tmp_path / "scan_trimmed.stl"]
        for path in saved:
            assert load_mesh(path).bounds[0, 2] == pytest.approx(-2.0, abs=0.01)
        assert trim_files([plane], model) == saved[:1]
