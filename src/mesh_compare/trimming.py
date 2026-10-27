"""Plane trimming: four side-view renders, cut planes drawn as lines, and the cuts themselves.

The views are simple orthographic renders made by splatting dense surface samples
into a z-buffer. Every pixel remembers the 3D point it shows, so the part a cut
would remove can be re-highlighted instantly while the user drags the line.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import ndimage

from mesh_compare.comparison import FloatArray, align_scan, load_mesh

RemoveSide = Literal["top", "bottom"]

VIEW_NAMES = ("Front", "Right", "Back", "Left")
UP_AXES: dict[str, tuple[float, float, float]] = {
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
}
# Samples per pixel of visible surface; enough that splats leave only tiny gaps.
SPLAT_DENSITY = 3.0
MAX_SAMPLES = 2_000_000
# Gaps up to about twice this many pixels wide inside the silhouette are filled in.
MAX_GAP_PIXELS = 2
# Samples this many pixel-widths behind their neighbours count as hidden surface.
BACKFACE_DEPTH_PIXELS = 4.0
# A sample is hidden if this share of its SPECKLE_WINDOW x SPECKLE_WINDOW neighbours
# is clearly nearer. Edge pixels have background on one side, so they stay.
SPECKLE_WINDOW = 5
SPECKLE_SURROUNDED = 0.6


@dataclass(frozen=True)
class TrimPlane:
    """A cutting plane. ``normal`` points toward the part that is kept."""

    origin: FloatArray
    normal: FloatArray

    def keeps(self, points: FloatArray) -> NDArray[np.bool_]:
        """True for points on the kept side of the plane."""
        return np.asarray((points - self.origin) @ self.normal >= 0)

    def transformed(self, transform: FloatArray) -> TrimPlane:
        """The same plane after moving space by a rigid 4x4 ``transform``."""
        rotation = transform[:3, :3]
        return TrimPlane(
            origin=rotation @ self.origin + transform[:3, 3],
            normal=rotation @ self.normal,
        )

    def apply(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Cut ``mesh`` and return the kept part. The cut is left open (not capped).

        Raises:
            ValueError: If the plane removes the whole model.
        """
        vertices, faces, _ = trimesh.intersections.slice_faces_plane(  # type: ignore[no-untyped-call]
            mesh.vertices, mesh.faces, plane_normal=self.normal, plane_origin=self.origin
        )
        if len(faces) == 0:
            raise ValueError("This cut removes the whole model")
        return trimesh.Trimesh(vertices, faces)


def kept_by_all(planes: Sequence[TrimPlane], points: FloatArray) -> NDArray[np.bool_]:
    """True for points that no plane cuts away."""
    kept = np.ones(len(points), dtype=bool)
    for plane in planes:
        kept &= plane.keeps(points)
    return kept


def apply_cuts(planes: Sequence[TrimPlane], mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Cut ``mesh`` with every plane in turn and return what is left.

    Raises:
        ValueError: If the cuts together remove the whole model.
    """
    for plane in planes:
        mesh = plane.apply(mesh)
    return mesh


@dataclass(frozen=True)
class Camera:
    """Orthographic side view. Pixel (0, 0) is the top-left corner of the image.

    Attributes:
        toward_viewer: Unit vector from the model toward the camera.
        right: Screen right direction in world space.
        up: Screen up direction in world space (same for all four views).
        center: World point shown in the middle of the image.
        half_size: World distance from the image centre to its edge.
        size: Image width and height in pixels.
    """

    toward_viewer: FloatArray
    right: FloatArray
    up: FloatArray
    center: FloatArray
    half_size: float
    size: int

    def to_pixels(self, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Project world points to ``(x, y, depth)``; larger depth is nearer the camera."""
        offset = points - self.center
        scale = self.size / (2 * self.half_size)
        x = (offset @ self.right) * scale + self.size / 2
        y = self.size / 2 - (offset @ self.up) * scale
        return x, y, offset @ self.toward_viewer

    def to_world(self, x: float, y: float) -> FloatArray:
        """World point under pixel ``(x, y)`` on the plane through ``center``."""
        scale = 2 * self.half_size / self.size
        return np.asarray(
            self.center
            + (x - self.size / 2) * scale * self.right
            + (self.size / 2 - y) * scale * self.up
        )


def side_cameras(mesh: trimesh.Trimesh, up_axis: str = "+Z", size: int = 360) -> list[Camera]:
    """Front, right, back and left cameras around ``up_axis``, sharing one scale.

    Sharing the centre and scale means a height on one image is the same height
    on every other image.
    """
    up = np.array(UP_AXES[up_axis])
    reference = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    front = np.cross(up, reference)
    front /= np.linalg.norm(front)
    center = np.asarray(mesh.bounds.mean(axis=0), dtype=np.float64)
    half_size = float(np.max(np.linalg.norm(mesh.vertices - center, axis=1))) * 1.05

    cameras = []
    toward = front
    for _ in VIEW_NAMES:
        cameras.append(Camera(toward, np.cross(up, toward), up, center, half_size, size))
        toward = np.cross(up, toward)
    return cameras


@dataclass(frozen=True)
class RenderedView:
    """A rendered side view.

    Attributes:
        camera: Camera the view was rendered with.
        shade: Grey level 0-255 per pixel (background is 0; see ``mask``).
        mask: True where the model is visible.
        positions: World point shown at each pixel (only meaningful where ``mask``).
    """

    camera: Camera
    shade: NDArray[np.uint8]
    mask: NDArray[np.bool_]
    positions: FloatArray

    def removed(self, planes: Sequence[TrimPlane]) -> NDArray[np.bool_]:
        """Pixels showing surface that any of ``planes`` would cut away."""
        removed = np.zeros_like(self.mask)
        removed[self.mask] = ~kept_by_all(planes, self.positions[self.mask])
        return removed


@dataclass(frozen=True)
class SurfaceSamples:
    """Dense points with normals used for rendering and estimating removed area."""

    points: FloatArray
    normals: FloatArray

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, camera: Camera, seed: int = 0) -> SurfaceSamples:
        """Sample densely enough for ``camera``'s resolution."""
        pixel = 2 * camera.half_size / camera.size
        count = int(np.clip(SPLAT_DENSITY * mesh.area / pixel**2, 50_000, MAX_SAMPLES))
        points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
        normals = np.asarray(mesh.face_normals, dtype=np.float64)[faces]
        return cls(np.asarray(points, dtype=np.float64), normals)

    def removed_fraction(self, planes: Sequence[TrimPlane]) -> float:
        """Approximate share of the surface area the planes together remove."""
        return float(np.mean(~kept_by_all(planes, self.points)))


def render_view(samples: SurfaceSamples, camera: Camera) -> RenderedView:
    """Render one view by splatting samples into a z-buffer (nearest sample wins)."""
    size = camera.size
    x, y, depth = camera.to_pixels(samples.points)
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    inside = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
    point_ids = np.nonzero(inside)[0]
    flat = iy[inside] * size + ix[inside]

    # Sort by pixel, then nearest first; the first entry of each pixel wins.
    order = np.lexsort((-depth[inside], flat))
    pixels, first = np.unique(flat[order], return_index=True)
    winner_flat = np.full(size * size, -1)
    winner_flat[pixels] = point_ids[order[first]]
    winner = winner_flat.reshape(size, size)

    # A far sample can win a pixel that no near sample happened to hit, letting the
    # back of the model show through as speckles. Such a pixel is surrounded by
    # nearer surface on (almost) all sides; silhouette pixels are not, so they stay.
    depth_image = np.full((size, size), -np.inf)
    depth_image[winner >= 0] = depth[winner[winner >= 0]]
    reach = SPECKLE_WINDOW // 2
    padded = np.pad(depth_image, reach, constant_values=-np.inf)
    threshold = BACKFACE_DEPTH_PIXELS * 2 * camera.half_size / size
    offsets = range(-reach, reach + 1)
    nearer_neighbours = sum(
        padded[reach + dy : reach + dy + size, reach + dx : reach + dx + size]
        > depth_image + threshold
        for dy in offsets
        for dx in offsets
        if dy or dx
    )
    neighbours = SPECKLE_WINDOW**2 - 1
    winner[np.asarray(nearer_neighbours) >= SPECKLE_SURROUNDED * neighbours] = -1

    # Fill small gaps between splats with the nearest visible pixel. Closing the
    # mask finds gaps inside the silhouette without growing its outline.
    empty = winner < 0
    gaps = empty & ndimage.binary_closing(
        ~empty, structure=np.ones((3, 3)), iterations=MAX_GAP_PIXELS, border_value=0
    )
    _, (near_y, near_x) = ndimage.distance_transform_edt(empty, return_indices=True)
    winner[gaps] = winner[near_y[gaps], near_x[gaps]]

    mask = winner >= 0
    light = camera.toward_viewer * 0.8 + camera.up * 0.4 + camera.right * 0.2
    light /= np.linalg.norm(light)
    brightness = np.zeros((size, size))
    # abs(): scans often have inconsistent normals; light both faces the same.
    brightness[mask] = 0.25 + 0.75 * np.abs(samples.normals[winner[mask]] @ light)
    positions = np.zeros((size, size, 3))
    positions[mask] = samples.points[winner[mask]]
    return RenderedView(camera, (brightness * 255).astype(np.uint8), mask, positions)


def plane_from_line(
    camera: Camera,
    start: tuple[float, float],
    end: tuple[float, float],
    remove: RemoveSide,
) -> TrimPlane:
    """Cutting plane through a line drawn on one view, extending straight back into it.

    Args:
        camera: The view the line was drawn on.
        start: Line start in pixels.
        end: Line end in pixels.
        remove: Which side of the line is cut away, relative to the view's up direction.

    Raises:
        ValueError: If the line has no length.
    """
    a, b = camera.to_world(*start), camera.to_world(*end)
    normal = np.cross(b - a, camera.toward_viewer)
    length = np.linalg.norm(normal)
    if length < 1e-12:
        raise ValueError("Trim line has no length")
    normal /= length
    points_up = normal @ camera.up > 0 or (
        abs(normal @ camera.up) < 1e-9 and normal @ camera.right > 0
    )
    if points_up == (remove == "top"):
        normal = -normal
    return TrimPlane(origin=a, normal=normal)


def line_on_view(plane: TrimPlane, camera: Camera) -> tuple[FloatArray, FloatArray] | None:
    """Where ``plane`` appears as a line on ``camera``'s image, as ``(point, direction)``.

    Returns ``None`` when the plane is not edge-on to this view (it then shows up
    as the boundary of the highlighted region rather than as a line).
    """
    if abs(plane.normal @ camera.toward_viewer) > 1e-6:
        return None
    x, y, _ = camera.to_pixels(plane.origin[None, :])
    direction = np.array([-(plane.normal @ camera.up), -(plane.normal @ camera.right)])
    return np.array([x[0], y[0]]), direction / np.linalg.norm(direction)


def default_line(
    camera: Camera, mesh: trimesh.Trimesh, height: float = 0.1
) -> tuple[tuple[float, float], tuple[float, float]]:
    """A level line across the image at ``height`` (0 = bottom, 1 = top) of the model."""
    heights = np.asarray(mesh.vertices - camera.center) @ camera.up
    low, high = float(np.min(heights)), float(np.max(heights))
    level = low + height * (high - low)
    y = camera.size / 2 - level * camera.size / (2 * camera.half_size)
    margin = camera.size * 0.08
    return (margin, float(y)), (camera.size - margin, float(y))


def trimmed_path(source: Path) -> Path:
    """Where the trimmed copy of ``source`` is saved: next to it, ``<name>_trimmed.stl``."""
    return source.with_name(f"{source.stem}_trimmed.stl")


def copy_cuts(
    planes: Sequence[TrimPlane], model: trimesh.Trimesh, other: trimesh.Trimesh
) -> list[TrimPlane]:
    """Express ``planes`` (drawn on ``model``) in ``other``'s coordinates.

    ``other`` is aligned onto ``model``; the returned planes then cut ``other`` at
    the same places on the object, so both trimmed models end in the same spots.
    """
    transform, _ = align_scan(other, model)
    inverse = np.linalg.inv(transform)
    return [plane.transformed(inverse) for plane in planes]


def trim_files(
    planes: Sequence[TrimPlane], model_path: Path, other_path: Path | None = None
) -> list[Path]:
    """Trim the model (and optionally the other model with the same cuts) and save them.

    Args:
        planes: Cuts in ``model_path``'s coordinates.
        model_path: The model the cut was drawn on.
        other_path: Another model to cut at the same places, after aligning it.

    Returns:
        Paths of the saved trimmed files, model first.
    """
    model = load_mesh(model_path)
    targets = [(model_path, apply_cuts(planes, model))]
    if other_path is not None:
        other = load_mesh(other_path)
        targets.append((other_path, apply_cuts(copy_cuts(planes, model, other), other)))

    saved = []
    for source, trimmed in targets:
        destination = trimmed_path(source)
        trimmed.export(destination)
        saved.append(destination)
    return saved
