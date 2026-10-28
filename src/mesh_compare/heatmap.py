"""Deviation heat map: the original model coloured by how far the scan is from it.

The original is rendered from the four side views. Each pixel is coloured by the
signed distance from the surface point it shows to the aligned scan, so areas the
scan missed (holes, uncaptured regions) stand out. Signs follow ``comparison``:
positive means the scan lies outside the original (too big), negative inside.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray

from mesh_compare.comparison import FloatArray, signed_distances
from mesh_compare.trimming import (
    VIEW_NAMES,
    RenderedView,
    SurfaceSamples,
    render_view,
    side_cameras,
)

WITHIN_COLOUR = np.array([34, 197, 94], dtype=np.float64)
# Colour stops from "just outside tolerance" to "at the missing threshold".
LARGER_STOPS = np.array([WITHIN_COLOUR, [250, 204, 21], [220, 38, 38]], dtype=np.float64)
SMALLER_STOPS = np.array([WITHIN_COLOUR, [34, 211, 238], [37, 99, 235]], dtype=np.float64)
MISSING_COLOUR = np.array([217, 70, 239], dtype=np.float64)
# Deviations beyond this many tolerances (but at least MIN_MISSING_MM) count as missing.
MISSING_TOLERANCES = 5.0
MIN_MISSING_MM = 0.5
# Darkest shading applied on top of the colours, so the shape stays readable.
MIN_BRIGHTNESS = 0.5


def default_missing_threshold(tolerance: float) -> float:
    """Deviation above which a point of the original counts as missing in the scan."""
    return max(MISSING_TOLERANCES * tolerance, MIN_MISSING_MM)


def _ramp(t: FloatArray, stops: FloatArray) -> FloatArray:
    positions = np.linspace(0.0, 1.0, len(stops))
    return np.stack([np.interp(t, positions, stops[:, c]) for c in range(3)], axis=1)


def deviation_colours(
    distances: FloatArray, tolerance: float, missing_threshold: float
) -> NDArray[np.uint8]:
    """Map signed deviations to RGB colours.

    Args:
        distances: Signed deviations (positive = scan larger).
        tolerance: Deviations within ±tolerance are green.
        missing_threshold: Deviations beyond this are magenta ("missing in scan").
            Between the two, colours run green → yellow → red for positive and
            green → cyan → blue for negative deviations.

    Returns:
        An ``(N, 3)`` array of 0-255 colours.

    Raises:
        ValueError: If ``missing_threshold`` is not larger than ``tolerance``.
    """
    if missing_threshold <= tolerance:
        raise ValueError(
            f"Missing threshold ({missing_threshold:g}) must be larger than "
            f"the tolerance ({tolerance:g})"
        )
    absolute = np.abs(distances)
    t = np.clip((absolute - tolerance) / (missing_threshold - tolerance), 0.0, 1.0)
    colours = np.where((distances >= 0)[:, None], _ramp(t, LARGER_STOPS), _ramp(t, SMALLER_STOPS))
    colours[absolute > missing_threshold] = MISSING_COLOUR
    return np.round(colours).astype(np.uint8)


@dataclass(frozen=True)
class HeatMapView:
    """One side view of the heat map.

    Attributes:
        name: View name, e.g. "Front".
        view: The rendered original (shading and visibility mask).
        deviations: Signed deviation per pixel; NaN where the model is not visible.
    """

    name: str
    view: RenderedView
    deviations: FloatArray


@dataclass(frozen=True)
class HeatMap:
    """The original model coloured by deviation, seen from four sides.

    Attributes:
        views: Front, right, back and left views.
        tolerance: Tolerance used for the green band.
        up_axis: Up axis the views were rendered around.
        original: The original model.
        scan: The scan, already placed onto the original.
    """

    views: list[HeatMapView]
    tolerance: float
    up_axis: str
    original: trimesh.Trimesh
    scan: trimesh.Trimesh

    def rgba(self, index: int, missing_threshold: float) -> NDArray[np.uint8]:
        """Shaded RGBA image of view ``index``; the background is transparent."""
        item = self.views[index]
        mask = item.view.mask
        size = mask.shape[0]
        pixels = np.zeros((size, size, 4), dtype=np.uint8)
        colours = deviation_colours(item.deviations[mask], self.tolerance, missing_threshold)
        shade = item.view.shade[mask].astype(np.float64) / 255.0
        brightness = MIN_BRIGHTNESS + (1.0 - MIN_BRIGHTNESS) * shade
        pixels[mask, :3] = np.round(colours * brightness[:, None]).astype(np.uint8)
        pixels[mask, 3] = 255
        return pixels

    def missing_pct(self, missing_threshold: float) -> float:
        """Percentage of visible pixels (all views together) flagged as missing."""
        visible = np.concatenate([item.deviations[item.view.mask] for item in self.views])
        if visible.size == 0:
            return 0.0
        return float(np.mean(np.abs(visible) > missing_threshold) * 100.0)

    def with_up_axis(self, up_axis: str, seed: int = 0) -> HeatMap:
        """The same heat map rendered around a different up axis."""
        size = self.views[0].view.camera.size
        return build_heatmap(self.original, self.scan, None, self.tolerance, up_axis, size, seed)


def build_heatmap(
    original: trimesh.Trimesh,
    scan: trimesh.Trimesh,
    transform: FloatArray | None,
    tolerance: float,
    up_axis: str = "+Z",
    size: int = 400,
    seed: int = 0,
) -> HeatMap:
    """Render the original from four sides, coloured by distance to the scan.

    Args:
        original: Reference model; this is the surface that gets coloured.
        scan: Scanned model. Not modified; ``transform`` is applied to a copy.
        transform: Transform placing the scan onto the original (e.g.
            ``ComparisonResult.scan_transform``), or ``None`` to use it as-is.
        tolerance: Deviations within ±tolerance are shown green.
        up_axis: Key of ``trimming.UP_AXES`` the side views turn around.
        size: Width and height of each view in pixels.
        seed: Random seed for surface sampling.

    Returns:
        The heat map with per-pixel deviations for every view.
    """
    placed = scan.copy()
    if transform is not None:
        placed.apply_transform(transform)
    cameras = side_cameras(original, up_axis, size)
    samples = SurfaceSamples.from_mesh(original, cameras[0], seed)
    views = []
    for name, camera in zip(VIEW_NAMES, cameras, strict=True):
        view = render_view(samples, camera)
        deviations = np.full(view.mask.shape, np.nan)
        # Flip sign: original outside the scan means the scan is too small.
        deviations[view.mask] = -signed_distances(view.positions[view.mask], placed)
        views.append(HeatMapView(name, view, deviations))
    return HeatMap(views, tolerance, up_axis, original, placed)
