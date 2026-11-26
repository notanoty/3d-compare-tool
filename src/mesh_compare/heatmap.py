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
# Deviations beyond this count as missing (raised just above the tolerance if needed).
DEFAULT_MISSING_MM = 0.5
# Darkest shading applied on top of the colours, so the shape stays readable.
MIN_BRIGHTNESS = 0.5

# Detail heat map: one ramp from "matches the original" to "far off", stretched over
# the deviations that actually occur instead of the fixed ±tolerance band.
DETAIL_STOPS = np.array(
    [
        [21, 128, 61],  # dark green: closest
        [34, 197, 94],  # green
        [163, 230, 53],  # lime
        [250, 204, 21],  # yellow
        [249, 115, 22],  # orange
        [220, 38, 38],  # red: furthest (not missing)
    ],
    dtype=np.float64,
)
# The ramp ends at this percentile of the non-missing deviations, so a few
# stray points do not squash everything else into the green.
DETAIL_HIGH_PERCENTILE = 99.0
# How far up the ramp the closest point can start (0.6 = yellow) when even the
# best match is at the missing threshold.
DETAIL_MAX_START = 0.6


def default_missing_threshold(tolerance: float) -> float:
    """Deviation above which a point of the original counts as missing in the scan."""
    return max(DEFAULT_MISSING_MM, round(tolerance + 0.01, 2))


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
class DetailRange:
    """Colour range of the detail heat map.

    Attributes:
        low: Smallest |deviation| (mm) seen on the original: the best-matching point.
        high: |deviation| (mm) at the red end of the ramp.
        start: Ramp position (0-1) of ``low``. Near 0 (dark green) when the best
            point is close to the original, up to ``DETAIL_MAX_START`` (yellow)
            when even the best point is at the missing threshold.
    """

    low: float
    high: float
    start: float


def detail_range(distances: FloatArray, missing_threshold: float) -> DetailRange:
    """Colour range for the detail heat map of these deviations.

    The range runs from the smallest |deviation| to the ``DETAIL_HIGH_PERCENTILE``
    of the deviations that are not missing (|deviation| <= ``missing_threshold``).
    """
    absolute = np.abs(distances[np.isfinite(distances)])
    kept = absolute[absolute <= missing_threshold]
    if kept.size == 0:  # everything is missing
        return DetailRange(missing_threshold, missing_threshold, DETAIL_MAX_START)
    low = float(kept.min())
    high = float(np.percentile(kept, DETAIL_HIGH_PERCENTILE))
    if high <= low:
        high = min(low + 1e-3, missing_threshold) if low < missing_threshold else low + 1e-3
    start = float(np.clip(low / missing_threshold, 0.0, 1.0)) * DETAIL_MAX_START
    return DetailRange(low, high, start)


def detail_colours(
    distances: FloatArray, colour_range: DetailRange, missing_threshold: float
) -> NDArray[np.uint8]:
    """Map deviations to RGB colours relative to the best-matching point.

    The closest point gets the ramp colour at ``colour_range.start`` (green when it
    is close to the original, yellow when it is far off); larger deviations run up
    the ramp towards red at ``colour_range.high``. Unlike ``deviation_colours`` there
    is no flat green band, so differences inside the tolerance stay visible. The
    colour shows the size of the deviation, not its sign. Deviations beyond
    ``missing_threshold`` are magenta, as in the standard heat map.

    Returns:
        An ``(N, 3)`` array of 0-255 colours.
    """
    absolute = np.abs(distances)
    span = colour_range.high - colour_range.low
    fraction = np.clip((absolute - colour_range.low) / span, 0.0, 1.0) if span > 0 else 0.0
    t = colour_range.start + (1.0 - colour_range.start) * fraction
    colours = _ramp(np.broadcast_to(t, absolute.shape).astype(np.float64), DETAIL_STOPS)
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

    def rgba(
        self, index: int, missing_threshold: float, detail: bool = False
    ) -> NDArray[np.uint8]:
        """Shaded RGBA image of view ``index``; the background is transparent.

        With ``detail`` the colours come from ``detail_colours`` over the range of
        all four views (``detail_range``), so the views share one scale.
        """
        item = self.views[index]
        mask = item.view.mask
        size = mask.shape[0]
        pixels = np.zeros((size, size, 4), dtype=np.uint8)
        if detail:
            colours = detail_colours(
                item.deviations[mask], self.detail_range(missing_threshold), missing_threshold
            )
        else:
            colours = deviation_colours(item.deviations[mask], self.tolerance, missing_threshold)
        shade = item.view.shade[mask].astype(np.float64) / 255.0
        brightness = MIN_BRIGHTNESS + (1.0 - MIN_BRIGHTNESS) * shade
        pixels[mask, :3] = np.round(colours * brightness[:, None]).astype(np.uint8)
        pixels[mask, 3] = 255
        return pixels

    def detail_range(self, missing_threshold: float) -> DetailRange:
        """Detail colour range over the visible pixels of all views together."""
        visible = np.concatenate([item.deviations[item.view.mask] for item in self.views])
        return detail_range(visible, missing_threshold)

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
