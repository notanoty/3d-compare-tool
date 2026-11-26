"""The window workflow without windows: preview and apply cuts, scale, compare, save images.

Used by the MCP server (``mcp_server``) and the stand-alone trimming window
(``manual_trim``). Cuts are described as plain dictionaries ("cut specs") so they
can travel as JSON:

    {"view": "Front", "remove": "bottom", "line": [[x1, y1], [x2, y2]]}
    {"view": "Front", "remove": "top", "height": 0.85, "tilt_deg": 10}

``line`` is in pixels on that view (``VIEW_SIZE`` square, origin top-left), the
same pixels the trimming window uses, so cuts carry over between the two.
``height`` is a level line at that fraction of the model's height (0 = bottom,
1 = top), optionally tilted counter-clockwise by ``tilt_deg`` degrees.

Images are drawn with Qt, so a ``QGuiApplication`` must exist first; call
``ensure_qt()`` (it uses the offscreen platform when it has to create one).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen

from mesh_compare.comparison import (
    SUPPORTED_SUFFIXES,
    ComparisonResult,
    align_scan,
    compare_meshes,
    estimate_scale,
    format_size,
    load_mesh,
)
from mesh_compare.heatmap import build_heatmap, default_missing_threshold
from mesh_compare.heatmap_dialog import DETAIL_SUFFIX, heatmap_image
from mesh_compare.scaling import scale_mesh, scaled_path
from mesh_compare.trim_dialog import CUT_COLOURS, CutLine, view_image
from mesh_compare.trimming import (
    UP_AXES,
    VIEW_NAMES,
    Camera,
    RenderedView,
    SurfaceSamples,
    TrimPlane,
    apply_cuts,
    copy_cuts,
    default_line,
    line_on_view,
    render_view,
    side_cameras,
    trimmed_path,
)

# Same size as the trimming window's views, so pixel lines mean the same in both.
VIEW_SIZE = 360
CUT_COLOUR_NAMES = ("blue", "green", "purple", "orange", "teal")
# Cuts that leave less than this share of the surface are refused.
MAX_REMOVED_FRACTION = 0.999

GRID_STEP = 30
GRID_LABEL_STEP = 60
RULER_WIDTH = 40
MARGIN = 16
TITLE_HEIGHT = 26
LABEL_HEIGHT = 20
FOOTER_HEIGHT = 22
HANDLE_RADIUS = 5.0

# Scale check colours: only the original, only the scan.
ORIGINAL_ONLY = np.array([249, 115, 22], dtype=np.float64)
SCAN_ONLY = np.array([37, 99, 235], dtype=np.float64)
BACKGROUND = np.array([245, 245, 245], dtype=np.float64)

_app: object | None = None


def ensure_qt() -> None:
    """Create a ``QApplication`` on the offscreen platform if none exists yet."""
    global _app
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        _app = QApplication(["mesh-compare", "-platform", "offscreen"])


def png_bytes(image: QImage) -> bytes:
    """``image`` encoded as PNG."""
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(data.data())


def save_image(image: QImage, path: Path) -> Path:
    """Save ``image`` (PNG or JPG by suffix).

    Raises:
        OSError: If the file could not be written.
    """
    if not image.save(str(path)):
        raise OSError(f"Could not save image {path}")
    return path


def check_writable(paths: Sequence[Path], overwrite: bool) -> None:
    """Refuse to replace existing files unless ``overwrite`` is set.

    Raises:
        FileExistsError: If any path exists and ``overwrite`` is false.
    """
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Already exists (pass overwrite=true to replace): " + ", ".join(existing)
        )


def _named(folder: Path, name: str | None, default: Path, suffix: str = ".stl") -> Path:
    if not name:
        return folder / default.name
    path = folder / name
    return path if path.suffix else path.with_suffix(suffix)


def list_models(folder: Path, recursive: bool = False) -> list[Path]:
    """Model files in ``folder`` (and below it when ``recursive``), sorted."""
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def describe_model(path: Path) -> str:
    """Short plain-text summary of a model file."""
    mesh = load_mesh(path)
    lines = [
        f"{path}",
        f"Size: {format_size(tuple(float(v) for v in mesh.extents))}",  # type: ignore[arg-type]
        f"Triangles: {len(mesh.faces):,}  Vertices: {len(mesh.vertices):,}",
        f"Surface area: {mesh.area:.1f} mm²",
        f"Closed (watertight): {'yes' if mesh.is_watertight else 'no'}",
    ]
    if mesh.is_watertight:
        lines.append(f"Volume: {mesh.volume:.1f} mm³")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- cuts


def level_line(
    camera: Camera, mesh: trimesh.Trimesh, height: float, tilt_deg: float = 0.0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """A line across the view at ``height`` of the model, tilted counter-clockwise."""
    (left, y), (right, _) = default_line(camera, mesh, height)
    half = (right - left) / 2
    centre = camera.size / 2
    angle = math.radians(tilt_deg)
    dx, dy = half * math.cos(angle), -half * math.sin(angle)
    return (centre - dx, y - dy), (centre + dx, y + dy)


def parse_cut(spec: Mapping[str, Any], cameras: Sequence[Camera], mesh: trimesh.Trimesh) -> CutLine:
    """Turn a cut spec (see module docstring) into a ``CutLine``.

    Raises:
        ValueError: If the spec is incomplete or malformed.
    """
    view_name = str(spec.get("view", "Front")).strip().capitalize()
    if view_name not in VIEW_NAMES:
        raise ValueError(f"Unknown view '{spec.get('view')}'. Use one of: {', '.join(VIEW_NAMES)}")
    view = VIEW_NAMES.index(view_name)
    remove = spec.get("remove")
    if remove not in ("top", "bottom"):
        raise ValueError('Each cut needs "remove": "top" or "bottom"')

    if spec.get("line") is not None:
        try:
            (x1, y1), (x2, y2) = spec["line"]
            start, end = (float(x1), float(y1)), (float(x2), float(y2))
        except (TypeError, ValueError) as error:
            raise ValueError('"line" must look like [[x1, y1], [x2, y2]]') from error
    elif spec.get("height") is not None:
        start, end = level_line(
            cameras[view], mesh, float(spec["height"]), float(spec.get("tilt_deg") or 0.0)
        )
    else:
        raise ValueError('Each cut needs either "line" or "height"')
    return CutLine(view, start, end, remove)


def cut_to_spec(cut: CutLine) -> dict[str, Any]:
    """The cut spec for ``cut`` (pixel line form, rounded to 0.1 px)."""
    return {
        "view": VIEW_NAMES[cut.view],
        "remove": cut.remove,
        "line": [[round(v, 1) for v in cut.start], [round(v, 1) for v in cut.end]],
    }


@dataclass(frozen=True)
class TrimViews:
    """A model rendered from the four sides, ready for cut previews."""

    mesh: trimesh.Trimesh
    up_axis: str
    views: list[RenderedView]
    samples: SurfaceSamples

    @classmethod
    def render(cls, mesh: trimesh.Trimesh, up_axis: str = "+Z", size: int = VIEW_SIZE) -> TrimViews:
        """Render ``mesh`` the way the trimming window does.

        Raises:
            ValueError: If ``up_axis`` is unknown.
        """
        if up_axis not in UP_AXES:
            raise ValueError(f"Unknown up axis '{up_axis}'. Use one of: {', '.join(UP_AXES)}")
        cameras = side_cameras(mesh, up_axis, size)
        samples = SurfaceSamples.from_mesh(mesh, cameras[0])
        return cls(mesh, up_axis, [render_view(samples, camera) for camera in cameras], samples)

    @property
    def cameras(self) -> list[Camera]:
        return [view.camera for view in self.views]

    @property
    def size(self) -> int:
        return self.views[0].camera.size

    @property
    def mm_per_pixel(self) -> float:
        camera = self.views[0].camera
        return 2 * camera.half_size / camera.size

    def cuts(self, specs: Sequence[Mapping[str, Any]]) -> list[CutLine]:
        return [parse_cut(spec, self.cameras, self.mesh) for spec in specs]

    def planes(self, cuts: Sequence[CutLine]) -> list[TrimPlane]:
        """Cutting planes for ``cuts``.

        Raises:
            ValueError: If a line is too short to define a plane.
        """
        planes = []
        for number, cut in enumerate(cuts, start=1):
            plane = cut.plane(self.cameras)
            if plane is None:
                raise ValueError(f"Cut {number}'s line is too short")
            planes.append(plane)
        return planes

    def height_y(self, height: float) -> float:
        """Pixel row of ``height`` (0 = bottom, 1 = top of the model) on every view."""
        return default_line(self.views[0].camera, self.mesh, height)[0][1]


def trim_preview_image(
    views: TrimViews,
    planes: Sequence[TrimPlane],
    cuts: Sequence[CutLine] = (),
    title: str = "",
    footer: str = "",
) -> QImage:
    """The four views with pixel grid, height ruler, cut lines and removed area in red."""
    size = views.size
    cell_width = size + RULER_WIDTH
    width = 2 * cell_width + 3 * MARGIN
    height = MARGIN + TITLE_HEIGHT + 2 * (LABEL_HEIGHT + size) + MARGIN + FOOTER_HEIGHT + MARGIN
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    bold = QFont(painter.font())
    bold.setBold(True)
    bold.setPointSize(11)
    small = QFont(painter.font())
    small.setPointSize(8)
    normal = QFont(painter.font())
    normal.setPointSize(10)

    painter.setFont(bold)
    painter.setPen(QColor("#111827"))
    painter.drawText(
        QRectF(MARGIN, MARGIN, width - 2 * MARGIN, TITLE_HEIGHT),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        title,
    )

    ruler = [(tenth / 10, views.height_y(tenth / 10)) for tenth in range(11)]
    for index, view in enumerate(views.views):
        x0 = MARGIN + (index % 2) * (cell_width + MARGIN)
        label_top = MARGIN + TITLE_HEIGHT + (index // 2) * (LABEL_HEIGHT + size)
        y0 = label_top + LABEL_HEIGHT
        painter.setFont(normal)
        painter.setPen(QColor("#374151"))
        painter.drawText(
            QRectF(x0, label_top, size, LABEL_HEIGHT), Qt.AlignmentFlag.AlignCenter, VIEW_NAMES[index]
        )
        painter.drawImage(QPointF(x0, y0), view_image(view, planes))

        painter.save()
        painter.translate(x0, y0)
        painter.setClipRect(QRectF(0, 0, size, size))
        painter.setPen(QPen(QColor(0, 0, 0, 30), 1))
        for p in range(GRID_STEP, size, GRID_STEP):
            painter.drawLine(QPointF(p, 0), QPointF(p, size))
            painter.drawLine(QPointF(0, p), QPointF(size, p))
        painter.setFont(small)
        painter.setPen(QColor(90, 90, 90))
        for p in range(GRID_LABEL_STEP, size, GRID_LABEL_STEP):
            painter.drawText(QPointF(p + 2, 10), str(p))
            painter.drawText(QPointF(2, p - 2), str(p))
        for number, plane in enumerate(planes):
            edge_on = line_on_view(plane, view.camera)
            if edge_on is None:
                continue
            point, direction = edge_on
            reach = 2.0 * size
            colour = CUT_COLOURS[number % len(CUT_COLOURS)]
            painter.setPen(QPen(colour, 2, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(*(point - reach * direction)), QPointF(*(point + reach * direction)))
        painter.setFont(normal)
        for number, cut in enumerate(cuts):
            if cut.view != index:
                continue
            colour = CUT_COLOURS[number % len(CUT_COLOURS)]
            painter.setPen(QPen(colour, 2))
            painter.setBrush(QColor("white"))
            for x, y in (cut.start, cut.end):
                painter.drawEllipse(QPointF(x, y), HANDLE_RADIUS, HANDLE_RADIUS)
            painter.drawText(QPointF(cut.start[0] + 8, cut.start[1] - 6), str(number + 1))
        painter.restore()

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor("#d1d5db"))
        painter.drawRect(QRectF(x0, y0, size, size))
        painter.setFont(small)
        painter.setPen(QColor("#374151"))
        for value, y in ruler:
            if 0 <= y <= size:
                painter.drawLine(QPointF(x0 + size + 2, y0 + y), QPointF(x0 + size + 8, y0 + y))
                painter.drawText(QPointF(x0 + size + 10, y0 + y + 3), f"{value:.1f}")

    painter.setFont(normal)
    painter.setPen(QColor("#111827"))
    painter.drawText(
        QRectF(MARGIN, height - MARGIN - FOOTER_HEIGHT, width - 2 * MARGIN, FOOTER_HEIGHT),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        footer,
    )
    painter.end()
    return image


def describe_cuts(cuts: Sequence[CutLine]) -> list[str]:
    lines = []
    for number, cut in enumerate(cuts, start=1):
        colour = CUT_COLOUR_NAMES[(number - 1) % len(CUT_COLOUR_NAMES)]
        spec = cut_to_spec(cut)
        lines.append(
            f"Cut {number} ({colour}): remove {cut.remove}, drawn on {spec['view']}, "
            f"line {spec['line']}"
        )
    return lines


@dataclass(frozen=True)
class TrimPreview:
    text: str
    image: QImage
    removed_fraction: float


def preview_trim(
    model_path: Path, specs: Sequence[Mapping[str, Any]], up_axis: str = "+Z"
) -> TrimPreview:
    """Show where ``specs`` would cut ``model_path``, without saving anything."""
    mesh = load_mesh(model_path)
    views = TrimViews.render(mesh, up_axis)
    cuts = views.cuts(specs)
    planes = views.planes(cuts)
    removed = views.samples.removed_fraction(planes) if planes else 0.0
    footer = (
        f"Red = removed, about {removed * 100:.1f} % of the surface"
        if planes
        else "No cuts yet. Grid: pixels on each view. Ruler: fraction of model height."
    )
    image = trim_preview_image(
        views, planes, cuts, f"{model_path.name}  (up axis {up_axis})", footer
    )
    lines = [
        f"Model: {model_path.name}, size {format_size(tuple(float(v) for v in mesh.extents))}"  # type: ignore[arg-type]
        f", up axis {up_axis}",
        f"Views are {views.size}×{views.size} px, 1 px = {views.mm_per_pixel:.4f} mm. Grid every "
        f"{GRID_STEP} px (labels every {GRID_LABEL_STEP}); the ruler right of each view is the "
        "fraction of model height used by height cuts.",
        *describe_cuts(cuts),
    ]
    if planes:
        lines.append(f"Removes about {removed * 100:.1f} % of the surface.")
        if removed >= MAX_REMOVED_FRACTION:
            lines.append("Warning: this removes almost everything.")
    return TrimPreview("\n".join(lines), image, removed)


@dataclass(frozen=True)
class TrimOutcome:
    models: list[Path]
    images: list[Path]
    removed_fraction: float


def trim_and_save(
    views: TrimViews,
    cuts: Sequence[CutLine],
    model_path: Path,
    other_path: Path | None = None,
    output_dir: Path | None = None,
    output_name: str | None = None,
    other_output_name: str | None = None,
    overwrite: bool = False,
    save_images: bool = True,
) -> TrimOutcome:
    """Cut the model shown in ``views`` (and optionally another model at the same places).

    Saves ``<name>_trimmed.stl`` (or ``output_name``) and, with ``save_images``,
    ``<trimmed name>_cuts.png`` showing what was removed.

    Raises:
        ValueError: If there are no cuts or they remove (almost) everything.
        FileExistsError: If an output exists and ``overwrite`` is false.
    """
    if not cuts:
        raise ValueError("Give at least one cut")
    planes = views.planes(cuts)
    removed = views.samples.removed_fraction(planes)
    if removed >= MAX_REMOVED_FRACTION:
        raise ValueError("These cuts remove (almost) the whole model")

    model_dest = _named(output_dir or model_path.parent, output_name, trimmed_path(model_path))
    jobs: list[tuple[Path, trimesh.Trimesh, Path, Sequence[TrimPlane], TrimViews | None]] = [
        (model_path, views.mesh, model_dest, planes, views)
    ]
    if other_path is not None:
        other = load_mesh(other_path)
        other_dest = _named(
            output_dir or other_path.parent, other_output_name, trimmed_path(other_path)
        )
        jobs.append((other_path, other, other_dest, copy_cuts(planes, views.mesh, other), None))
    if len({dest for _, _, dest, _, _ in jobs}) < len(jobs):
        raise ValueError("Both trimmed models would get the same file name; set other_output_name")

    image_paths = [dest.with_name(f"{dest.stem}_cuts.png") for _, _, dest, _, _ in jobs]
    check_writable(
        [dest for _, _, dest, _, _ in jobs] + (image_paths if save_images else []), overwrite
    )

    saved_models, saved_images = [], []
    for (source, mesh, dest, job_planes, job_views), image_path in zip(
        jobs, image_paths, strict=True
    ):
        trimmed = apply_cuts(job_planes, mesh)
        dest.parent.mkdir(parents=True, exist_ok=True)
        trimmed.export(dest)
        saved_models.append(dest)
        if save_images:
            shown = job_views or TrimViews.render(mesh, views.up_axis)
            job_removed = shown.samples.removed_fraction(job_planes)
            picture = trim_preview_image(
                shown,
                job_planes,
                cuts if job_views is not None else (),
                f"{source.name} → {dest.name}  (up axis {views.up_axis})",
                f"Red = removed, about {job_removed * 100:.1f} % of the surface",
            )
            saved_images.append(save_image(picture, image_path))
    return TrimOutcome(saved_models, saved_images, removed)


def model_views_image(path: Path, up_axis: str = "+Z", title: str | None = None) -> QImage:
    """Plain four-view render of a model file (e.g. to check a trimmed result)."""
    mesh = load_mesh(path)
    views = TrimViews.render(mesh, up_axis)
    footer = f"Size {format_size(tuple(float(v) for v in mesh.extents))}"  # type: ignore[arg-type]
    return trim_preview_image(views, [], (), title or path.name, footer)


# ----------------------------------------------------------------------------- scale


def scale_check_image(
    original: trimesh.Trimesh, scan: trimesh.Trimesh, up_axis: str, title: str
) -> QImage:
    """Both models from four sides at one mm-per-pixel: grey where they overlap,
    orange where only the original is, blue where only the scan is."""
    framing = trimesh.util.concatenate([original, scan])
    cameras = side_cameras(framing, up_axis, VIEW_SIZE)
    original_samples = SurfaceSamples.from_mesh(original, cameras[0])
    scan_samples = SurfaceSamples.from_mesh(scan, cameras[0])

    views = []
    for camera in cameras:
        a = render_view(original_samples, camera)
        b = render_view(scan_samples, camera)
        shade = np.where(a.mask, a.shade, b.shade).astype(np.float64)[..., None] / 255.0
        rgb = np.repeat(shade * 255.0, 3, axis=2)
        rgb[a.mask & ~b.mask] = ORIGINAL_ONLY * (0.55 + 0.45 * shade[a.mask & ~b.mask])
        rgb[b.mask & ~a.mask] = SCAN_ONLY * (0.55 + 0.45 * shade[b.mask & ~a.mask])
        rgb[~a.mask & ~b.mask] = BACKGROUND
        views.append(np.ascontiguousarray(rgb.astype(np.uint8)))

    size = VIEW_SIZE
    width = 2 * size + 3 * MARGIN
    height = MARGIN + TITLE_HEIGHT + 2 * (LABEL_HEIGHT + size) + MARGIN + FOOTER_HEIGHT + MARGIN
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    font = QFont(painter.font())
    font.setPointSize(10)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("#111827"))
    painter.drawText(
        QRectF(MARGIN, MARGIN, width - 2 * MARGIN, TITLE_HEIGHT),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        title,
    )
    font.setBold(False)
    painter.setFont(font)
    for index, pixels in enumerate(views):
        x0 = MARGIN + (index % 2) * (size + MARGIN)
        label_top = MARGIN + TITLE_HEIGHT + (index // 2) * (LABEL_HEIGHT + size)
        painter.setPen(QColor("#374151"))
        painter.drawText(
            QRectF(x0, label_top, size, LABEL_HEIGHT), Qt.AlignmentFlag.AlignCenter, VIEW_NAMES[index]
        )
        picture = QImage(pixels.data, size, size, 3 * size, QImage.Format.Format_RGB888).copy()
        painter.drawImage(QPointF(x0, label_top + LABEL_HEIGHT), picture)
    painter.setPen(QColor("#111827"))
    painter.drawText(
        QRectF(MARGIN, height - MARGIN - FOOTER_HEIGHT, width - 2 * MARGIN, FOOTER_HEIGHT),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        "Grey = both overlap · orange = only the original · blue = only the scaled scan",
    )
    painter.end()
    return image


@dataclass(frozen=True)
class ScaleOutcome:
    factor: float
    model: Path
    image: QImage | None
    image_path: Path | None
    alignment_rmse: float | None
    scaled_size: tuple[float, float, float]
    original_size: tuple[float, float, float] | None


def scale_and_save(
    scan_path: Path,
    factor: float,
    original_path: Path | None = None,
    output_dir: Path | None = None,
    output_name: str | None = None,
    overwrite: bool = False,
    save_images: bool = True,
    up_axis: str = "+Z",
) -> ScaleOutcome:
    """Scale the scan by ``factor`` and save it; with an original, also make a check image
    (``<scaled name>_scale-check.png``) of the scaled scan aligned onto the original.

    Raises:
        ValueError: If ``factor`` is not positive.
        FileExistsError: If an output exists and ``overwrite`` is false.
    """
    scaled = scale_mesh(load_mesh(scan_path), factor)
    dest = _named(output_dir or scan_path.parent, output_name, scaled_path(scan_path))
    image_path = dest.with_name(f"{dest.stem}_scale-check.png")
    want_image = original_path is not None
    check_writable([dest] + ([image_path] if want_image and save_images else []), overwrite)

    dest.parent.mkdir(parents=True, exist_ok=True)
    scaled.export(dest)
    size = tuple(float(v) for v in scaled.extents)

    image, saved_image, rmse, original_size = None, None, None, None
    if original_path is not None:
        original = load_mesh(original_path)
        original_size = tuple(float(v) for v in original.extents)
        transform, rmse = align_scan(scaled, original)
        placed = scaled.copy()
        placed.apply_transform(transform)
        image = scale_check_image(
            original,
            placed,
            up_axis,
            f"{dest.name} (× {factor:.6g})  vs  {original_path.name}",
        )
        if save_images:
            saved_image = save_image(image, image_path)
    return ScaleOutcome(
        factor,
        dest,
        image,
        saved_image,
        rmse,
        (size[0], size[1], size[2]),
        None if original_size is None else (original_size[0], original_size[1], original_size[2]),
    )


def auto_scale(
    scan_path: Path,
    original_path: Path,
    output_dir: Path | None = None,
    output_name: str | None = None,
    overwrite: bool = False,
    save_images: bool = True,
    up_axis: str = "+Z",
) -> ScaleOutcome:
    """Estimate the factor that makes the scan the original's size (as Auto-fit does),
    then scale and save like ``scale_and_save``."""
    factor = estimate_scale(load_mesh(scan_path), load_mesh(original_path))
    return scale_and_save(
        scan_path, factor, original_path, output_dir, output_name, overwrite, save_images, up_axis
    )


# ----------------------------------------------------------------------------- compare


@dataclass(frozen=True)
class CompareOutcome:
    result: ComparisonResult
    image: QImage
    results_path: Path | None
    heatmap_path: Path | None
    missing_pct: float
    missing_threshold: float
    detail_image: QImage
    detail_heatmap_path: Path | None


def compare_and_save(
    original_path: Path,
    scan_path: Path,
    tolerance: float = 0.1,
    output_dir: Path | None = None,
    results_name: str | None = "data.txt",
    heatmap_name: str | None = "heatmap.png",
    up_axis: str = "+Z",
    missing_threshold: float | None = None,
    align: bool = True,
    overwrite: bool = False,
) -> CompareOutcome:
    """Compare as the main window does, save the results table and the heat map images.

    Files go into ``output_dir`` (the scan's folder by default). Pass ``None`` as a
    name to skip saving that file. Two heat maps are saved: the standard one
    (``heatmap_name``) and the detail one next to it (``<heatmap stem>_detail.png``),
    coloured relative to the best-matching point.

    Raises:
        FileExistsError: If an output exists and ``overwrite`` is false.
        ValueError: If ``up_axis`` is unknown or the threshold is not above the tolerance.
    """
    if up_axis not in UP_AXES:
        raise ValueError(f"Unknown up axis '{up_axis}'. Use one of: {', '.join(UP_AXES)}")
    threshold = missing_threshold or default_missing_threshold(tolerance)
    if threshold <= tolerance:
        raise ValueError("missing_threshold must be larger than the tolerance")
    folder = output_dir or scan_path.parent
    results_path = folder / results_name if results_name else None
    heatmap_path = _named(folder, heatmap_name, Path("heatmap.png"), ".png") if heatmap_name else None
    detail_path = (
        heatmap_path.with_name(heatmap_path.stem + DETAIL_SUFFIX + heatmap_path.suffix)
        if heatmap_path is not None
        else None
    )
    check_writable(
        [p for p in (results_path, heatmap_path, detail_path) if p is not None], overwrite
    )

    original, scan = load_mesh(original_path), load_mesh(scan_path)
    result = compare_meshes(original, scan, tolerance=tolerance, align=align)
    heatmap = build_heatmap(original, scan, result.scan_transform, tolerance, up_axis)
    title = f"{original_path.name}  vs  {scan_path.name}  (tolerance ±{tolerance:g} mm)"
    image = heatmap_image(heatmap, threshold, title)
    detail_image = heatmap_image(heatmap, threshold, f"{original_path.name}  vs  {scan_path.name}  (detail)", detail=True)

    folder.mkdir(parents=True, exist_ok=True)
    if results_path is not None:
        results_path.write_text(result.as_text(), encoding="utf-8")
    if heatmap_path is not None:
        save_image(image, heatmap_path)
    if detail_path is not None:
        save_image(detail_image, detail_path)
    return CompareOutcome(
        result,
        image,
        results_path,
        heatmap_path,
        heatmap.missing_pct(threshold),
        threshold,
        detail_image,
        detail_path,
    )
