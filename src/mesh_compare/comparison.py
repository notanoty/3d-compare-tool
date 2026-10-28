"""Load two meshes, align the scan onto the original, and summarise surface deviations.

All distances are in the units of the input files (normally millimetres).
Sign convention for signed deviations: positive means the scan surface lies
*outside* the original (scan too big), negative means *inside* (scan too small).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

FloatArray = NDArray[np.float64]

SUPPORTED_SUFFIXES = frozenset({".stl", ".ply", ".obj", ".off", ".glb", ".gltf"})

# Fraction of closest point pairs kept in each ICP step. Rejecting the worst 10%
# stops leftover untrimmed material from dragging the alignment off.
ICP_OVERLAP = 0.9
# Candidate triangles checked per query point when computing exact distances.
NEAREST_FACE_CANDIDATES = 16
# Bounding-box size ratio beyond which we warn about mismatched units/scale.
SCALE_WARNING_RATIO = 1.5


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a triangle mesh from disk.

    Args:
        path: Path to an STL, PLY, OBJ, OFF or glTF file.

    Returns:
        The loaded mesh, with normals flipped outward if it is closed and inside-out.

    Raises:
        ValueError: If the file type is unsupported or the file holds no triangles.
    """
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported file type '{path.suffix}'. Use one of: {supported}")
    loaded = trimesh.load(path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(
            f"'{path.name}' contains no triangle mesh (point clouds are not supported)"
        )
    if loaded.is_watertight and loaded.volume < 0:
        loaded.invert()
    return loaded


@dataclass(frozen=True)
class DeviationStats:
    """Summary of one set of signed point-to-surface distances.

    Attributes:
        rms: Root mean square deviation.
        mean_signed: Average signed deviation (positive = scan larger).
        mean_abs: Average absolute deviation.
        std: Standard deviation of the signed deviations.
        p95_abs: 95th percentile of absolute deviation.
        max_abs: Largest absolute deviation (one-sided Hausdorff distance).
        within_tolerance_pct: Percentage of points with |deviation| <= tolerance.
        tolerance: Tolerance used for ``within_tolerance_pct``.
    """

    rms: float
    mean_signed: float
    mean_abs: float
    std: float
    p95_abs: float
    max_abs: float
    within_tolerance_pct: float
    tolerance: float

    @classmethod
    def from_distances(cls, distances: FloatArray, tolerance: float) -> DeviationStats:
        """Build summary statistics from signed distances.

        Args:
            distances: Signed distances, one per sampled point.
            tolerance: Absolute deviation counted as "within tolerance".

        Raises:
            ValueError: If ``distances`` is empty or ``tolerance`` is negative.
        """
        if distances.size == 0:
            raise ValueError("No distances to summarise")
        if tolerance < 0:
            raise ValueError(f"Tolerance must be non-negative, got {tolerance}")
        absolute = np.abs(distances)
        return cls(
            rms=float(np.sqrt(np.mean(distances**2))),
            mean_signed=float(np.mean(distances)),
            mean_abs=float(np.mean(absolute)),
            std=float(np.std(distances)),
            p95_abs=float(np.percentile(absolute, 95)),
            max_abs=float(np.max(absolute)),
            within_tolerance_pct=float(np.mean(absolute <= tolerance) * 100.0),
            tolerance=tolerance,
        )


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing a scan against an original model.

    Attributes:
        scan_to_original: Deviations measured from scan points to the original surface.
            Shows how accurate the scanned surface is; inflated by untrimmed extra material.
        original_to_scan: Deviations measured from original points to the scan surface,
            sign-flipped so positive still means "scan larger". Inflated by holes or
            missing areas in the scan.
        alignment_rmse: Trimmed RMSE of the final ICP step, or ``None`` if not aligned.
        original_size: Bounding-box extents of the original.
        scan_size: Bounding-box extents of the scan.
        sample_count: Points sampled on each surface.
        warnings: Human-readable problems worth checking.
        scan_transform: Transform that placed the scan onto the original, or ``None``
            if the scan was compared as-is.
    """

    scan_to_original: DeviationStats
    original_to_scan: DeviationStats
    alignment_rmse: float | None
    original_size: tuple[float, float, float]
    scan_size: tuple[float, float, float]
    sample_count: int
    warnings: list[str] = field(default_factory=list)
    scan_transform: FloatArray | None = None

    def metric_rows(self) -> list[tuple[str, str, str]]:
        """Return ``(label, scan->original, original->scan)`` rows formatted for display."""
        a, b = self.scan_to_original, self.original_to_scan
        tol = a.tolerance

        def mm(value: float) -> str:
            return f"{value:.4f} mm"

        def signed_mm(value: float) -> str:
            return f"{value:+.4f} mm"

        return [
            ("RMS deviation", mm(a.rms), mm(b.rms)),
            (
                f"Within ±{tol:g} mm",
                f"{a.within_tolerance_pct:.1f} %",
                f"{b.within_tolerance_pct:.1f} %",
            ),
            ("Mean absolute deviation", mm(a.mean_abs), mm(b.mean_abs)),
            ("Mean signed (+ = scan larger)", signed_mm(a.mean_signed), signed_mm(b.mean_signed)),
            ("Standard deviation", mm(a.std), mm(b.std)),
            ("95th percentile |deviation|", mm(a.p95_abs), mm(b.p95_abs)),
            ("Max |deviation|", mm(a.max_abs), mm(b.max_abs)),
        ]

    def as_text(self) -> str:
        """Return the full result as tab-separated text, ready to paste into a spreadsheet."""
        lines = ["Metric\tScan → Original\tOriginal → Scan"]
        lines += ["\t".join(row) for row in self.metric_rows()]
        lines.append("")
        lines.append(f"Original size\t{format_size(self.original_size)}")
        lines.append(f"Scan size\t{format_size(self.scan_size)}")
        if self.alignment_rmse is not None:
            lines.append(f"Alignment RMSE (best 90%)\t{self.alignment_rmse:.4f} mm")
        lines.append(f"Points sampled per surface\t{self.sample_count}")
        lines += [f"Warning\t{warning}" for warning in self.warnings]
        return "\n".join(lines)


def format_size(size: tuple[float, float, float]) -> str:
    return " × ".join(f"{v:.2f}" for v in size) + " mm"


def _sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> FloatArray:
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float64)


def _sample_with_normals(
    mesh: trimesh.Trimesh, count: int, seed: int
) -> tuple[FloatArray, FloatArray]:
    points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[faces]
    return np.asarray(points, dtype=np.float64), normals


def _apply(transform: FloatArray, points: FloatArray) -> FloatArray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _rigid_fit(source: FloatArray, target: FloatArray, with_scale: bool = False) -> FloatArray:
    """Best rigid transform mapping ``source`` onto ``target`` (Kabsch).

    With ``with_scale`` it also fits a uniform scale factor (Umeyama).
    """
    src_center = source.mean(axis=0)
    dst_center = target.mean(axis=0)
    centered = source - src_center
    covariance = centered.T @ (target - dst_center)
    u, singular, vt = np.linalg.svd(covariance)
    reflection_fix = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ reflection_fix @ u.T
    if with_scale:
        rotation *= np.trace(np.diag(singular) @ reflection_fix) / np.sum(centered**2)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = dst_center - rotation @ src_center
    return transform


def _plane_fit(
    source: FloatArray, target: FloatArray, normals: FloatArray, with_scale: bool = False
) -> FloatArray:
    """Small rigid step minimising point-to-plane distances (linearised least squares).

    With ``with_scale`` it also fits a small uniform scale change about the centroid.
    """
    center = source.mean(axis=0)
    centered = source - center
    columns = [np.cross(centered, normals), normals]
    if with_scale:
        columns.append(np.einsum("ij,ij->i", centered, normals)[:, None])
    offsets = np.einsum("ij,ij->i", target - source, normals)
    solution, *_ = np.linalg.lstsq(np.hstack(columns), offsets, rcond=None)
    scale = 1.0 + solution[6] if with_scale else 1.0
    rotation = scale * Rotation.from_rotvec(solution[:3]).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center + solution[3:6] - rotation @ center
    return transform


def _icp(
    source: FloatArray,
    target_tree: cKDTree,
    target: FloatArray,
    initial: FloatArray,
    iterations: int,
    target_normals: FloatArray | None = None,
) -> tuple[FloatArray, float]:
    """Trimmed ICP. Returns the transform and its trimmed RMSE.

    Uses point-to-point matching, or point-to-plane matching when ``target_normals``
    is given. Point-to-plane measures against the local surface rather than the
    nearest sample, so its accuracy is not limited by the sample spacing.
    """

    def residuals(transform: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        moved = _apply(transform, source)
        dist, idx = target_tree.query(moved)
        if target_normals is not None:
            dist = np.abs(np.einsum("ij,ij->i", moved - target[idx], target_normals[idx]))
        return moved, dist, idx

    transform = initial.copy()
    previous_rmse = np.inf
    for _ in range(iterations):
        moved, dist, idx = residuals(transform)
        keep = dist <= np.quantile(dist, ICP_OVERLAP)
        if target_normals is None:
            step = _rigid_fit(moved[keep], target[idx[keep]])
        else:
            step = _plane_fit(moved[keep], target[idx[keep]], target_normals[idx[keep]])
        transform = step @ transform
        rmse = float(np.sqrt(np.mean(dist[keep] ** 2)))
        if previous_rmse - rmse < 1e-7:
            break
        previous_rmse = rmse
    _, dist, _ = residuals(transform)
    trimmed = dist[dist <= np.quantile(dist, ICP_OVERLAP)]
    return transform, float(np.sqrt(np.mean(trimmed**2)))


def _principal_axes(points: FloatArray) -> FloatArray:
    """Columns are principal axes, largest variance first."""
    _, vectors = np.linalg.eigh(np.cov((points - points.mean(axis=0)).T))
    return np.asarray(vectors[:, ::-1], dtype=np.float64)


def _initial_guesses(source: FloatArray, target: FloatArray) -> list[FloatArray]:
    """Starting poses: as-is, centroid-matched, and every proper PCA axis pairing."""
    src_center = source.mean(axis=0)
    dst_center = target.mean(axis=0)
    src_axes = _principal_axes(source)
    dst_axes = _principal_axes(target)

    guesses = [np.eye(4)]
    for order in permutations(range(3)):
        for signs in product((1.0, -1.0), repeat=3):
            mapping = np.zeros((3, 3))
            mapping[list(order), range(3)] = signs
            rotation = dst_axes @ mapping @ src_axes.T
            if np.linalg.det(rotation) <= 0:
                continue
            guess = np.eye(4)
            guess[:3, :3] = rotation
            guess[:3, 3] = dst_center - rotation @ src_center
            guesses.append(guess)
    centered = np.eye(4)
    centered[:3, 3] = dst_center - src_center
    guesses.append(centered)
    return guesses


def align_scan(
    scan: trimesh.Trimesh,
    original: trimesh.Trimesh,
    seed: int = 0,
) -> tuple[FloatArray, float]:
    """Find the rigid transform that best places ``scan`` onto ``original``.

    Tries many starting orientations with a quick point-to-point ICP, then refines
    the best one, finishing with point-to-plane ICP for sub-sample accuracy.

    Returns:
        The 4x4 transform to apply to the scan and the trimmed alignment RMSE.
    """
    target, target_normals = _sample_with_normals(original, 50_000, seed)
    target_tree = cKDTree(target)
    coarse_source = _sample_surface(scan, 2_000, seed + 1)
    fine_source = _sample_surface(scan, 10_000, seed + 2)

    candidates = [
        _icp(coarse_source, target_tree, target, guess, iterations=20)
        for guess in _initial_guesses(coarse_source, target)
    ]
    best_guess, _ = min(candidates, key=lambda candidate: candidate[1])
    refined, _ = _icp(fine_source, target_tree, target, best_guess, iterations=50)
    return _icp(fine_source, target_tree, target, refined, 100, target_normals)


def estimate_scale(scan: trimesh.Trimesh, original: trimesh.Trimesh, seed: int = 0) -> float:
    """Uniform scale factor that best makes ``scan`` the same size as ``original``.

    Starts from the surface-area ratio and aligns, then refines pose and scale
    together: first point-to-point (robust while far off), then point-to-plane
    (precise). Treat the result as a starting point: if the two models are
    trimmed differently, the overlapping parts may not determine the size well.
    """
    initial = float(np.sqrt(original.area / scan.area))
    sized = trimesh.Trimesh(scan.vertices * initial, scan.faces, process=False)
    transform, _ = align_scan(sized, original, seed)

    target, target_normals = _sample_with_normals(original, 50_000, seed)
    target_tree = cKDTree(target)
    source = _sample_surface(sized, 10_000, seed + 3)
    for iterations, normals in ((30, None), (200, target_normals)):
        for _ in range(iterations):
            moved = _apply(transform, source)
            dist, idx = target_tree.query(moved)
            if normals is not None:
                dist = np.abs(np.einsum("ij,ij->i", moved - target[idx], normals[idx]))
            keep = dist <= np.quantile(dist, ICP_OVERLAP)
            if normals is None:
                step = _rigid_fit(moved[keep], target[idx[keep]], with_scale=True)
            else:
                step = _plane_fit(moved[keep], target[idx[keep]], normals[idx[keep]], True)
            transform = step @ transform
            if np.allclose(step, np.eye(4), atol=1e-9):
                break
    return initial * float(np.cbrt(np.linalg.det(transform[:3, :3])))


def signed_distances(points: FloatArray, mesh: trimesh.Trimesh) -> FloatArray:
    """Exact signed distance from each point to the nearest triangle of ``mesh``.

    Positive means the point lies on the side the triangle normal points to (outside).
    Candidate triangles are found via their centroids, then checked exactly.
    """
    k = min(NEAREST_FACE_CANDIDATES, len(mesh.faces))
    tree = cKDTree(mesh.triangles_center)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    result = np.empty(len(points))

    for start in range(0, len(points), 10_000):
        chunk = points[start : start + 10_000]
        _, candidates = tree.query(chunk, k=k)
        candidates = np.asarray(candidates).reshape(len(chunk), k)
        repeated = np.repeat(chunk, k, axis=0)
        closest = trimesh.triangles.closest_point(  # type: ignore[no-untyped-call]
            triangles[candidates.ravel()], repeated
        )
        dist = np.linalg.norm(repeated - closest, axis=1).reshape(len(chunk), k)
        best = np.argmin(dist, axis=1)
        rows = np.arange(len(chunk))
        offsets = chunk - closest.reshape(len(chunk), k, 3)[rows, best]
        sides = np.einsum("ij,ij->i", offsets, normals[candidates[rows, best]])
        result[start : start + len(chunk)] = np.where(sides < 0, -1.0, 1.0) * dist[rows, best]
    return result


def compare_meshes(
    original: trimesh.Trimesh,
    scan: trimesh.Trimesh,
    tolerance: float = 0.1,
    samples: int = 50_000,
    align: bool = True,
    seed: int = 0,
) -> ComparisonResult:
    """Compare a scan against the original model.

    Args:
        original: Reference model.
        scan: Scanned model. Not modified; alignment is applied to a copy.
        tolerance: Absolute deviation counted as "within tolerance".
        samples: Number of points sampled on each surface.
        align: Automatically align the scan to the original before measuring.
        seed: Random seed for surface sampling, so results are repeatable.

    Returns:
        Statistics in both measuring directions plus sanity-check information.
    """
    if samples <= 0:
        raise ValueError(f"Sample count must be positive, got {samples}")

    warnings: list[str] = []
    original_size = tuple(float(v) for v in original.extents)
    scan_size = tuple(float(v) for v in scan.extents)
    ratio = float(np.linalg.norm(scan.extents) / np.linalg.norm(original.extents))
    if not 1 / SCALE_WARNING_RATIO <= ratio <= SCALE_WARNING_RATIO:
        warnings.append(
            f"Scan is {ratio:.2f}× the size of the original. Check that both use the same units."
        )

    placed = scan.copy()
    alignment_rmse: float | None = None
    transform: FloatArray | None = None
    if align:
        transform, alignment_rmse = align_scan(scan, original, seed)
        placed.apply_transform(transform)

    scan_points = _sample_surface(placed, samples, seed + 10)
    original_points = _sample_surface(original, samples, seed + 11)
    scan_to_original = signed_distances(scan_points, original)
    # Flip sign: original outside the scan means the scan is too small.
    original_to_scan = -signed_distances(original_points, placed)

    return ComparisonResult(
        scan_to_original=DeviationStats.from_distances(scan_to_original, tolerance),
        original_to_scan=DeviationStats.from_distances(original_to_scan, tolerance),
        alignment_rmse=alignment_rmse,
        original_size=(original_size[0], original_size[1], original_size[2]),
        scan_size=(scan_size[0], scan_size[1], scan_size[2]),
        sample_count=samples,
        warnings=warnings,
        scan_transform=transform,
    )
