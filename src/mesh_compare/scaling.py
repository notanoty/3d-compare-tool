"""Uniform scaling of a model so it matches another model's size."""

from __future__ import annotations

import math
from pathlib import Path

import trimesh

from mesh_compare.comparison import load_mesh

GRID_STEPS_MM = (1.0, 2.0, 5.0)


def scaled_path(source: Path) -> Path:
    """Where the scaled copy of ``source`` is saved: next to it, ``<name>_scaled.stl``."""
    return source.with_name(f"{source.stem}_scaled.stl")


def scale_mesh(mesh: trimesh.Trimesh, factor: float) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` scaled uniformly about the origin.

    Raises:
        ValueError: If ``factor`` is not a positive number.
    """
    if not factor > 0 or math.isinf(factor):
        raise ValueError(f"Scale factor must be a positive number, got {factor}")
    return trimesh.Trimesh(mesh.vertices * factor, mesh.faces, process=False)


def scale_file(source: Path, factor: float) -> Path:
    """Scale the model in ``source`` by ``factor`` and save it next to the original.

    Returns:
        Path of the saved ``<name>_scaled.stl``.
    """
    destination = scaled_path(source)
    scale_mesh(load_mesh(source), factor).export(destination)
    return destination


def grid_step(span_mm: float, max_lines: int = 10) -> float:
    """A round grid spacing (1, 2 or 5 × 10^n mm) giving at most ``max_lines`` lines."""
    if span_mm <= 0:
        raise ValueError(f"Span must be positive, got {span_mm}")
    target = span_mm / max_lines
    exponent = math.floor(math.log10(target))
    for multiplier in (*GRID_STEPS_MM, 10.0):
        step = multiplier * 10.0**exponent
        if step >= target:
            return step
    raise AssertionError("unreachable: 10 × 10^n always exceeds the target")  # pragma: no cover
