from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import trimesh


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep the app's saved settings (e.g. the output folder) out of the tests."""
    try:
        from PySide6.QtCore import QSettings
    except ImportError:  # pragma: no cover
        yield
        return
    folder: Path = tmp_path_factory.mktemp("settings")
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(folder))
    yield


@pytest.fixture(scope="session")
def lumpy_mesh() -> trimesh.Trimesh:
    """Closed, asymmetric blob about 20 mm across, so alignment has one right answer."""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    v = sphere.vertices
    x, y, z = (v / 10.0).T
    scale = 1.0 + 0.25 * x + 0.15 * y * z + 0.1 * z**2
    return trimesh.Trimesh(v * scale[:, None], sphere.faces, process=True)


def rigid_transform(
    angles: tuple[float, float, float], shift: tuple[float, float, float]
) -> np.ndarray:
    transform = trimesh.transformations.euler_matrix(*angles)
    transform[:3, 3] = shift
    return np.asarray(transform)
