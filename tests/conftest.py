import numpy as np
import pytest
import trimesh


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
