import numpy as np


def create_grid(N=64, extent=2.0):
    """Create a cubic 3D grid in dimensionless coordinates x̃ = x / λ̄_C.

    Parameters
    ----------
    N : int
        Number of grid points along each axis.
    extent : float
        Half-width of the domain; coordinates range from -extent to +extent.

    Returns
    -------
    x, y, z : ndarray
        3D arrays of coordinates (shape: N x N x N).
    dx : float
        Grid spacing (assumed equal in all directions).
    """
    axis = np.linspace(-extent, extent, N)
    dx = axis[1] - axis[0]
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return x, y, z, dx
