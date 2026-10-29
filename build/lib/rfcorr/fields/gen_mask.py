import numbers
from typing import Optional, Tuple, Union

import numpy as np


def gen_mask(dim: Union[int, Tuple[int, ...]],
             shape: str = "ellipse",
             axes: Optional[Union[float, Tuple[float, ...]]] = None) -> np.ndarray:
    """
    Generate a fixed mask of valid grid points.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    dim : int or tuple of int
        Dimensions of the grid.
    shape : str, optional (default="ellipse")
        Mask shape. Only 'ellipse' is supported: an ellipse (ellipsoid for n_dim > 2)
        centered in the grid with its axes along the grid axes.
    axes : float, tuple of float, or None, optional (default=None)
        Semi-axis in each dimension as a fraction of half the side length, in (0, 1].
        A float uses the same fraction in every dimension. If None, all fractions
        are 1, i.e. the ellipse inscribed in the grid.

    ---------------------------------------------------------------------------
    OUTPUT:
    mask : np.ndarray of bool
        True at valid grid points, shape `dim`.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    if isinstance(dim, numbers.Integral):
        dim = (dim,)
    n_dim = len(dim)
    if shape != "ellipse":
        raise ValueError("shape must be 'ellipse'")
    if axes is None:
        axes = (1.0,) * n_dim
    elif isinstance(axes, numbers.Number):
        axes = (axes,) * n_dim
    if len(axes) != n_dim:
        raise ValueError("axes must have one entry per dimension")

    # centered coordinates scaled by the semi-axes, so the ellipse is the unit ball
    coords = np.indices(dim, dtype=float)
    dist2 = sum(
        ((coords[i] - (dim[i] - 1) / 2) / (axes[i] * dim[i] / 2)) ** 2 for i in range(n_dim)
    )
    return dist2 <= 1
