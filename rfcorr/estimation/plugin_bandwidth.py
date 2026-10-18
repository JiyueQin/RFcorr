from typing import List, Sequence

import numpy as np


def plugin_bandwidth(cov_ls: List[np.ndarray],
                     vec: np.ndarray,
                     n: int,
                     dim: Sequence[int],
                     constant: float = 1,
                     r_max: float = 0.97) -> np.ndarray:
    """
    Data-driven lag truncation for the product Bartlett kernel (parametric plug-in).

    Development function: a building block of rc.corr_dev; its interface may change.

    The variance being estimated is the long-run variance of the scalar series
    V = vec' T, whose lag-h autocovariance is vec' C(h) vec. Along each axis i, V is
    approximated by an AR(1) with lag-1 autocorrelation r_i (from the same lag
    estimates used for the variance), giving the bias constant
    a_i = 2 r_i / (1 - r_i^2) (Andrews 1991). With one bandwidth m for all axes, the
    asymptotic MSE of the Bartlett estimator, proportional to
    (sum_i a_i / m)^2 + (2 / n) (2 m / 3)^n_dim, is minimized at
    m = [(sum_i a_i)^2 n / (n_dim (2/3)^n_dim)]^(1 / (n_dim + 2)),
    which in 1D is Andrews' 1.1447 (a^2 n)^(1/3). Smoother fields give larger r_i
    and so a larger m.

    Note: this plug-in rule is still under development and has not been verified
    (in theory or in simulations). Use the fixed rules ('auto') for results.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    cov_ls : list of np.ndarray
        The 9 lag-wise covariance arrays of T (the 3x3 entries by row), as in
        rc.get_se. Entries may be scalars (zero terms).
    vec : np.ndarray
        Weights of the 3 components of T in V.
    n : int
        Number of valid grid points.
    dim : sequence of int
        Shape of the grid.
    constant : float, optional (default=1)
        Scaling constant for the bandwidth.
    r_max : float, optional (default=0.97)
        Upper bound on |r_i|, so that nearly nonstationary fields do not give an
        unbounded lag.

    ---------------------------------------------------------------------------
    OUTPUT:
    M_tuple : np.ndarray
        Lag truncation in each dimension. The Bartlett weights in
        rc.kernel_weights are 1 - |h| / (M + 1), so M = m - 1, rounded up, at
        least 1, and at most half the side.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    n_dim = len(dim)
    center = [d - 1 for d in dim]

    def acov(idx):
        # vec' C(h) vec at the lag with array index idx; entries can be scalars (zero terms)
        return sum(
            vec[a] * vec[b] * (np.asarray(cov_ls[3 * a + b])[tuple(idx)]
                               if np.ndim(cov_ls[3 * a + b]) else cov_ls[3 * a + b])
            for a in range(3) for b in range(3)
        )

    s0 = acov(center)
    a_sum = 0.0
    for i in range(n_dim):
        if dim[i] < 2 or not s0 > 0:
            continue
        plus, minus = list(center), list(center)
        plus[i] += 1
        minus[i] -= 1
        r = np.nan_to_num((acov(plus) + acov(minus)) / (2 * s0))
        r = min(abs(r), r_max)
        a_sum += 2 * r / (1 - r**2)

    m = (a_sum**2 * n / (n_dim * (2 / 3) ** n_dim)) ** (1 / (n_dim + 2))
    M = max(1.0, np.ceil(constant * m - 1))
    return np.array([min(M, max(1, (d - 1) // 2)) for d in dim], dtype=float)
