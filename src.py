import numpy as np
import gstools as gs
import pandas as pd
def gen_field(rho, nx=64, ny=None, var=1.0, len_scale=10.0, seed=None, M_trunc = None, return_corrdinate = False):
    """
    Generate a pair of correlated 2D Gaussian random fields with optional truncation.

    Parameters
    ----------
    rho : float
        Desired pointwise correlation between the two fields (between -1 and 1).
    nx : int, optional (default=64)
        Number of grid points in the x-direction.
    ny : int or None, optional (default=None)
        Number of grid points in the y-direction. If None, uses the same as `nx`.
    var : float, optional (default=1.0)
        Variance parameter for the Gaussian covariance model.
    len_scale : float, optional (default=10.0)
        Length scale parameter of the Gaussian covariance (larger length scale is smoother).
    seed : int or None, optional (default=None)
        Random seed for reproducibility. If None, uses a random seed.
    M_trunc : float or None, optional (default=None)
        Truncation threshold. If provided, both fields are clipped to [-M_trunc, M_trunc].

    Returns
    -------
    X, Y : ndarray
        Two 2D NumPy arrays of shape (nx, ny), representing correlated Gaussian fields.
        If `M_trunc` is not None, the arrays are truncated at ±M_trunc.

    Examples
    --------
    >>> X, Y = gen_field(rho=0.7, nx=50, len_scale=5.0, seed=42, M_trunc=2.0)
    >>> X.shape, Y.shape
    ((50, 50), (50, 50))
    """
    # if seed = None, rng.integers will genereate a random number
    rng = np.random.default_rng(seed)
    # check whether var can be a free parameter?
    model = gs.Gaussian(dim=2, var=var, len_scale=len_scale)
    srf = gs.SRF(model, seed=rng.integers(0, 1e9))

    # structured grid with origin at the center
    if ny is None:
        ny = nx

    origin = (nx//2, ny//2)
    #x = range(nx)
    #y = range(ny)
    x = np.arange(nx) - origin[0]
    y = np.arange(nx) - origin[1]
    results = []
    beta = np.sqrt(max(0.0, 1.0 - rho**2))
    # two independent fields
    U = srf.structured([x, y], seed=rng.integers(0, 1e9))
    V = srf.structured([x, y], seed=rng.integers(0, 1e9))

    # correlated pair of fields
    X = U
    Y = rho * U + beta * V
    # optional: correlated pair of truncated fields
    if M_trunc is not None:
        X = np.clip(X, -M_trunc, M_trunc)
        Y = np.clip(Y, -M_trunc, M_trunc)
    if return_corrdinate: 
        return (X,Y, x, y)
    else:
        return (X,Y)






import numpy as np
from scipy.signal import correlate2d, correlate

def est_exp_X0Yh(X, Y, return_counts_lags = False):
    """
    Estimate E[X_0 * Y_h] for all 2-D integer lags h=(hx,hy) on a grid.

    Parameters
    ----------
    X, Y : (nx, ny) 2D arrays (same shape)
        Input fields. 
    Returns
    -------
    E : (2*nx-1, 2*ny-1) array
        Estimated E[X_0 Y_h] at each lag h.
    counts : (2*nx-1, 2*ny-1) array
        Number of valid pairs at each lag.
    lags_x, lags_y : 1D arrays
        x and y coordinate of the lag
    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    nx, ny = X.shape
    
    mask =  np.ones((nx, ny))
    
    # Sum of products at each lag
    S = correlate(X, Y, mode='full')
    # number of valid pairs at each lag (also known as edge correction for unbiased estimate)
    C = correlate(mask, mask, mode='full')
    # max number of pairs 
    # C = nx*ny
    # estimate of the expectaion 
    E = S/C
    if not return_counts_lags: 
        return E
    else: 
        lags_x = np.arange(-(nx-1), nx)
        lags_y = np.arange(-(ny-1), ny)
        return S, C, E, lags_x, lags_y
    




import time
def corr(X, Y, M=None, true_rho = None, method = 'bartlett'):
    start_time = time.time()
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    nx, ny = X.shape
    X_standard =  (X-X.mean())/X.std()
    Y_standard =  (Y-Y.mean())/Y.std()
    #--------weight function and truncation lag numbder -----------
    if M is None:
        M = np.floor((nx/5, ny/5))
    lags_x = np.arange(-(nx-1), nx)
    lags_y = np.arange(-(ny-1), ny)
    lags_x_all, lags_y_all = np.meshgrid(lags_x, lags_y)
    if method == 'bartlett':
        weights =(np.abs(lags_x_all)<M[0])*(np.abs(lags_y_all)<M[1])*(1-np.abs(lags_x_all)/M[0])*(1-np.abs(lags_y_all)/M[1])
    if method == 'constant':
        weights = (np.abs(lags_x_all)<M[0])*(np.abs(lags_y_all)<M[1])
    if method == 'naive':
        # known to perform poorly, testing only!!
        weights = np.ones((len(lags_x),len(lags_y)))
    #print(weights)
    #print(weights.shape)
    #print(X_standard.mean())
    #print(X_standard.std())
    rho_est = np.corrcoef(X.ravel(), Y.ravel())[0, 1]
    X_standard_square = X_standard **2
    Y_standard_square = Y_standard **2
    X_standard_Y_standard = X_standard * Y_standard
    #print(X_standard_Y_standard.shape)
    if true_rho == 0:
        T1= T2 = 0
    else:
        T1 = (est_exp_X0Yh(X_standard_square, X_standard_square) + est_exp_X0Yh(Y_standard_square, Y_standard_square) + 
              est_exp_X0Yh(X_standard_square, Y_standard_square) + est_exp_X0Yh(Y_standard_square, X_standard_square))
        T2 = (est_exp_X0Yh(X_standard_square, X_standard_Y_standard) + est_exp_X0Yh(X_standard_Y_standard, X_standard_square) +
              est_exp_X0Yh(X_standard_Y_standard, Y_standard_square) + est_exp_X0Yh(Y_standard_square, X_standard_Y_standard))
    T3 = est_exp_X0Yh(X_standard_Y_standard, X_standard_Y_standard)
    #print(np.sqrt(np.sum(T3)/(nx*ny)))
    if true_rho is None:
        h  = weights*(1/4*(rho_est**2)*T1 -1/2*rho_est*T2 +T3)
    else:
        h  = weights*(1/4*(true_rho**2)*T1 -1/2*true_rho*T2 +T3)
    #print(h)
    #print(h.shape)
    #print(np.max(h))
    var = np.sum(h)/(nx*ny)
    #print(var)
    if var<0:
        se = -1
    else:
        se = np.sqrt(var)
    runtime_secs = time.time() - start_time
    return rho_est, se, runtime_secs



def sim_single(rho, m_sim=5, nx=64, ny=None, var=1.0, len_scale=10.0, seed=None, M_trunc =None,
               return_summary = True):
    """
    Returns
    -------
    pandas.DataFrame with columns ['sim_id', 'pearson_r'].
    """
    # if seed = None, rng.integers will genereate a random number
    rng = np.random.default_rng(seed)
    
    if ny is None:
        ny = nx
        
   # if M_trunc is None: 
   #     rho_final = rho
   # else:
   #     rho_final = get_rho_trunc(rho, nx, ny, var, len_scale, seed, M_trunc):    
    
    results = []

    for i in range(1, m_sim+1):
        #? look at len_scale, this may depend on field size
        X,Y = gen_field(rho, nx, ny, var, len_scale, seed=rng.integers(0, 1e9), M_trunc = M_trunc)

        #print('shape of X:', X.shape, 'shape of Y:', Y.shape)

        # Pearson correlation and se
        est, se, runtime_secs = corr(X,Y, true_rho = 0)
        # python list append is similar to preallocate and then fill, faster than concat a row to a pandas dataframe
        results.append({"sim_id": i, "est": est, "se": se, "runtime": runtime_secs})

    results = pd.DataFrame(results) 
    results = results.assign(lower = results["est"]-1.96*results["se"], 
                             upper = results["est"]+1.96*results["se"])
    results = results.assign(cover = 1*(rho<=results["upper"]) & (rho>= results["lower"]))
    if return_summary:
        results = pd.DataFrame({'cover_mean': results["cover"].mean(),
                                'est_mean': results['est'].mean(),
                                'est_sd': results['est'].std(),
                                'se_mean': results['se'].mean(),
                                'runtime_mean': results['runtime'].mean()},
                              index = [0])
    results = results.assign(rho = rho, nx=nx, ny=ny, len_scale=len_scale, M_trunc=M_trunc, m_sim = m_sim) 
    return results



def sim_multiple(setting_df, m_sim, return_summary = False):
    results = []
    for i in setting_df.index:
        print(
            f'----performing simulation, current setting number: {i + 1}, remaining settings: {setting_df.shape[0] - i - 1}----')
        results_single = sim_single(rho=setting_df['rho'][i], m_sim=m_sim, nx=setting_df['nx'][i], len_scale=setting_df['len_scale'][i], return_summary = return_summary)
        results_single = results_single.assign(setting_id = i+1)
        results.append(results_single)
    results = pd.concat(results, ignore_index=True)
    return results
    