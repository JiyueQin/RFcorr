import numpy as np
import rfcorr as rc

x = np.linspace(-1.5, 1.5, 7)

#%% Example 1: Bartlett kernel
print(rc.kernel(x, "bartlett"))

#%% Example 2: Quadratic Spectral kernel
print(rc.kernel(x, "qs").round(3))

#%% Example 3: constant kernel (returns 1)
print(rc.kernel(x, "constant"))
