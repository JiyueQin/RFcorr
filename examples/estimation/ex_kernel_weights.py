import numpy as np
import rfcorr as rc

lags = np.arange(-5, 6)

#%% Example 1: Bartlett weights 1 - |h| / (M + 1) up to M = 3
print(rc.kernel_weights(lags, "bartlett", bandwidth=None, M=3))

#%% Example 2: constant weights up to M = 2
print(rc.kernel_weights(lags, "constant", bandwidth=None, M=2))

#%% Example 3: QS weights with bandwidth 2, cut at M = 4
print(rc.kernel_weights(lags, "qs", bandwidth=2, M=4).round(3))
