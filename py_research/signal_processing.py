
import numpy as np

def fill_nans(X):
    # source: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    rows,cols = X.shape

    for r in range(rows):
        nan_indxs = np.where(np.isnan(X[r,:]))[0]
        
        if nan_indxs.size > 0:
            good_indxs = np.setdiff1d(nan_indxs,np.arange(cols))
            good_data = X[r,good_indxs]
            interpolated = np.interp(nan_indxs, good_indxs, good_data)
            X[r,nan_indxs] = interpolated
    return X


def finite_diff(x,dt=0.001,lag=3):
    x_ = np.zeros_like(x)
    for i in range(lag,x.shape[1]-lag):
        x_[:,i] = ( x[:,i+lag] - x[:,i-lag] ) / (2*lag*dt)
    return x_[:,lag:-lag]