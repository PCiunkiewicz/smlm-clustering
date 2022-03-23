"""
Author: Philip Ciunkiewicz
"""
import numpy as np


def custom_round(x, base=5):
    """Rounding to a custom base.
    """
    return int(base * np.ceil(float(x)/base))


def custom_ravel(a, b):
    """Custom ravelling function for arrays
    of varying lengths (len(a) > len(b)).
    """
    ab = []
    for i, x in enumerate(a):
        ab.append(x)
        try:
            ab.append(b[i])
        except IndexError:
            pass
    return ab


def resample_2d(X, resolution):
    """Resample input data for efficient plotting.

    Parameters:
    -----------
    X : array_like
        Input data for clustering.
    resolution : int
        Number of "pixels" for 2d histogram downscaling.
        Default 'auto' downscales to 200x200 for >5000
        samples, and no downscaling for <=5000 samples.

    Returns:
    --------
    xx[mask] : array_like
        Rescaled x meshgrid.
    yy[mask] : array_like
        Rescaled y meshgrid.
    """
    x, y = X[:,0], X[:,1]
    nbins = np.ptp(X, axis=0) / resolution

    hh, locx, locy = np.histogram2d(x, y, bins=np.ceil(nbins).astype('int'))
    xwidth, ywidth = np.diff(locx).mean(), np.diff(locy).mean()
    mask = hh != 0

    locx = locx[:-1] + xwidth
    locy = locy[:-1] + ywidth
    yy, xx = np.meshgrid(locy, locx)
    np.random.seed(0)
    yy += np.random.uniform(-xwidth/2, xwidth/2, size=hh.shape)
    xx += np.random.uniform(-ywidth/2, ywidth/2, size=hh.shape)

    return xx[mask], yy[mask]
