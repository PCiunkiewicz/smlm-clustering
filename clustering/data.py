"""
Author: Philip Ciunkiewicz
"""
import numpy as np
from sklearn import datasets


def make_training_dataset(size=1000, noise=0.03, std=0.1, state=8):
    """Generate training dataset with various
    cluster geometries and noise levels.

    Parameters:
    -----------
    size : int
        Number of samples for each structure.
    noise : float
        Noisiness of circles and moons (SKLearn).
    std : float
        Standard deviation of blob clusters (SKLearn).
    state : int
        Random state for consistent dataset generation.

    Returns:
    --------
    XY : array_like
        2D coordinate array of length 3.25x size.
    """
    noisy_circles = datasets.make_circles(n_samples=size, random_state=state, factor=.3, noise=noise)
    noisy_moons = datasets.make_moons(n_samples=size, random_state=state, noise=noise)
    blobs = datasets.make_blobs(n_samples=size, random_state=state, centers=15, cluster_std=std)
    np.random.seed(state)
    bkgnd = np.dstack((np.random.uniform(-12, 15, size=size//4), np.random.uniform(-15, 12, size=size//4)))[0]

    moons = noisy_moons[0] * 4
    moons[:,0] += 5
    moons[:,1] -= 10

    circles = noisy_circles[0] * 2
    circles[:,0] -= 7
    circles[:,1] += 3

    XY = np.vstack((circles, moons, blobs[0], bkgnd)) * 1000

    return XY
