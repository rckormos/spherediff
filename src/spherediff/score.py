from .kernel import *

def score_spherical_kernel(n, x, mean, var, hemisphere=False):
    """Compute the score function from the diffusion kernel on the n-D sphere.

    Parameters
    ----------
    n : int
        The dimensionality of the space in which the spherical surface is 
        embedded.
    x : np.array [N x n]
        The n-dimensional unit vectors at which to compute the score.
    mean : np.array [N x n]
        The n-dimensional unit vectors to be used as the means of the 
        distributions according to which to compute the score.
    var : np.array [N] 
        Scalar variances of the distributions for each vector.
    hemisphere : bool
        If True, sample on the n-hemisphere instead of the n-sphere.

    Returns
    -------
    score : np.array [N x n]
        The score function evaluated at each point x.
    """
    assert x.shape[0] == mean.shape[0] and x.shape[1] == mean.shape[1]
    assert len(mean) == len(var)
    assert len(mean[0]) == n
    N = len(var)
    projections = (x * mean).sum(axis=1, keepdims=True)
    phi = np.arccos(projections)
    rejections = mean - projections * x
    rejections /= -np.linalg.norm(rejections, axis=1, keepdims=True)
    ker, ker_deriv = raw_kernel(n, var, phi)
    if hemisphere:
        ker_rev, ker_deriv_rev = raw_kernel(n, var, np.pi - phi)
        ker += ker_rev
        ker_deriv += ker_deriv_rev
    return rejections * ker_deriv / ker
