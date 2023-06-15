from .kernel import *

def sample_spherical_kernel(n, mean, var, hemisphere=False):
    """Sample from the diffusion kernel on the n-dimensional sphere.

    Parameters
    ----------
    n : int
        The dimensionality of the space in which the spherical surface is 
        embedded.
    mean : np.array [N x n]
        The n-dimensional unit vectors to be used as the means of the 
        distributions from which to sample.
    var : np.array [N] 
        Scalar variances of the distributions for each vector.
    hemisphere : bool
        If True, sample on the n-hemisphere instead of the n-sphere.
    """
    assert len(mean) == len(var)
    assert len(mean[0]) == n
    N = len(var)
    normal = np.random.randn(N, n)
    rejections = normal - (normal * mean).sum(axis=1, keepdims=True) * mean
    rejections /= np.linalg.norm(rejections, axis=1, keepdims=True)
    phi = newton_raphson(n, np.random.random(N), var, hemisphere)
    cosines, sines = np.cos(phi).reshape((-1, 1)), np.sin(phi).reshape((-1, 1))
    return mean * cosines + rejections * sines

@nb.jit(nopython=True, cache=True)
def newton_raphson(n, u_phi, var, hemisphere=False):
    """Invert CDF - u_phi by Newton-Raphson iteration.

    Parameters
    ----------
    n : int
        The dimensionality of the space in which the spherical surface is 
        embedded.
    u_omega : np.array [N]
        CDF values for which to find the corresponding angles phi.
    var : np.array [N] 
        Scalar variances of the distributions for each angle.
    hemisphere : bool
        If True, use PDFs and CDFs from the n-hemisphere instead of the 
        n-sphere.

    Returns
    -------
    phi_opt : np.array [N]
        Array of angles phi for which the CDF is u_phi.
    """
    phi = angle_mean(n, var)
    phi[var == 0.] = 0.
    counter, d_phi = 0, np.ones_like(phi)
    idxs = np.logical_and(np.abs(d_phi) > 1e-14, var > 0.)
    while np.any(idxs) and counter < 1000:
        pdf, cdf = kernel(n, var[idxs], phi[idxs])
        if hemisphere:
            pdf_rev, cdf_rev = kernel(n, var[idxs], np.pi - phi[idxs])
            pdf = pdf + pdf_rev
            cdf = 1. + cdf - cdf_rev
        d_phi = (cdf - u_phi[idxs]) / pdf
        phi[idxs] = phi[idxs] - d_phi
        idxs[idxs] = np.abs(d_phi) > 1e-12
        counter += 1
    return phi
