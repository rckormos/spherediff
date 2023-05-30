from .coeffs import *

@nb.jit(nopython=True, cache=True)
def kernel(n, var, phi):
    """Compute the PDF and CDF of the diffusion kernel on the (n-1)D sphere.

       These are functions of phi_{n-1}, the final polar angle in the 
       hyperspherical coordinates of a sphere embedded in nD space. The 
       volume element, sin^{n-2}(phi_{n-1}) is included, such that the 
       PDF multiplied by the volume element integrates to 1 over [0, pi].

    Parameters
    ----------
    n : int
        The dimensionality of the space in which the spherical surface is 
        embedded.
    var : np.array [N]
        Array of variance parameters for which to compute the PDF and CDF 
        of the diffusion kernel at each angle phi.
    phi : np.array [N]
        Array of angles (in radians) at which to compute the PDF and CDF 
        of the diffusion kernel.

    Returns
    -------
    pdf : np.array [N]
        The PDF of the diffusion kernel on the (n-1)D sphere, evaluated at 
        each angle phi.
    cdf : np.array [N]
        The CDF of the diffusion kernel on the (n-1)D sphere, evaluated at 
        each angle phi.
    """
    norm_square = (2. / np.pi) ** (int(n - 3) % 2) * \
                  2 ** (n - 3)
    for j in range(n - 4, 0, -2):
        norm_square *= j ** 2
    for j in range(2 * n - 6, 0, -2):
        norm_square /= j
    pdf, cdf = np.zeros_like(phi), np.zeros_like(phi)
    gegenbauer_l_1 = 1. # lth Gegenbauer polynomial evaluated at 1
    gegenbauer_lp1_1 = n - 2 # l+1th Gegenbauer polynomial evaluated at 1
    coeffs_l = get_cheby_coeffs(n, 0)
    coeffs_lp1 = get_cheby_coeffs(n, 1)
    converged, prev_converged = False, False
    l = 0
    while not (converged and prev_converged):
        prev_converged = converged
        dpdf, dcdf = np.zeros_like(phi), np.zeros_like(phi)
        for k, coeff in enumerate(coeffs_l):
            if not coeff:
                continue
            if n % 2: # n is odd; use a sine series
                dpdf += coeff * np.sin((k + 1) * phi)
                dcdf += coeff * (1 - np.cos((k + 1) * phi)) / (k + 1) 
            elif k > len(coeffs_l) - n: # n is even; use a cosine series
                if k:
                    dpdf += coeff * np.cos(k * phi)
                    dcdf += coeff * np.sin(k * phi) / k
                else:
                    dpdf += coeff
                    dcdf += coeff * phi
        exp_factor = np.exp(-0.5 * l * (l + n - 2) * var)
        dpdf *= 0.5 * (2 * l + n - 2) * \
                gegenbauer_l_1 * norm_square * exp_factor
        dcdf *= 0.5 * (2 * l + n - 2) * \
                gegenbauer_l_1 * norm_square * exp_factor
        pdf += dpdf
        cdf += dcdf
        if np.abs(dpdf).max() < 1e-14 and np.abs(dcdf).max() < 1e-14:
            converged = True
        else:
            converged = False
        # update coeffs_l and coeffs_lp1 
        coeffs_lp2 = np.empty(len(coeffs_lp1) + 1)
        prefactor = 2. ** ((n + 1) % 2)
        factor1 = (2 * l + n) / (2 * l + 4)
        factor2 = (l + n - 2) / (l + 2)
        coeffs_lp2[0] = factor1 * coeffs_lp1[1] - factor2 * coeffs_l[0]
        coeffs_lp2[1] = \
            factor1 * (coeffs_lp1[2] + prefactor * coeffs_lp1[0]) - \
            factor2 * coeffs_l[1]
        for k in range(2, len(coeffs_lp2) - 2):
            coeffs_lp2[k] = \
                factor1 * (coeffs_lp1[k + 1] + coeffs_lp1[k - 1]) - \
                factor2 * coeffs_l[k]
        coeffs_lp2[-1] = factor1 * coeffs_lp1[-1]
        coeffs_lp2[-2] = factor1 * coeffs_lp1[-2]
        coeffs_l = coeffs_lp1
        coeffs_lp1 = coeffs_lp2
        # update gegenbauer_l_1 and gegenbauer_lp1_1
        gegenbauer_lp1_1, gegenbauer_l_1 = \
            (2 * l + n) / (l + 2) * gegenbauer_lp1_1 - \
            (l + n - 2) / (l + 2) * gegenbauer_l_1, \
            gegenbauer_lp1_1
        # update l and the squared normalization coefficient
        l += 1
        norm_square *= 2 * l / (2 * l + 2 * n - 6)
    return pdf, cdf

@nb.jit(nopython=True, cache=True)
def angle_mean(n, var):
    """Compute the mean of the diffusion kernel on the (n-1)D sphere.

       The PDF over which to average is a function of phi_{n-1}, the 
       final polar angle in the hyperspherical coordinates of a sphere 
       embedded in nD space. The volume element, sin^{n-2}(phi_{n-1}) is 
       included, such that the PDF multiplied by the volume element 
       integrates to 1 over [0, pi].

    Parameters
    ----------
    n : int
        The dimensionality of the space in which the spherical surface is 
        embedded.
    var : np.array [N]
        Array of variance parameters for which to compute the mean of the 
        diffusion kernel.

    Returns
    -------
    mean : np.array [N]
        The mean of the diffusion kernel on the (n-1)D sphere.
    """
    norm_square = (2. / np.pi) ** (int(n - 3) % 2) * \
                  2 ** (n - 3)
    for j in range(n - 4, 0, -2):
        norm_square *= j ** 2
    for j in range(2 * n - 6, 0, -2):
        norm_square /= j
    mean = np.zeros_like(var)
    gegenbauer_l_1 = 1. # lth Gegenbauer polynomial evaluated at 1
    gegenbauer_lp1_1 = n - 2 # l+1th Gegenbauer polynomial evaluated at 1
    coeffs_l = get_cheby_coeffs(n, 0)
    coeffs_lp1 = get_cheby_coeffs(n, 1)
    converged, prev_converged = False, False
    l = 0
    while not (converged and prev_converged):
        prev_converged = converged
        dmean = np.zeros_like(var)
        for k, coeff in enumerate(coeffs_l):
            if not coeff:
                continue
            if n % 2: # n is odd; use a sine series
                dmean += coeff * (np.sin(np.pi * (k + 1)) + 
                                  np.pi * (k + 1) * np.cos(np.pi * k)) / \
                         (k + 1) ** 2
            elif k > len(coeffs_l) - n: # n is even; use a cosine series
                if k:
                    dmean += coeff * (np.pi * k * np.sin(np.pi * k) + 
                                      np.cos(np.pi * k) - 1.) / k ** 2
                else:
                    dmean += coeff * np.pi ** 2 / 2.
        exp_factor = np.exp(-0.5 * l * (l + n - 2) * var)
        dmean *= 0.5 * (2 * l + n - 2) * \
                 gegenbauer_l_1 * norm_square * exp_factor
        mean += dmean
        if np.abs(dmean).max() < 1e-14:
            converged = True
        else:
            converged = False
        # update coeffs_l and coeffs_lp1 
        coeffs_lp2 = np.empty(len(coeffs_lp1) + 1)
        prefactor = 2. ** ((n + 1) % 2)
        factor1 = (2 * l + n) / (2 * l + 4)
        factor2 = (l + n - 2) / (l + 2)
        coeffs_lp2[0] = factor1 * coeffs_lp1[1] - factor2 * coeffs_l[0]
        coeffs_lp2[1] = \
            factor1 * (coeffs_lp1[2] + prefactor * coeffs_lp1[0]) - \
            factor2 * coeffs_l[1]
        for k in range(2, len(coeffs_lp2) - 2):
            coeffs_lp2[k] = \
                factor1 * (coeffs_lp1[k + 1] + coeffs_lp1[k - 1]) - \
                factor2 * coeffs_l[k]
        coeffs_lp2[-1] = factor1 * coeffs_lp1[-1]
        coeffs_lp2[-2] = factor1 * coeffs_lp1[-2]
        coeffs_l = coeffs_lp1
        coeffs_lp1 = coeffs_lp2
        # update gegenbauer_l_1 and gegenbauer_lp1_1
        gegenbauer_lp1_1, gegenbauer_l_1 = \
            (2 * l + n) / (l + 2) * gegenbauer_lp1_1 - \
            (l + n - 2) / (l + 2) * gegenbauer_l_1, \
            gegenbauer_lp1_1
        # update l and the squared normalization coefficient
        l += 1
        norm_square *= 2 * l / (2 * l + 2 * n - 6)
    return mean

@nb.jit(nopython=True, cache=True)
def raw_kernel(n, var, phi):
    """Compute and differentiate the raw diffusion kernel on the (n-1)D sphere.

       This kernel is a function of phi_{n-1}, the final polar angle in the 
       hyperspherical coordinates of a sphere embedded in nD space. The 
       volume element, sin^{n-2}(phi_{n-1}) is not included, hence the 
       designation of the kernel as raw.

    Parameters
    ----------
    n : int
        The dimensionality of the space in which the spherical surface is 
        embedded.
    var : np.array [N]
        Array of variance parameters for which to compute the raw diffusion 
        kernel and its derivative at each angle phi.
    phi : np.array [N]
        Array of angles (in radians) at which to compute the raw diffusion 
        kernel and its derivative.

    Returns
    -------
    kernel : np.array [N]
        The raw diffusion kernel on the (n-1)D sphere, evaluated at each 
        angle phi.
    kernel_deriv : np.array [N]
        The derivative of the raw diffusion kernel on the (n-1)D sphere, 
        evaluated at each angle phi.
    """
    norm_square = (2. / np.pi) ** (int(n - 3) % 2) * \
                  2 ** (n - 3)
    for j in range(n - 4, 0, -2):
        norm_square *= j ** 2
    for j in range(2 * n - 6, 0, -2):
        norm_square /= j
    kernel, kernel_deriv = np.zeros_like(phi), np.zeros_like(phi)
    gegenbauer_l_1 = 1. # lth Gegenbauer polynomial evaluated at 1
    gegenbauer_lp1_1 = n - 2 # l+1th Gegenbauer polynomial evaluated at 1
    coeffs_l = np.array([1.])
    coeffs_lp1 = np.array([0., n - 2])
    converged, prev_converged = False, False
    l = 0
    while not (converged and prev_converged):
        prev_converged = converged
        dker, dker_deriv = np.zeros_like(phi), np.zeros_like(phi)
        for k, coeff in enumerate(coeffs_l):
            if not coeff:
                continue
            dker += coeff * np.cos(k * phi)
            dker_deriv -= coeff * k * np.sin(k * phi)
        exp_factor = np.exp(-0.5 * l * (l + n - 2) * var)
        dker *= 0.5 * (2 * l + n - 2) * \
                gegenbauer_l_1 * norm_square * exp_factor
        dker_deriv *= 0.5 * (2 * l + n - 2) * \
                      gegenbauer_l_1 * norm_square * exp_factor
        kernel += dker
        kernel_deriv += dker_deriv
        if np.abs(dker).max() < 1e-14 and np.abs(dker_deriv).max() < 1e-14:
            converged = True
        else:
            converged = False
        # update coeffs_l and coeffs_lp1
        coeffs_lp2 = np.empty(len(coeffs_lp1) + 1)
        factor1 = (2 * l + n) / (2 * l + 4)
        factor2 = (l + n - 2) / (l + 2)
        coeffs_lp2[0] = factor1 * coeffs_lp1[1] - factor2 * coeffs_l[0]
        coeffs_lp2[1] = \
            factor1 * (coeffs_lp1[2] + 2. * coeffs_lp1[0]) - \
            factor2 * coeffs_l[1]
        for k in range(2, len(coeffs_lp2) - 2):
            coeffs_lp2[k] = \
                factor1 * (coeffs_lp1[k + 1] + coeffs_lp1[k - 1]) - \
                factor2 * coeffs_l[k]
        coeffs_lp2[-1] = factor1 * coeffs_lp1[-1]
        coeffs_lp2[-2] = factor1 * coeffs_lp1[-2]
        coeffs_l = coeffs_lp1
        coeffs_lp1 = coeffs_lp2
        # update gegenbauer_l_1 and gegenbauer_lp1_1
        gegenbauer_lp1_1, gegenbauer_l_1 = \
            (2 * l + n) / (l + 2) * gegenbauer_lp1_1 - \
            (l + n - 2) / (l + 2) * gegenbauer_l_1, \
            gegenbauer_lp1_1
        # update l and the squared normalization coefficient
        l += 1
        norm_square *= 2 * l / (2 * l + 2 * n - 6)
    return kernel, kernel_deriv
