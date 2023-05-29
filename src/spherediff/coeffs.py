from .poly import *

@nb.jit(nopython=True, cache=True)
def get_coeffs(n, l):
    """Determine the coefficients of the lth Gegenbauer polynomial with  
       alpha = (n - 2)/2

    Parameters
    ----------
    n : int
        The dimension of the space in which the sphere is embedded. The 
        alpha parameter of each Gegenbauer polynomial is (n - 2)/2.
    l : int
        The integer index of the Gegenbauer polynomial of interest.

    Returns
    -------
    coeffs : np.array
        The coefficients of the lth Gegenbauer polynomial with 
        alpha = (n - 2)/2
    """
    coeffs = np.zeros(l + n - 1)
    for i in range(l + n - 1):
        if l % 2 != i % 2:
            continue
        for j in range(0, i + 1, 2):
            if j > n - 2 or i - j > l:
                continue
            comb_factor = comb((n - 2) // 2, j // 2)
            coeffs[i] += comb_factor * \
                fac2_ratio(l + i - j + n - 4, l - i + j, 
                           n - 4, 2 * i - 2 * j) * 2 ** (i - j)
        coeffs[i] *= (-1) ** ((l - i) // 2)
    return coeffs

@nb.jit(nopython=True, cache=True)
def get_coeffs_recursive(n, l, round_coeffs=False):
    """Recursively determine the coefficients of the lth Gegenbauer 
       polynomial with alpha = (n - 2)/2

    Parameters
    ----------
    n : int
        The dimension of the space in which the sphere is embedded. The 
        alpha parameter of each Gegenbauer polynomial is (n - 2)/2.
    l : int
        The integer index of the Gegenbauer polynomial of interest.
    round_coeffs : bool
        Round Gegenbauer coefficients to the nearest integer. Should only  
        be done if n is even.

    Returns
    -------
    coeffs : np.array
        The coefficients of the lth Gegenbauer polynomial with
        alpha = (n - 2)/2
    """
    coeffs_j = np.zeros(n - 1)
    coeffs_jp1 = np.zeros(n)
    for k in range(0, n - 1, 2):
        coeffs_j[k] = (-1) ** (k // 2) * comb((n - 2) // 2, k // 2)
        coeffs_jp1[k + 1] = (n - 2) * coeffs_j[k]
    if l == 0:
        return coeffs_j
    if l == 1:
        return coeffs_jp1
    for j in range(l - 1):
        coeffs_jp2 = np.empty(len(coeffs_jp1) + 1)
        coeffs_jp2 = (2 * j + n) / (j + 2) * \
                     np.hstack((np.zeros(1), coeffs_jp1)) - \
                     (j + n - 2) / (j + 2) * \
                     np.hstack((coeffs_j, np.zeros(2)))
        coeffs_j = coeffs_jp1
        coeffs_jp1 = coeffs_jp2
        if round_coeffs:
            np.around(coeffs_jp2, 0, coeffs_jp2)
    return coeffs_jp2 # j == l - 2, so j + 2 == l

@nb.jit(nopython=True, cache=True)
def get_cheby_coeffs(n, l, round_coeffs=False):
    """Determine the coefficients of Chebyshev polynomials necessary to 
       expand the lth Gegenbauer polynomial with alpha = (n - 2)/2

    Parameters
    ----------
    n : int
        The dimension of the space in which the sphere is embedded. The 
        alpha parameter of each Gegenbauer polynomial is (n - 2)/2.
    l : int
        The integer index of the Gegenbauer polynomial of interest.
    round_coeffs : bool
        If True, round the Gegenbauer coefficients before mapping them 
        through the transformation matrix to Chebyshev expansion coefficients.

    Returns
    -------
    coeffs : np.array
        The coefficients of Chebyshev polynomials necessary to expand the lth 
        Gegenbauer polynomial with alpha = (n - 2)/2, optionally multiplied by 
        the volume element sin**(2 * floor(n - 2)). If n is even, these 
        Chebyshev polynomials are of the first kind; if n is odd, they are of 
        the second kind.
    """
    if n % 2:
        coeffs = np.dot(get_coeffs_recursive(n, l, round_coeffs),
                        chebyshev_second_inv_matrix(l + n - 2))
    else:
        coeffs = np.dot(get_coeffs_recursive(n, l, round_coeffs),  
                        chebyshev_first_inv_matrix(l + n - 2))
    return coeffs

@nb.jit(nopython=True, cache=True)
def get_cheby_coeffs_recursive(n, l, include_volume_element=True):
    """Recursively determine the coefficients of Chebyshev polynomials 
       necessary to expand the lth Gegenbauer polynomial with 
       alpha = (n - 2)/2

    Parameters
    ----------
    n : int
        The dimension of the space in which the sphere is embedded. The 
        alpha parameter of each Gegenbauer polynomial is (n - 2)/2.
    l : int
        The integer index of the Gegenbauer polynomial of interest.
    include_volume_element : bool
        If True, include the volume element sin**(2 * floor(n - 2)) in 
        the expression to be expanded in Chebyshev polynomials.

    Returns
    -------
    coeffs : np.array
        The coefficients of Chebyshev polynomials necessary to expand the lth 
        Gegenbauer polynomial with alpha = (n - 2)/2, optionally multiplied 
        by the volume element sin**(2 * floor(n - 2)). If n is even, these 
        Chebyshev polynomials are of the first kind; if n is odd, they are of 
        the second kind.
    """
    if include_volume_element:
        if l == 0 or l == 1:
            return get_cheby_coeffs(n, l)
        prefactor = 2. ** ((n + 1) % 2) # find coefficients of cosine or 
                                        # sine series, depending on dimension
        coeffs_j = get_cheby_coeffs(n, 0)
        coeffs_jp1 = get_cheby_coeffs(n, 1)
    else:
        if l == 0:
            return np.array([1.])
        if l == 1:
            return np.array([0., n - 2])
        prefactor = 2. # find coefficients of cosine series
        coeffs_j = np.array([1.])
        coeffs_jp1 = np.array([0., n - 2])
    for j in range(l - 1):
        coeffs_jp2 = np.empty(len(coeffs_jp1) + 1)
        factor1 = (2 * j + n) / (2 * j + 4)
        factor2 = (j + n - 2) / (j + 2)
        coeffs_jp2[0] = factor1 * coeffs_jp1[1] - factor2 * coeffs_j[0]
        coeffs_jp2[1] = \
            factor1 * (coeffs_jp1[2] + prefactor * coeffs_jp1[0]) - \
            factor2 * coeffs_j[1]
        for k in range(2, len(coeffs_jp2) - 2):
            coeffs_jp2[k] = \
                factor1 * (coeffs_jp1[k + 1] + coeffs_jp1[k - 1]) - \
                factor2 * coeffs_j[k]
        coeffs_jp2[-1] = factor1 * coeffs_jp1[-1]
        coeffs_jp2[-2] = factor1 * coeffs_jp1[-2]
        coeffs_j = coeffs_jp1
        coeffs_jp1 = coeffs_jp2
    np.round(coeffs_jp2, 14, coeffs_jp2)
    return coeffs_jp2 # j == l - 2, so j + 2 == l

@nb.jit(nopython=True, cache=True)
def fac2_ratio(num, denom1, denom2, denom3):
    """Evaluate the ratio of double factorials for Gegenbauer coefficients.

       This ratio is (num)!! / (denom1)!! / (denom2)!! / (denom3)!!

    Parameters
    ----------
    num : int
        The integer whose double factorial appears in the numerator.
    denom1 : int
        The first integer whose double factorial appears in the denominator.
    denom2 : int
        The second integer whose double factorial appears in the denominator.
    denom3 : int
        The third integer whose double factorial appears in the denominator.

    Returns
    -------
    ratio : float
        The ratio of double factorials defined in the function description.
    """
    assert num == denom1 + denom2 + denom3
    denoms = [denom1, denom2, denom3]
    like_parity_denoms = [denom for denom in denoms
                          if num % 2 == denom % 2]
    max_lp_denom = max(like_parity_denoms)
    for i in range(3):
        if denoms[i] == max_lp_denom:
            max_lp_idx = i
            break
    ratio = 1.
    for k in range(max_lp_denom + 2, num + 2, 2):
        ratio *= k
    for i in range(3):
        if i != max_lp_idx:
            ratio /= factorial2(denoms[i])
    return ratio 

@nb.jit(nopython=True, cache=True)
def factorial2(n):
    """Evaluate the double factorial of an integer.

    Parameters
    ----------
    n : int
        The integer of which to calculate the double factorial.

    Returns
    -------
    fac2 : int
        The double factorial of n.
    """
    fac2 = 1.
    for j in range(n, 0, -2):
        fac2 *= j
    return int(fac2)
