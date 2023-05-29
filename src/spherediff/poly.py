import numpy as np
import numba as nb

@nb.jit(nopython=True, cache=True)
def chebyshev_first_matrix(n_max):
    """Return a lower triangular matrix defined such that each row gives 
       the coefficients of the Chebyshev polynomials of the first kind.

    Parameters
    ----------
    n_max : int
        The maximum-degree Chebyshev polynomial in the matrix.

    Returns
    -------
    M : np.array [(n_max + 1) x (n_max + 1)]
        A lower triangular matrix defined such that each row gives 
        the coefficients of the Chebyshev polynomials of the first kind.
    """
    M = np.zeros((n_max + 1, n_max + 1))
    for n in range(n_max + 1):
        for k in range(n + 1):
            if k % 2 != n % 2:
                continue
            M[n, k] = (-1) ** ((n - k) // 2) * factorial((n + k) // 2 - 1) / \
                      factorial((n - k) // 2) / factorial(k) * 2 ** k * n // 2
    M[0, 0] = 1
    return M

@nb.jit(nopython=True, cache=True)
def chebyshev_first_inv_matrix(n_max):
    """Return the matrix to expand a degree-n_max polynomial in Chebyshev 
       polynomials of the first kind.

    Parameters
    ----------
    n_max : int
        Degree of polynomial to expand in Chebyshev polynomials of the 
        first kind.

    Returns
    -------
    M : np.array [(n_max + 1) x (n_max + 1)]
        Matrix to expand a degree-n_max polynomial in Chebyshev 
        polynomials of the first kind. Left-multiplication by the row 
        vector of coefficients converts it to the row vector of Chebyshev 
        expansion coefficients.
    """
    M = np.zeros((n_max + 1, n_max + 1))
    for n in range(n_max + 1):
        for k in range(n + 1):
            if k % 2 != n % 2:
                continue
            M[n, k] = 2. ** (1 - n) * comb(n, (n - k) // 2)
            if k == 0:
                M[n, k] /= 2.
    return M

@nb.jit(nopython=True, cache=True)
def chebyshev_second_matrix(n_max):
    """Return a lower triangular matrix defined such that each row gives 
       the coefficients of the Chebyshev polynomials of the second kind.

    Parameters
    ----------
    n_max : int
        The maximum-degree Chebyshev polynomial in the matrix.

    Returns
    -------
    M : np.array [(n_max + 1) x (n_max + 1)]
        A lower triangular matrix defined such that each row gives 
        the coefficients of the Chebyshev polynomials of the second kind.
    """
    M = np.zeros((n_max + 1, n_max + 1))
    for n in range(n_max + 1):
        for k in range(n + 1):
            if k % 2 != n % 2:
                continue
            M[n, k] = (-1) ** ((n - k) // 2) * factorial((n + k) // 2) / \
                      factorial((n - k) // 2) / factorial(k) * 2 ** k
    return M

@nb.jit(nopython=True, cache=True)
def chebyshev_second_inv_matrix(n_max):
    """Return the matrix to expand a degree-n_max polynomial in Chebyshev 
       polynomials of the second kind.

    Parameters
    ----------
    n_max : int
        Degree of polynomial to expand in Chebyshev polynomials of the 
        second kind.

    Returns
    -------
    M : np.array [(n_max + 1) x (n_max + 1)]
        Matrix to expand a degree-n_max polynomial in Chebyshev 
        polynomials of the second kind. Left-multiplication by the row 
        vector of coefficients converts it to the row vector of Chebyshev 
        expansion coefficients.
    """
    M = np.zeros((n_max + 1, n_max + 1))
    for n in range(n_max + 1):
        for k in range(n + 1):
            if k % 2 != n % 2:
                continue
            M[n, k] = 2. ** -n * comb(n, (n - k) // 2) * (k + 1) / \
                      ((n + k) // 2 + 1)
    return M

@nb.jit(nopython=True, cache=True)
def factorial(n):
    """Evaluate the factorial of n.

    Parameters
    ----------
    n : int
        The integer from which to compute the factorial.

    Returns
    -------
    fac : int
        The factorial of n.
    """
    fac = 1
    for i in range(1, n + 1):
        fac *= i
    return fac

@nb.jit(nopython=True, cache=True)
def comb(n, k):
    """Evaluate the binomial coefficient nCk.

    Parameters
    ----------
    n : int
        The number of objects from which to choose.
    k : int
        The number of objects to choose.

    Returns
    -------
    nCk : int
        The number of ways of choosing k objects from n objects.
    """
    nCk = 1.
    for j in range(1, k + 1):
        nCk *= (n - k + j) / j
    return np.round(nCk)
