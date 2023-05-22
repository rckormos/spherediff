import numpy as np
import numba as nb

@nb.jit(nopython=True, cache=True)
def chebyshev_first_matrix(n_max):
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
    fac = 1
    for i in range(1, n + 1):
        fac *= i
    return fac

@nb.jit(nopython=True, cache=True)
def comb(n, k):
    nCk = 1.
    for j in range(1, k + 1):
        nCk *= (n - k + j) / j
    return np.round(nCk)
