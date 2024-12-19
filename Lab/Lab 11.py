from typing import cast

import numpy as np
import scipy.linalg

from random_time_series import random_time_series

def hankel_matrix(series: np.ndarray, delay: int):
    columns = series.size - delay + 1
    matrix = np.empty((delay, columns), dtype=series.dtype)
    for i in range(delay):
        matrix[i, :] = series[i: columns + i]
    return matrix

def main():
    SERIES_LEN = 1000
    series, _, _, _ = random_time_series(SERIES_LEN)

    L = 10
    x = hankel_matrix(series, L)

    xxt = x @ x.T
    xtx = x.T @ x
    xxt_eig = np.linalg.eigvals(xxt)
    xtx_eig = np.linalg.eigvals(xtx)
    x_svd = cast(np.ndarray, scipy.linalg.svdvals(x))

    # X * X.T eigenvalues equal singular values squared
    assert np.allclose(xxt_eig.imag, 0)
    assert np.allclose(np.sort(xxt_eig.real), np.sort(x_svd ** 2))

    # non-zero X.T * X eigenvalues equal singular values squared
    assert np.allclose(xtx_eig.imag, 0)
    assert np.allclose(np.sort((xtx_eig.real))[:-L], 0)
    assert np.allclose(np.sort((xtx_eig.real))[-L:], np.sort(x_svd ** 2))


if __name__ == "__main__":
    main()
