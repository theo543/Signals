import numpy as np

from replace_me import insert_comment

from random_time_series import random_time_series
from ar_model import mse, ar_predict

def train_sparse_ar_model(series: np.ndarray, p: int, m: int, zero_indices: list[int]) -> np.ndarray:
    assert all(0 <= index < p for index in zero_indices)
    assert np.unique(zero_indices).size == len(zero_indices)

    matrix = np.empty((m + len(zero_indices), p))
    for i in range(p):
        matrix[:m, i] = series[p - i : p - i + m][::-1]
    for i, index in enumerate(zero_indices):
        matrix[m + i, :] = 0
        matrix[m + i, index] = 1_000_000_000_000

    b = np.zeros(m + len(zero_indices))
    b[:m] = series[-m:]

    model, _, _, _ = np.linalg.lstsq(matrix, b, rcond=None)

    assert model.shape == (p,)
    assert np.allclose(model[zero_indices], 0)

    return model

def greedy_train_sparse_ar_model(series: np.ndarray, p: int, m: int, sparse_p: int) -> np.ndarray:
    zero_indices = []
    for _ in range(p - sparse_p):
        choices_mse = np.full(p, np.inf)
        for i in range(p):
            if i in zero_indices:
                continue
            model = train_sparse_ar_model(series, p, m, zero_indices + [i])
            choices_mse[i] = mse(series[p:], ar_predict(series, model))
        best_i = np.argmin(choices_mse)
        zero_indices.append(best_i)
    return train_sparse_ar_model(series, p, m, zero_indices)

def poly_roots(poly: np.ndarray) -> np.ndarray:
    companion = np.zeros((poly.size, poly.size))
    companion[0, -1] = -poly[0]
    for i, c in enumerate(poly[1:]):
        companion[i + 1, i] = 1
        companion[i + 1, -1] = -c
    return np.linalg.eigvals(companion)

def main():
    np.random.seed(42)
    SERIES_LEN = 1000
    series, _, _, _ = random_time_series(SERIES_LEN)
    p = 50
    m = 700
    sparse_p = 10

    spare_model = greedy_train_sparse_ar_model(series, p, m, sparse_p)
    sparse_mse = mse(series[p:], ar_predict(series, spare_model))

    insert_comment(f"{sparse_mse:.3f}")
    # 5.987

    insert_comment(np.around(np.abs(poly_roots(spare_model)), decimals=3))
    # array([0.999, 0.999, 1.005, 1.005, 1.009, 1.009, 1.009, 1.009, 1.006,
    #        1.006, 1.001, 1.001, 0.999, 0.999, 1.   , 1.   , 0.999, 0.999,
    #        0.994, 0.994, 0.978, 0.978, 0.969, 0.969, 0.988, 0.988, 0.999,
    #        0.999, 1.001, 1.001, 0.998, 0.998, 0.986, 0.986, 0.962, 0.962,
    #        0.957, 0.957, 0.991, 0.991, 0.984, 0.984, 0.968, 0.968, 0.96 ,
    #        0.96 , 0.972, 0.972, 0.973, 0.973])

if __name__ == "__main__":
    main()
