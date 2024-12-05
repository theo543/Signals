import numpy as np

def train_ar_model(series: np.ndarray, p: int, m: int) -> np.ndarray:
    matrix = np.empty((m, p))
    for i in range(p):
        matrix[:, i] = series[p - i : p - i + m][::-1]
    model, _, _, _ = np.linalg.lstsq(matrix, series[-m:], rcond=None)
    assert model.shape == (p,)
    return model

def ar_predict(series: np.ndarray, model: np.ndarray) -> np.ndarray:
    predictions = np.zeros_like(series)[model.size:]
    for i in range(model.size, series.size):
        predictions[i - model.size] = model.T @ series[i - model.size : i][::-1]
    return predictions

def mse(series: np.ndarray, prediction: np.ndarray) -> float:
    return np.square(series - prediction).mean()
