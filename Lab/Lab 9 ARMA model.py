import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig
from random_time_series import random_time_series
from ar_model import mse, train_ar_model, ar_predict

def train_arma_model(series: np.ndarray, ar_p: int, ar_m: int, ma_p: int, ma_m: int) -> tuple[np.ndarray, np.ndarray]:
    ar = train_ar_model(series, ar_p, ar_m)
    errors = np.zeros_like(series)
    errors[ar_p:] = series[ar_p:] - ar_predict(series, ar)
    ma = train_ar_model(errors, ma_p, ma_m)
    return ar, ma

def arma_predict(series: np.ndarray, ar: np.ndarray, ma: np.ndarray) -> np.ndarray:
    predictions = ar_predict(series, ar)
    errors = series[ar.size:ar.size+ma.size] - predictions[:ma.size]
    for i in range(ma.size, series.size):
        predicted_error = ma @ errors
        predictions[i - ar.size] += predicted_error
        errors[:-1] = errors[1:]
        errors[-1] = series[i] - predictions[i - ar.size]
    return predictions

def plot_arma(series: np.ndarray, prediction: np.ndarray, p: int, div_point: int, title: str, label_a: str, label_b: str):
    plt.plot(np.arange(series.size), series)
    plt.plot(np.arange(p, series.size), prediction)
    plt.legend((["Series", "Prediction"]))
    plt.title(f"{title}\nMSE = {mse(series[p:], prediction):.2f}")
    plt.axvline(div_point, color="red", linestyle="--")
    left_mse = mse(series[p:div_point], prediction[:div_point - p])
    right_mse = mse(series[div_point:], prediction[div_point - p:])
    plt.text(div_point - 200, series[:div_point].max(), f"{label_a}\nMSE: {left_mse:.2f}", color="red")
    plt.text(div_point + 50, prediction[div_point:].min() - 10, f"{label_b}\nMSE: {right_mse:.2f}", color="red")

def main():
    series, _, _, _ = random_time_series(1000)

    train_percent = 0.6
    train_size = int(series.size * train_percent)
    train = series[:train_size]

    ar_p = 15
    ar_m = 500
    ma_p = 15
    ma_m = 500
    prediction = arma_predict(series, *train_arma_model(train, ar_p, ar_m, ma_p, ma_m))
    best_mse = mse(train[ar_p:], prediction[:train_size - ar_p])

    plot_arma(series, prediction, ar_p, train_size, f"AR: {ar_p}, {ar_m}, MA: {ma_p}, {ma_m}", "", "")
    savefig("Lab 9 ARMA model")
    plt.close()

    random = np.random.default_rng(seed=0)
    no_change = 0
    while no_change < 2000:
        print(f"AR: {ar_p}, {ar_m}, MA: {ma_p}, {ma_m}, MSE: {best_mse}, No change: {no_change}")
        new_ar_p = int(np.clip(ar_p + random.normal(0, 5), 1, 20))
        new_ar_m = int(np.clip(ar_m + random.normal(0, train_size), 1, train.size - 21))
        new_ma_p = int(np.clip(ma_p + random.normal(0, 5), 1, 20))
        new_ma_m = int(np.clip(ma_m + random.normal(0, train_size), 1, train.size - 21))
        arma = train_arma_model(train, new_ar_p, new_ar_m, new_ma_p, new_ma_m)
        prediction = arma_predict(train, *arma)
        new_mse = mse(train[new_ar_p:], prediction)
        if new_mse < best_mse:
            best_mse = new_mse
            ar_p, ar_m, ma_p, ma_m = new_ar_p, new_ar_m, new_ma_p, new_ma_m
            no_change = 0
        else:
            no_change += 1

    prediction = arma_predict(series, *train_arma_model(train, ar_p, ar_m, ma_p, ma_m))
    plot_arma(series, prediction, ar_p, train_size, f"AR: {ar_p}, {ar_m}, MA: {ma_p}, {ma_m}", "Train", "Test")
    savefig("Lab 9 ARMA model hyperparameter tuning")
    plt.close()


if __name__ == "__main__":
    main()
