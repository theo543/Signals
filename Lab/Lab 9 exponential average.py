import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig
from random_time_series import random_time_series
from ar_model import mse

def exponential_average(series: np.ndarray, alpha: float):
    prediction = np.empty_like(series)
    prediction[0] = series[0]
    for i in range(1, series.size):
        prediction[i] = alpha * series[i] + (1 - alpha) * prediction[i - 1]
    return prediction

def train_ma_model(error: np.ndarray, p: int, m: int):
    matrix = np.empty((m, p))
    for i in range(p):
        matrix[:, i] = error[p - i : p - i + m][::-1]
    model, _, _, _ = np.linalg.lstsq(matrix, error[p:], rcond=None)
    return model

def ma_predict(series: np.ndarray, model: np.ndarray):
    predictions = np.empty_like(series)
    errors = np.empty_like(series)
    predictions[:model.size] = series[:model.size]
    errors[:model.size] = 0
    for i in range(model.size, series.size):
        predictions[i] = model.T @ errors[i - model.size : i][::-1]
        errors[i] = np.abs(series[i] - predictions[i])
    return predictions

def plot_exp_avg(series: np.ndarray, prediction: np.ndarray, alpha: float):
    plt.plot(series[1:])
    plt.plot(prediction)
    plt.xlabel("Time")
    plt.legend(["Series", "Prediction"])
    plt.title(f"MSE = {mse(series[1:], prediction[:-1]):.3f}\nAlpha = {alpha:.3f}")

def plot_vline(series: np.ndarray, train_size: int, label_a: str, label_b: str):
    plt.axvline(x=train_size, color="red", linestyle="--")
    plt.text(train_size - 300, series[:train_size].max(), label_a)
    plt.text(train_size + 50, series[train_size:].min() - 5, label_b)

def main():
    series, _, _, _ = random_time_series(1000)

    train_percent = 0.75
    train_size = int(series.size * train_percent)
    train = series[:train_size]

    alpha = 0.5
    prediction = exponential_average(series, alpha)
    best_mse = mse(series[1:], prediction[:-1])

    plot_exp_avg(series, prediction, alpha)
    plot_vline(series, train_size,
                f"MSE={mse(train[1:], prediction[:train_size - 1]):.3f}",
                f"MSE={mse(series[train_size + 1:], prediction[train_size:-1]):.3f}"
    )

    savefig(f"Lab 9 exponential average alpha {alpha}")
    plt.close()

    for new_alpha in np.linspace(0, 1, num=1_000):
        new_prediction = exponential_average(train, new_alpha)
        new_mse = mse(train[1:], new_prediction[:-1])
        if best_mse > new_mse:
            best_mse, prediction, alpha = new_mse, new_prediction, new_alpha

    prediction = exponential_average(series, alpha)
    plot_exp_avg(series, prediction, alpha)
    plot_vline(series, train_size,
                f"Train data\nMSE={mse(train[1:], prediction[:train_size - 1]):.3f}",
                f"Test data\nMSE={mse(series[train_size + 1:], prediction[train_size:-1]):.3f}"
    )
    savefig("Lab 9 exponential average alpha search")
    plt.close()

if __name__ == "__main__":
    main()
