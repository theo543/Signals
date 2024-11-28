import numpy as np
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from savefig import savefig

def train_ar_model(series: np.ndarray, p: int, m: int):
    matrix = np.empty((m, p))
    for i in range(p):
        matrix[:, i] = series[p - i: p - i + m][::-1]
    model, _, _, _ = np.linalg.lstsq(matrix, series[-m:], rcond=None)
    assert model.shape == (p,)
    return model

def predict(series: np.ndarray, model: np.ndarray):
    predictions = np.zeros_like(series)
    for i in range(model.size, series.size):
        predictions[i] = model.T @ series[i - model.size : i][::-1]
    return predictions

def mse(series: np.ndarray, prediction: np.ndarray):
    return np.square(series - prediction).mean()

def normalize(autocorr: np.ndarray):
    autocorr /= np.max(np.abs(autocorr))
    return autocorr

def autocorrelation(series: np.ndarray):
    autocorr = np.zeros(shape=series.size)
    for delay in range(series.size):
        autocorr[delay] = np.sum((series[:-delay] if delay > 0 else series) * series[delay:])
    return normalize(autocorr)

def autocorrelation_numpy(series: np.ndarray):
    return normalize(np.correlate(series, series, 'full')[series.size - 1:])

def main():
    N = 1000
    time = np.arange(N)
    a = np.random.random() / 10000
    b = np.random.random() / 100
    def rand_freq(time) -> np.ndarray:
        return np.sin(2 * np.pi * time * (np.random.random() / 30)) * (time / 400)
    trend = a * np.square(time) + b * time
    season = rand_freq(time) + rand_freq(time)
    noise = np.random.normal(0, 0.5, N)
    series = trend + season + noise
    axs : list[Axes]
    fig, axs = plt.subplots(nrows=4, ncols=1)
    for ax, array, title in zip(axs, [series, trend, season, noise], ["Series", "Trend", "Season", "Noise"]):
        ax.set_ylabel(title)
        ax.plot(array)
    plt.xlabel("Time")
    fig.tight_layout()
    savefig("Lab 8 components")
    plt.close()

    autocorr = autocorrelation(series)
    assert np.allclose(autocorr, autocorrelation_numpy(series))
    plt.plot(autocorr)
    plt.xlabel("Delay")
    plt.ylabel("Correlation")
    savefig("Lab 8 autocorrelation")
    plt.close()

    p = 5
    m = 700
    model = train_ar_model(series, p=p, m=m)
    prediction = predict(series, model)
    plt.plot(series)
    plt.plot(prediction)
    plt.legend(["Series", "Prediction"])
    plt.xlabel("Time")
    plt.title(f"MSE = {mse(series, prediction):.3f}")
    savefig("Lab 8 prediction")

if __name__ == "__main__":
    main()
