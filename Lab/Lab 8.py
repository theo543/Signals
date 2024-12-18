import numpy as np
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from savefig import savefig
from ar_model import mse, train_ar_model, ar_predict
from random_time_series import random_time_series

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
    series, trend, season, noise = random_time_series(N)
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
    prediction = ar_predict(series, model)
    plt.plot(series)
    plt.plot(prediction)
    plt.legend(["Series", "Prediction"])
    plt.xlabel("Time")
    plt.title(f"MSE = {mse(series[p:], prediction):.3f}")
    savefig("Lab 8 prediction")
    plt.close()

    train_percent = 0.2
    train_size = int(series.size * train_percent)
    train = series[:train_size]

    best_mse = np.inf
    p_step = 10
    m_step = 150
    p_max = int(train.size * 0.2)
    random = np.random.default_rng(seed=0)
    best_p = random.integers(2, p_max)
    best_m = random.integers(1, train.size - 1)
    no_change = 0
    convergence_threshold = 5_000
    while no_change < convergence_threshold:
        p = int(np.clip(best_p + random.normal(0, p_step), 2, p_max))
        m = int(np.clip(best_m + random.normal(0, m_step), 1, train.size - p))
        model = train_ar_model(train, p=p, m=m)
        new_mse = mse(train[p:], ar_predict(train, model))
        if best_mse > new_mse:
            best_p = p
            best_m = m
            best_mse = new_mse
            no_change = 0
        else:
            no_change += 1
        print(f"p = {best_p:3d}, m = {best_m:3d}, MSE = {best_mse:.3f}, will stop if no change for {convergence_threshold - no_change:5d} more iterations", end="\r")
    print()

    full_prediction = ar_predict(series, train_ar_model(train, p=best_p, m=best_m))
    full_mse = mse(series[best_p:], full_prediction)
    plt.plot(np.arange(series.size), series)
    plt.plot(np.arange(best_p, series.size), full_prediction)
    plt.axvline(train_size, color="red", linestyle="--")
    plt.text(train_size - 200, train.max(), "Train Data", color="red")
    plt.legend(["Series", "Prediction"])
    plt.xlabel("Time")
    plt.title(f"MSE on full data = {full_mse:.3f}\nMSE on train data = {best_mse:.3f}")
    savefig("Lab 8 hyperparameter tuning")

if __name__ == "__main__":
    main()
