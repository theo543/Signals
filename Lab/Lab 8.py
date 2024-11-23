import numpy as np
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from savefig import savefig

def train_ar(data, size):
    matrix = np.empty((data.size - size, 2))
    for i in range(size):
        matrix[:, i] = data[size - i: data.size - i].reverse()
    return matrix

def main():
    N = 1000
    time = np.arange(N)
    a = np.random.random() / 10000
    b = np.random.random() / 100
    def rand_freq(time):
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

    autocorr = np.correlate(series, series, 'full')[len(series):]
    autocorr /= np.max(np.abs(autocorr))
    plt.plot(autocorr)
    plt.xlabel("Delay")
    plt.ylabel("Correlation")
    savefig("Lab 8 autocorrelation")
    plt.close()

if __name__ == "__main__":
    main()
