from typing import Callable

import numpy as np
import scipy
from matplotlib import pyplot as plt

from savefig import savefig

def gaussian(mean: np.ndarray, covariance: np.ndarray) -> Callable[[], np.ndarray]:
    assert mean.size == covariance.shape[0] == covariance.shape[1]

    eigenvals, eigenvecs = np.linalg.eig(covariance)
    tmp = eigenvecs @ np.diag(np.sqrt(np.abs(eigenvals)))
    def sample_gaussian_distribution() -> np.ndarray:
        return tmp @ np.random.normal(size=mean.size) + mean
    return sample_gaussian_distribution

def matrices_of(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.broadcast_to(x[:, np.newaxis], (x.size, y.size)), np.broadcast_to(y[np.newaxis, :], (x.size, y.size))

def gaussian_process(kernel: Callable[[np.ndarray, np.ndarray], np.ndarray], samples: int, points: int = 300, min_: float = -1, max_: float = 1):
    sample_points = np.linspace(min_, max_, points)
    covariance = kernel(*matrices_of(sample_points, sample_points))
    distribution = gaussian(np.zeros(points), covariance)
    for _ in range(samples):
        x = distribution()
        assert np.max(np.abs(x.imag)) < 1e-6
        plt.plot(sample_points, x.real, marker="o", markersize=1, linewidth=0.5)
    savefig(f"Lab 12 Gaussian Process - {kernel.__name__}")
    plt.close()

def linear(x, y, a=10):
    return x * y * a

def brownian(x, y):
    return np.minimum(x, y)

def exponential_squared(x, y, a=1000.0, l=1.0):
    return np.exp(- a * np.abs(x - y) ** 2 / l ** 2)

def ornstein_uhlenbeck(x, y, a=2.0):
    return np.exp(- a * np.abs(x - y))

def periodic(x, y, a = 1.0, b = 2.0):
    return np.exp(- a * np.sin(b * np.pi * np.abs(x - y)) ** 2)

def symmetric(x, y, a = 100.0):
    return np.exp(- a * np.minimum(np.abs(x - y), np.abs(x + y)) ** 2)

def gaussian_figures():
    gaussian_1d = gaussian(np.array([5]), np.array([[10]]))
    plt.plot([gaussian_1d() for _ in range(1_000)])
    savefig("Lab 12 1D Gaussian")
    plt.close()

    gaussian_2d = gaussian(np.array([5, -20]), np.array([[1, -3/5], [-3/5, 2]]))
    points = np.array([gaussian_2d() for _ in range(10_000)])
    plt.scatter(points[:, 0], points[:, 1], s=0.1, c="black")
    plt.gca().set_aspect("equal")
    savefig("Lab 12 2D Gaussian")
    plt.close()

    gaussian_process(linear, points=5, samples=20)
    gaussian_process(brownian, samples=1)
    gaussian_process(exponential_squared, samples=1)
    gaussian_process(ornstein_uhlenbeck, samples=1)
    gaussian_process(periodic, samples=2)
    gaussian_process(symmetric, samples=1)

def main():
    gaussian_figures()

    data = scipy.io.arff.loadarff("mauna-loa-atmospheric-co2-2.arff")[0]
    year = []
    monthly = []
    acc = data[0][6]
    acc_count = 1
    for prev_record, record in zip(data[:-1], data[1:]):
        if prev_record[0] == record[0] and prev_record[1] == record[1]:
            acc += record[6]
            acc_count += 1
        else:
            year.append(prev_record[0] + prev_record[1] / 12)
            monthly.append(acc / acc_count)
            acc = record[6]
            acc_count = 1
    year.append(data[-1][0] + data[-1][1] / 12)
    monthly.append(acc / acc_count)

    year = np.array(year)
    monthly = np.array(monthly)

    def labels():
        plt.xlabel("Year")
        plt.ylabel("Montly CO2 Average (ppm)")

    plt.plot(year, monthly, color="black")
    labels()
    savefig("Lab 12 CO2 Monthly")
    plt.close()

    linear_trend_poly = np.polyfit(year, monthly, 1)
    ob_linear_trend = np.polyval(linear_trend_poly, year)
    monthly -= ob_linear_trend

    # use only last 12 months for model training
    MONTHS = 12

    observation_indices = year[-MONTHS:]
    observed_data = monthly[-MONTHS:]

    def kernel(x, y):
        return 0.3 * exponential_squared(x, y, 0.3, 2) + periodic(x, y, 50, 1)

    prediction_indices = np.concatenate([year, np.arange(2002, 2025, 1/12)])
    c_aa = kernel(*matrices_of(prediction_indices, prediction_indices))
    c_ab = kernel(*matrices_of(prediction_indices, observation_indices))
    c_ba = kernel(*matrices_of(observation_indices, prediction_indices))
    c_bb = kernel(*matrices_of(observation_indices, observation_indices))
    c_bb_inv = np.linalg.inv(c_bb)
    mean = c_ab @ c_bb_inv @ observed_data
    cov = c_aa - c_ab @ c_bb_inv @ c_ba
    model = gaussian(mean, cov)

    plt.plot(year, monthly + ob_linear_trend, "--", color="black", zorder=10)
    pd_linear_trend = np.polyval(linear_trend_poly, prediction_indices)
    for _ in range(10):
        plt.plot(prediction_indices, model().real + pd_linear_trend, color="blue", alpha=0.1)
    labels()
    savefig("Lab 12 CO2 Model")
    plt.close()

if __name__ == "__main__":
    main()
