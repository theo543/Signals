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

def gaussian_process(cov_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], samples: int, points: int = 300, min_: float = -1, max_: float = 1):
    sample_points = np.linspace(min_, max_, points)
    covariance = cov_fn(np.broadcast_to(sample_points[np.newaxis, :], (points, points)), np.broadcast_to(sample_points[:, np.newaxis], (points, points)))
    distribution = gaussian(np.zeros(points), covariance)
    for _ in range(samples):
        x = distribution()
        assert np.max(np.abs(x.imag)) < 1e-6
        plt.plot(sample_points, x.real, marker="o", markersize=1, linewidth=0.5)
    savefig(f"Lab 12 Gaussian Process - {cov_fn.__name__}")
    plt.close()

def linear(x, y, a=10):
    return x * y * a

def brownian(x, y):
    return np.minimum(x, y)

def exponential_squared(x, y, a=1000):
    return np.exp(- a * np.abs(x - y) ** 2)

def ornstein_uhlenbeck(x, y, a=2):
    return np.exp(- a * np.abs(x - y))

def periodic(x, y, a = 1, b = 2):
    return np.exp(- a * np.sin(b * np.pi * np.abs(x - y)) ** 2)

def symmetric(x, y, a = 100):
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

    plt.plot(year, monthly, color="black")
    plt.xlabel("Year")
    plt.ylabel("Montly CO2 Average (ppm)")
    savefig("Lab 12 CO2 Monthly")
    plt.close()

if __name__ == "__main__":
    main()
