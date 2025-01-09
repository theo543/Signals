from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from savefig import savefig

def gaussian(mean: np.ndarray, covariance: np.ndarray) -> Callable[[], np.ndarray]:
    assert mean.size == covariance.shape[0] == covariance.shape[1]

    eigenvals, eigenvecs = np.linalg.eig(covariance)
    tmp = eigenvecs @ np.diag(np.sqrt(eigenvals))
    def sample_gaussian_distribution() -> np.ndarray:
        return tmp @ np.random.normal(size=mean.size) + mean
    return sample_gaussian_distribution

def main():
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

if __name__ == "__main__":
    main()
