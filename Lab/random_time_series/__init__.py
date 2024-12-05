import numpy as np

def random_time_series(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(size)
    a = np.random.random() / 10000
    b = np.random.random() / 100
    def rand_freq(time) -> np.ndarray:
        return np.sin(2 * np.pi * time * (np.random.random() / 30)) * (time / 400)
    trend = a * np.square(time) + b * time
    season = rand_freq(time) + rand_freq(time)
    noise = np.random.normal(0, 0.5, size)
    series = trend + season + noise
    return series, trend, season, noise
