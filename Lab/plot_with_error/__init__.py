import numpy as np
from matplotlib import pyplot as plt

def plot_with_error(sizes, data):
    plt.errorbar(np.arange(len(sizes)), np.average(data, axis=1), np.std(data, axis=1))
    plt.plot(np.min(data, axis=1))
    _, right = plt.xlim()
    tick_pos = right * (np.arange(len(sizes)) / len(sizes))
    plt.xticks(tick_pos, [str(size) for size in sizes])
