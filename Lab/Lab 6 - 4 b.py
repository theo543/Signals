import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig

def main():
    data = np.genfromtxt("Lab 5 Data.csv", delimiter=",", skip_header=1, usecols=(2,), max_rows=72)
    fig, axs = plt.subplots(nrows=5, ncols=1)
    axs[0].plot(data)
    for i, w in enumerate([5, 10, 15, 20]):
        smooth = np.convolve(data, np.ones(w), 'valid')
        axs[i + 1].plot(smooth)
    fig.tight_layout()
    savefig("Lab 6 - 4 b")

if __name__ == "__main__":
    main()
