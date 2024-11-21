import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig

def main():
    x = np.random.random(size=100)
    fig, axs = plt.subplots(nrows=4, ncols=1)
    for w in range(4):
        axs[w].plot(x)
        x = np.convolve(x, x)
        x /= np.abs(np.max(x))
    fig.tight_layout()
    savefig("Lab 6 convolution bell")
    # The process quickly converges to a bell curve.

if __name__ == "__main__":
    main()
