import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig

def xy(N: int):
    x = np.broadcast_to(np.arange(0, N), (N, N))
    y = np.broadcast_to(np.arange(0, N)[:, np.newaxis], (N, N))
    return x, y

def im1(N: int):
    x, y = xy(N)
    return np.sin(2 * np.pi * x + 3 * np.pi * y)

def im2(N: int):
    x, y = xy(N)
    return np.sin(4 * np.pi * x) + np.cos(6 * np.pi * y)

def single_freq(N: int, x: int, y: int):
    freq = np.zeros(shape=(N, N))
    freq[x, y] = 1
    freq[(N - x) % N, (N - y) % N] = 1
    return np.fft.ifft2(freq).real

def im3(N: int):
    return single_freq(N, 0, 5)

def im4(N: int):
    return single_freq(N, 5, 0)

def im5(N: int):
    return single_freq(N, 5, 5)

def main():
    N = 50
    for fn in [im1, im2, im3, im4, im5]:
        im = fn(N)
        freq = np.fft.fft2(im)
        ampl = np.abs(freq)
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(im)
        axs[1].imshow(20 * np.log10(ampl + 10**-50))
        fig.suptitle(fn.__name__)
        fig.tight_layout()
        savefig(f"Lab 7 - 1 {fn.__name__}", svg=False)

    # Observations:
    # 1. im1 is rapidly changing both horizontally and vertically, so it contains many frequencies.
    # 2. im2 is only changing horizontally because np.cos(6 * np.pi * y) is constant.
    # 3. im3, im4, and im5 look as expected, with a single frequency.
    # For im2, im3, im4, and im5, extremely small, unexpected frequencies are visible, likely due to floating-point rounding.

if __name__ == "__main__":
    main()
