import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from savefig import savefig

def main():
    N = 8
    row = np.arange(0, 8)
    col = np.arange(0, 8)[:, np.newaxis]
    w = np.e ** (-2 * np.pi * 1j / N)
    F = (1 / np.sqrt(N)) * w ** (row * col)
    assert np.allclose(np.matmul(F, F.conj().T), np.identity(N))
    _, ax = plt.subplots(nrows=N, ncols=1)
    for line in range(N):
        ax[line].plot(row, F[line].real)
        ax[line].plot(row, F[line].imag)
    savefig("Lab 3 - 1")
    plt.close()

    NUM = 5_000
    n = np.linspace(0, 1, num=NUM)
    signal = np.cos(n * 2 * np.pi * 15)
    w_vals = [3, 5, 15, 40]

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    axs[1].set_xlabel("Real")
    axs[1].set_ylabel("Imaginary")
    axs[0].set_xlim([0, 1])
    axs[1].set_xlim([-1.1, 1.1])
    axs[1].set_ylim([-1.1, 1.1])
    axs[0].axhline(color="black")
    axs[1].axhline(color="black")
    axs[1].axvline(color="black")
    axs[1].set(aspect="equal")
    plt.tight_layout()
    axs[0].plot(n, signal, color="green")
    shape = signal * np.e ** (-2 * np.pi * 1j * n)
    axs[1].plot(shape.real, shape.imag)
    stem_idx = int(NUM * 0.21)
    axs[0].plot([n[stem_idx]] * 2, [signal[stem_idx], 0], "r-o", markevery=2)
    axs[1].plot([shape.real[stem_idx], 0], [shape.imag[stem_idx], 0], "r-o", markevery=2)
    plt.subplots_adjust(left=0.12)
    savefig("Lab 3 - 2 Fig 1")
    plt.close()

    fig, axs = plt.subplots(nrows=2, ncols=2)
    for i, w in enumerate(w_vals):
        ax = axs[i // 2][i % 2]
        ax.set_title(f"w = {w}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.axhline(color="black")
        ax.axvline(color="black")
        ax.set(aspect="equal")
    fig.tight_layout(pad=0, h_pad=0.5)
    for i, w in enumerate(w_vals):
        ax = axs[i // 2][i % 2]
        shape = signal * np.e ** (-2 * np.pi * 1j * w * n)
        distance = (1 - (np.abs(shape[1:]) * 0.7 + 0.3))
        colors = plt.colormaps["ocean"](distance)
        coords = np.column_stack((shape.real, shape.imag))
        assert coords.shape == (NUM, 2)
        size = coords.itemsize
        assert coords.strides == (size * 2, size)
        segments = np.lib.stride_tricks.as_strided(coords, (NUM - 1, 2, 2), (size * 2, size * 2, size), writeable=False)
        assert segments.size == colors.size
        col = LineCollection(segments=segments, colors=colors)
        ax.add_collection(col)
    savefig("Lab 3 - 2 Fig 2")
    plt.close()

if __name__ == "__main__":
    main()
