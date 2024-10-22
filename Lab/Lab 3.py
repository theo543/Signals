import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from savefig import savefig

def main():
    N = 8
    row = np.arange(0, N)
    col = np.arange(0, N)[:, np.newaxis]
    w = np.e ** (-2 * np.pi * 1j / N)
    F = (1 / np.sqrt(N)) * w ** (row * col)
    assert np.allclose(F @ F.conj().T, np.identity(N))
    _, ax = plt.subplots(nrows=N, ncols=1)
    for line in range(N):
        ax[line].plot(row, F[line].real)
        ax[line].plot(row, F[line].imag)
    savefig("Lab 3 - 1")
    plt.close()

    N = 5_000
    n = np.linspace(0, 1, num=N)
    signal = np.sin(n * 2 * np.pi * 5) * 3 + np.cos(n * 2 * np.pi * 11 + 111) + np.sin(n * 2 * np.pi * 26 - 50) * 2
    w_vals = [5, 11, 26, 30]

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    axs[1].set_xlabel("Real")
    axs[1].set_ylabel("Imaginary")
    axs[0].set_xlim([0, 1])
    lim = max(np.max(np.abs(signal.real)), np.max(np.abs(signal.imag))) * 1.1
    lims = [-lim, lim]
    axs[1].set_xlim(lims)
    axs[1].set_ylim(lims)
    axs[0].axhline(color="black")
    axs[1].axhline(color="black")
    axs[1].axvline(color="black")
    axs[1].set(aspect="equal")
    fig.tight_layout()
    axs[0].plot(n, signal, color="green")
    shape = signal * np.e ** (-2 * np.pi * 1j * n)
    axs[1].plot(shape.real, shape.imag)
    stem_idx = int(N * 0.21)
    axs[0].plot([n[stem_idx]] * 2, [signal[stem_idx], 0], "r-o", markevery=2)
    axs[1].plot([shape.real[stem_idx], 0], [shape.imag[stem_idx], 0], "r-o", markevery=2)
    fig.subplots_adjust(left=0.12)
    savefig("Lab 3 - 2 Fig 1")
    plt.close()

    fig, axs = plt.subplots(nrows=2, ncols=2)
    for i, w in enumerate(w_vals):
        ax = axs[i // 2][i % 2]
        ax.set_title(f"w = {w}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.axhline(color="black")
        ax.axvline(color="black")
        ax.set(aspect="equal")
    fig.suptitle("Signal contains 5 Hz, 11 Hz, and 26 Hz frequencies.")
    fig.tight_layout(pad=0, h_pad=0.5)
    fig.subplots_adjust(top=0.85)
    for i, w in enumerate(w_vals):
        ax = axs[i // 2][i % 2]
        shape = signal * np.e ** (-2 * np.pi * 1j * w * n)
        distance = (1 - (np.abs(shape[1:] / lim) * 0.7 + 0.3))
        colors = plt.colormaps["ocean"](distance)
        coords = np.column_stack((shape.real, shape.imag))
        assert coords.shape == (N, 2)
        size = coords.itemsize
        assert coords.strides == (size * 2, size)
        segments = np.lib.stride_tricks.as_strided(coords, (N - 1, 2, 2), (size * 2, size * 2, size), writeable=False)
        assert segments.size == colors.size
        col = LineCollection(segments=segments, colors=colors)
        ax.add_collection(col)
        # plot center
        ax.plot([coords[:, 0].mean(), 0], [coords[:, 1].mean(), 0], color="darkblue", marker="o", linestyle="solid", linewidth=2, markevery=2)
    savefig("Lab 3 - 2 Fig 2")
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    axs[1].set_xlabel("Frequency")
    axs[1].set_ylabel("Amplitude")
    fig.tight_layout()
    row = np.arange(0, N)
    col = np.arange(0, N)[:, np.newaxis]
    w = np.e ** (-2 * np.pi * 1j / N)
    F = (1 / N) * w ** (row * col)
    F_inv = F.conj().T * N
    assert np.allclose(F @ F_inv, np.identity(N))
    signal = -5 + np.sin(2 * np.pi * n * 3 - 0.2) * 3 + np.sin(2 * np.pi * n * 5 + 3) * 0.3 + np.sin(2 * np.pi * n * 9 + 0.6) * 9
    freq = signal @ F
    assert np.allclose(signal, freq @ F_inv)
    # almost all bins in the DFT are 0 since the signal is low frequency, so display only up to 10 Hz
    max_freq = 11
    amplitudes = np.abs(freq[:max_freq])
    amplitudes[1:] *= 2
    axs[0].plot(n, signal)
    axs[1].stem(np.arange(max_freq), amplitudes)
    savefig("Lab 3 - 3")
    plt.close()

if __name__ == "__main__":
    main()
