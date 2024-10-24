import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from savefig import savefig

def draw_winding_plot(w_freq, signal, n, ax, colormap=plt.colormaps["ocean"], plot_center=True, stem_idx=None):
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    lim = np.max(np.abs(signal)) * 1.1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.axhline(color="black")
    ax.axvline(color="black")
    ax.set(aspect="equal")

    shape = signal * np.e ** (-2 * np.pi * 1j * w_freq * n)

    if colormap is not None:
        distance = (1 - (np.abs(shape[1:] / lim) * 0.7 + 0.3))
        colors = colormap(distance)

        coords = np.column_stack((shape.real, shape.imag))
        assert coords.shape == (n.size, 2)
        size = coords.itemsize
        assert coords.strides == (size * 2, size)
        segments = np.lib.stride_tricks.as_strided(coords, (n.size - 1, 2, 2), (size * 2, size * 2, size), writeable=False)

        col = LineCollection(segments=segments, colors=colors)
        ax.add_collection(col)
    else:
        ax.plot(shape.real, shape.imag)

    if plot_center:
        center = np.mean(shape)
        ax.plot([center.real, 0], [center.imag.mean(), 0], color="darkblue", marker="o", linestyle="solid", linewidth=3, markevery=2, solid_capstyle="round")

    if stem_idx is not None:
        ax.plot([shape.real[stem_idx], 0], [shape.imag[stem_idx], 0], "r-o", markevery=2)

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
    signal = -1 + np.sin(2 * np.pi * n * 3 - 0.2) * 2 + np.sin(2 * np.pi * n * 5 + 3) * 1.5 + np.sin(2 * np.pi * n * 9 + 0.6) * 4
    signal_formula = r"$y = -1 + 2\sin(2\pi x \cdot 3 - 0.2) + 1.5\sin(2\pi x \cdot 5 + 3) + 4\sin(2\pi x \cdot 9 + 0.6)$"
    w_vals = [3, 5, 9, 0, 4, 6]

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim([0, 1])
    axs[0].axhline(color="black")
    stem_idx = int(N * 0.21)
    draw_winding_plot(1, signal, n, axs[1], plot_center=False, colormap=None, stem_idx=stem_idx)
    fig.tight_layout()
    axs[0].plot(n, signal, color="green")
    axs[0].plot([n[stem_idx]] * 2, [signal[stem_idx], 0], "r-o", markevery=2)
    fig.subplots_adjust(left=0.12)
    savefig("Lab 3 - 2 Fig 1")
    plt.close()

    fig, axs = plt.subplots(nrows=2, ncols=3)
    for i, w in enumerate(w_vals):
        ax = axs[i // 3][i % 3]
        ax.set_title(f"$\\omega = {w}$")
        draw_winding_plot(w, signal, n, ax)
    fig.suptitle(signal_formula)
    fig.tight_layout(pad=0, h_pad=3.5)
    fig.subplots_adjust(top=0.87, bottom=0.1)
    savefig("Lab 3 - 2 Fig 2")
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    axs[0].axhline(color="black")
    axs[1].set_xlabel("Frequency")
    axs[1].set_ylabel("Amplitude")
    fig.tight_layout()
    row = np.arange(0, N)
    col = np.arange(0, N)[:, np.newaxis]
    w = np.e ** (-2 * np.pi * 1j / N)
    F = (1 / N) * w ** (row * col)
    F_inv = F.conj().T * N
    assert np.allclose(F @ F_inv, np.identity(N))
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
