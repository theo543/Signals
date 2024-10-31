import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig

def find_sin_from_samples(samples):
    fft = np.fft.fft(samples)
    amplitudes = np.abs(fft)
    for i in range(1, samples.size):
        if np.allclose(amplitudes[i], samples.size / 2):
            return i, amplitudes[i] * 2 / samples.size, np.angle(fft[i]) + np.pi / 2 # DFT gives -pi/2 for phase of sin
    raise ValueError("No sin found in samples.")

def main():
    fig, axs = plt.subplots(nrows=3, ncols=2)
    time = np.linspace(0, 1, num=1_000_000)
    small_sampling_freq = 3
    signal_freqs = [small_sampling_freq * i - 1 for i in range(1, 3 + 1)]
    large_sampling_freq = max(signal_freqs) * 2 + 1
    for i, sample_freq in enumerate([small_sampling_freq, large_sampling_freq]):
        sample_time = np.arange(0, 1, step=1 / sample_freq)
        for j, freq in enumerate(signal_freqs):
            ax = axs[j][i]
            ax.set_title(f"{freq} Hz sampled at {sample_freq} Hz")
            signal = np.sin(2 * np.pi * time * freq + 0.1)
            samples = np.sin(2 * np.pi * sample_time * freq + 0.1)
            ax.plot(time, signal)
            ax.plot(sample_time, samples, "o")
            frequency, amplitude, phase = find_sin_from_samples(samples)
            ax.plot(time, np.sin(2 * np.pi * time * frequency + phase) * amplitude)
    fig.legend(["Signal", "Samples", "Reconstructed Signal"], loc="lower center", ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    savefig("Lab 4 - 2 3")

if __name__ == "__main__":
    main()
