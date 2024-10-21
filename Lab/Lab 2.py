import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig

def square(t):
    return np.mod(np.floor(t), 2) * 2 - 1

def main():
    t = 0.1
    time = np.linspace(0, t, num=1_000_000)
    freq = 60
    ampl = 311
    cos_phase = - (np.pi ** 2) * freq
    x = np.sin(time * 2 * np.pi * freq) * ampl
    y = np.cos(time * 2 * np.pi * freq + cos_phase) * ampl
    _, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, x)
    ax[1].plot(time, y)
    ax[2].plot(time, x)
    ax[2].plot(time, y)
    savefig("Lab 2 - 1")
    plt.close()

    phases = np.array([0, 1, 2, 3]) * (np.pi ** 2) * freq
    snr = [0.1, 1, 10, 1000]
    y_vals = []
    for snr, phase in zip(snr, phases):
        x = np.sin(time * 2 * np.pi * freq + phase)
        z = np.random.normal(size=time.shape)
        y = np.sqrt(np.linalg.norm(x) ** 2 / (snr * np.linalg.norm(z) ** 2))
        y_vals.append(f"SNR = {snr}, γ = {y}")
        plt.plot(time, x + y * z)
    plt.legend(y_vals)
    savefig("Lab 2 - 2")
    plt.close()

    x = np.sin(time * 2 * np.pi * freq)
    y = square(time * 2 * np.pi * freq)
    _, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, x)
    ax[1].plot(time, y)
    ax[2].plot(time, x + y)
    savefig("Lab 2 - 4")
    plt.close()

    sampling_freq = 1000
    sample_times = np.arange(0, t, 1 / sampling_freq)
    ratios = [2, 4, np.inf]
    _, ax = plt.subplots(nrows=len(ratios), ncols=1)
    for i, ratio in enumerate(ratios):
        sig = np.sin(time * 2 * np.pi * sampling_freq / ratio)
        sig_sampled = np.sin(sample_times * 2 * np.pi * sampling_freq / ratio)
        ax[i].plot(time, sig)
        ax[i].stem(sample_times, sig_sampled)
    savefig("Lab 2 - 6")
    plt.close()
    # Observations:
    # Signal 1 is always zero when it is sampled.
    # Signal 2 is sampled perfectly. All peaks and zeroes of the signal are sampled.
    # The samples of signal 1 are identical to the samples of signal 3, so there's no way to tell those signals apart without changing the sampling rate.

    t = 1
    time = np.linspace(0, t, num=1_000_000)
    sample_times = np.arange(0, t, 1 / sampling_freq)
    freq = 10
    x = np.sin(time * 2 * np.pi * freq)
    x_s = np.sin(sample_times * 2 * np.pi * freq)
    deci_idx = np.arange(0, x_s.size, 4)
    _, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time[:time.size//4], x[:x.size//4])
    ax[1].stem(sample_times[:sample_times.size//4], x_s[deci_idx])
    deci_idx = np.arange(2, x_s.size, 4)
    ax[2].stem(sample_times[:sample_times.size//4], x_s[deci_idx])
    savefig("Lab 2 - 7")
    plt.close()
    # Observations:
    # Keeping every 4th sample multiplies the frequency by 4.
    # The starting index doesn't matter. Signals 1 and 2 (starting at indexes 0 and 2) look the same.

    a = np.linspace(-np.pi/2, np.pi/2, num=1_000_000)
    sin = np.sin(a)
    err = np.abs(sin - a)
    pade_approx = (a - 7 * (a ** 3) / 60) / (1 + (a ** 2) / 20)
    pade_err = np.abs(sin - pade_approx)
    fig, ax = plt.subplots(nrows=2, ncols=4)
    fig.tight_layout()
    for row in [0, 1]:
        if row == 1:
            for col in [0, 1, 2, 3]:
                ax[row][col].set_yscale('log')
        ax[row][0].plot(a, sin)
        ax[row][0].plot(a, a)
        ax[row][1].plot(a, err)
        ax[row][2].plot(a, sin)
        ax[row][2].plot(a, pade_approx)
        ax[row][3].plot(a, pade_err)
    savefig("Lab 2 - 8")
    plt.close()

if __name__ == "__main__":
    main()
