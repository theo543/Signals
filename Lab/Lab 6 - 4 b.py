import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, cheby1, filtfilt
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
    plt.close()

    # The cutoff frequency should be once per day, to remove noise from traffic fluctuations.
    # once per day = 1 / 86400 = 0.1157 * 10^-4
    # Nyquist frequency = sampling frequency / 2 = (24 times per day) / 2 = 12 times per day
    # once per day = 1/12 * Nyquist frequency

    ORDER = 5
    RIPPLE = 5 # dB
    CUTOFF = 1/12

    butter_b, butter_a = butter(ORDER, CUTOFF)
    cheby1_b, cheby1_a = cheby1(ORDER, RIPPLE, CUTOFF)

    #fig, axs = plt.subplots(nrows=3, ncols=1)
    plt.stem(data)
    plt.plot(filtfilt(cheby1_b, cheby1_a, data), color="orange")
    plt.plot(filtfilt(butter_b, butter_a, data), color="green")
    plt.legend(["Chebyshev", "Butter", "Raw Data"])
    fig.tight_layout()
    savefig("Lab 6 - e")
    # Both look similar.
    # The Butter filter seems like it shows a bit more detail.
    plt.close()

    # With these parameters, the Chevyshev filter looks good too.
    ORDER_2 = 3
    RIPPLE_2 = 1 # dB

    cheby1_b_2, cheby1_a_2 = cheby1(ORDER_2, RIPPLE_2, CUTOFF)

    plt.plot(data)
    plt.plot(filtfilt(cheby1_b_2, cheby1_a_2, data))
    savefig("Lab 6 - f")

if __name__ == "__main__":
    main()
