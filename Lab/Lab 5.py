import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from replace_me import insert_comment
from savefig import savefig

# (a)
# The frequency is once per hour = 1/3600 Hz = 277.8 uHz

# (b)
# The samples are from 25-08-2012 00:00 to 25-09-2014 23:00
# Two years and one month

# (c)
# The maximum possible frequency is once per two hours = 1/7200 Hz = 138.9 uHz

def main():
    data = np.genfromtxt("Lab 5 Data.csv", delimiter=",", skip_header=1, usecols=(2,))#[888:1631]

    plt.plot(data)
    savefig("Lab 5 - signal")
    plt.close()

    mean = np.mean(data)
    data -= mean

    insert_comment(f"Mean value = {mean:.3f}")
    # Mean value = 138.958

    fft = np.fft.fft(data)
    amplitudes = np.abs(fft)[:data.size//2] / data.size
    plt.plot(amplitudes)

    sample_rate_days = 24
    sample_rate_hz = 1/3600
    sample_rate_uhz = sample_rate_hz * 1000000

    uhz_formatter = FuncFormatter(lambda x, _: f"{x * sample_rate_uhz / data.size:.4f} μHz")
    plt.gca().xaxis.set_major_formatter(uhz_formatter)
    savefig("Lab 5 - signal DFT")
    plt.close()

    top_nr = 4
    top_freq_result = f"Top {top_nr} frequencies:"
    top_indexes = np.argsort(amplitudes)[-top_nr:][::-1]
    for i, idx in enumerate(top_indexes):
        freq_uhz = idx / data.size * sample_rate_uhz
        days_period = 1 / (idx * sample_rate_days / data.size)
        top_freq_result += f"\n{i + 1}. bin {idx: >3}, frequency {freq_uhz:6.3f} μHz, period {days_period: >3.0f} days, amplitude {amplitudes[idx]:.3f}"

    insert_comment(top_freq_result)
    # Top 4 frequencies:
    # 1. bin   1, frequency  0.015 μHz, period 762 days, amplitude 66.854
    # 2. bin   2, frequency  0.030 μHz, period 381 days, amplitude 35.219
    # 3. bin 762, frequency 11.574 μHz, period   1 days, amplitude 27.102
    # 4. bin   3, frequency  0.046 μHz, period 254 days, amplitude 25.220

    # Frequency 3 is likely from the day-night cycle, more cars pass through at day.
    # Frequency 1 might be caused by the slow increase of the signal.
    # Frequencies 2 and 4 might be from the year cycle (perhaps in some seasons there are more cars), or also due to the increase, or both.

    # Filter frequencies higher than once per day
    max_idx = data.size // 24 + 1
    fft[max_idx:-max_idx] = 0
    filtered_data = np.fft.ifft(fft).real
    plt.plot(filtered_data)
    savefig("Lab 5 - filtered")

if __name__ == "__main__":
    main()
