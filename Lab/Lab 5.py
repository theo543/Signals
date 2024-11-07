from pathlib import Path
from datetime import datetime
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

# (h)
# -- Ideas --
# Weeks can be detected by filtering frequencies higher then one per day.
# Dividing the signal into years could by done by detecting holidays which are always on the same date,
# or if different months look different in the signal (i.e. maybe less cars in summer because of no school?)
# By checking which weekday the years start on and by looking for holidays which are on different dates each year, the specific year could be found.
# Besides holidays, other major events such as natural disasters, national elections, etc, could be detected.
# -- Potential problems --
# Looking for holidays and major events could be difficult since they don't cause sine waves. Something other than DFT might be needed.
# The slow increase in traffic could cause confusion.
# Looking for holidays and major events requires knowing which country the signal is from, or having a large database.
# Factors which could complicate the signal include: the local economy, road construction, local events (sports games?) and holidays, local weather, etc.
# Significant research might be needed to create a reliable algorithm to find the date with high confidence from traffic data taken from an unknown location.

def filter_freq(data: np.ndarray, div: int):
    max_idx = data.size // div
    fft = np.fft.fft(data)
    fft[max_idx+1:-max_idx] = 0
    assert np.allclose(fft[1:], np.flip(fft[1:]).conj())
    filtered = np.fft.ifft(fft).real
    return filtered

DATA = Path("Lab 5 Data.csv")

def format_date(expected_idx: int, line: str):
    [idx, date, _value] = line.split(",")
    assert expected_idx == int(idx)
    date_with_weekday = datetime.strptime(date, r"%d-%m-%Y %H:%M").strftime(r"%d-%m-%Y %H:%M %A")
    return f"Sample {idx} was on {date_with_weekday}."

# Sample rate is 24 times per day, so once a day is sample rate / 24.
# Filter frequencies higher than once per day.
# They are likely caused by either short-term traffic patterns or noise, so are irrelevant to long-term prediction.
FILTER_DAY = 24

def main():
    data = np.genfromtxt(DATA, delimiter=",", skip_header=1, usecols=(2,))

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

    month_start = 5256
    month_end = 5975

    # Making sure the month starts on a Monday.
    dates = DATA.read_text(encoding="ascii").splitlines()
    dates = [format_date(idx, dates[idx + 1]) for idx in [month_start, month_end, month_end + 1]]
    insert_comment("\n".join(dates))
    # Sample 5256 was on 01-04-2013 00:00 Monday.
    # Sample 5975 was on 30-04-2013 23:00 Tuesday.
    # Sample 5976 was on 01-05-2013 00:00 Wednesday.

    month = data[month_start:month_end + 1]
    plt.plot(month)
    plt.plot(filter_freq(month, FILTER_DAY))
    days_formatter = FuncFormatter(lambda x, _: f"day {x / 24 + 1:.0f}")
    plt.gca().xaxis.set_major_formatter(days_formatter)
    plt.legend(["Month", "Filtered Month"])
    savefig("Lab 5 - month")
    plt.close()

    plt.plot(filter_freq(data, FILTER_DAY))
    savefig("Lab 5 - filtered")

if __name__ == "__main__":
    main()
