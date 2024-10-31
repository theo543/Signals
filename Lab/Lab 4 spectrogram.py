from pathlib import Path
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.integrate
from scipy.io import wavfile
from savefig import savefig

# spectrogram SVGs are way too big (~80 MB)
# they cause an error in VSCode and take extremely long to load in the browser
savefig = partial(savefig, svg=False)

def square(time):
    _, whole = np.modf(time)
    return whole % 2

def triangle(time):
    fractional, _ = np.modf(time + 0.5)
    return np.abs(1 - fractional * 2)

TEST_FILE = Path("Lab 4 - test signal.wav")
RECORDING = Path("Lab 4 - vowels.wav")

def generate_test_signal():
    rng = np.random.default_rng(seed=0)
    sample_rate = 48_000
    time = np.arange(0, 10, step=1 / sample_rate)
    start_freq = 600
    end_freq = 1100
    amplitude = 30000
    sweep_speed = 0.5
    freqs = triangle(time / sweep_speed) * (end_freq - start_freq) + start_freq + rng.random(size=time.shape) * (end_freq - start_freq) / 2
    warped_time = scipy.integrate.cumulative_trapezoid(freqs, time, initial=0)
    signal = np.sin(2 * np.pi * warped_time) * amplitude + rng.random(size=time.shape) * (amplitude / 30)
    wavfile.write(TEST_FILE, rate=sample_rate, data=signal.astype(np.int16))

def generate_spectrogram(file: Path):
    sampling_rate, sound = wavfile.read(file)
#    plt.specgram(sound, Fs=sampling_rate)
#    plt.show()
#    print(sampling_rate, sound.size, sound.size // 100)
    window_size = sound.size // 100
    assert sound.shape == (sound.size,)
    size = sound.itemsize
    windows = np.lib.stride_tricks.as_strided(sound, shape=(sound.size // window_size * 2 - 1, window_size), strides=(window_size // 2 * size, size), writeable=False)
    ffts = np.zeros(shape=(windows.shape[0], windows.shape[1] // 2))
    for i, window in enumerate(windows):
        fft = np.fft.fft(window)
        ffts[i, :] = 10 * np.log10(np.abs(fft[: window.size // 2]))
    plt.pcolormesh(ffts.T)
    #plt.xticks([0, 1, 3, 4, 5, 7, 8])
    ticks, _labels = plt.xticks()
    seconds = sound.size // sampling_rate
    left, right = plt.xlim()
    assert left == 0
    plt.xticks(ticks[:-1], ["0"] + [f"{min(seconds, (tick / right) * seconds):1.1f}" for tick in ticks[1:-1]])
    plt.tight_layout()
    savefig(file.with_suffix(""))

def main():
    if not TEST_FILE.exists():
        generate_test_signal()
    generate_spectrogram(TEST_FILE)
    if not RECORDING.exists():
        print("Recording file does not exist.")
    else:
        generate_spectrogram(RECORDING)

if __name__ == "__main__":
    main()
