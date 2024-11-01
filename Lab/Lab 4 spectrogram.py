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

def windowed_view(array: np.ndarray, window_size: int, window_stride: int) -> np.ndarray:
    assert array.shape == (array.size,)
    assert len(array.strides) == 1
    # offset not guaranteed to equal array.itemsize
    # for the vowels recording, the two channels are interleaved
    offset = array.strides[0]
    total_windows = (array.size - window_size + 1) // window_stride
    shape = (total_windows, window_size)
    strides = (window_stride * offset, offset)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides, writeable=False)

def generate_spectrogram(file: Path):
    sampling_rate, sound = wavfile.read(file)

    if sound.shape == (sound.size // 2, 2):
        sound = sound[:, 0]
    else:
        assert sound.shape == (sound.size,)

    window_size = sound.size // 100
    window_stride = window_size // 2
    windows = windowed_view(sound, window_size, window_stride)
    ffts = np.zeros(shape=(windows.shape[0], windows.shape[1] // 2))
    for i, window in enumerate(windows):
        fft = np.fft.fft(window)
        ffts[i, :] = 10 * np.log10(np.abs(fft[: window.size // 2]))
    plt.pcolormesh(ffts.T)

    def set_ticks(fn, max_position, max_value, value_step, suffix):
        values = range(0, max_value, value_step)
        ticks = [(max_position * value) // max_value for value in values]
        labels = [str(value) + suffix for value in values]
        fn(ticks, labels)

    left, right = plt.xlim()
    bottom, top = plt.ylim()
    assert bottom == 0
    assert left == 0
    max_freq = sampling_rate // 2
    duration = sound.size // sampling_rate
    set_ticks(plt.xticks, right, duration, 1, " sec")
    set_ticks(plt.yticks, top, max_freq, 2000, " Hz")
    plt.tight_layout()
    savefig(file.with_suffix(""))
    plt.close()

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
